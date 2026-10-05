# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import io
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch_geometric.data import HeteroData

from anemoi.graphs.projection_helpers import DEFAULT_DATASET_NAME
from anemoi.graphs.projection_helpers import uses_fused_dataset_graph
from anemoi.graphs.utils import load_graph_from_file
from anemoi.graphs.utils import validate_loaded_graph
from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.migrations import Migrator
from anemoi.models.preprocessing.imputer import BaseImputer
from anemoi.utils.checkpoints import save_metadata

LOGGER = logging.getLogger(__name__)


def clear_imputer_runtime_state(model: torch.nn.Module) -> None:
    """Clear imputer state before serialising a model."""
    for module in model.modules():
        if isinstance(module, BaseImputer):
            module.nan_locations = None
            module.loss_mask_training = None


_PROVIDER_KEY = re.compile(
    r"^(?P<path>.*\.(?P<kind>encoder|decoder|processor)_graph_provider(?:\.(?P<dataset>[^.]+))?)\.trainable\.trainable$",
)
_NODE_KEY = re.compile(r"^.*\.node_attributes\.trainable_tensors\.(?P<nodes>[^.]+)\.trainable$")


def _coordinate_keys(coords: torch.Tensor) -> np.ndarray:
    """Exact row keys for node coordinates (radians), robust to float noise."""
    return np.round(coords.detach().cpu().double().numpy() * 1e8).astype(np.int64)


def node_row_keys(graph: HeteroData, nodes: str) -> np.ndarray:
    """One key per node of ``nodes``: its coordinates, in node order."""
    return _coordinate_keys(graph[nodes].x)


def edge_row_keys(graph: HeteroData, edge_type: tuple[str, str, str]) -> np.ndarray:
    """One key per edge: source and destination coordinates, in the trainable layout.

    Graph providers store edge parameters with the edges sorted by destination
    (``sort_edge_index_by_dst``), so the keys follow that order.
    """
    src, _, dst = edge_type
    edge_index, _ = sort_edge_index_by_dst(graph[edge_type].edge_index, max_value=graph[dst].num_nodes)
    edge_index = edge_index.cpu()
    return np.concatenate(
        [_coordinate_keys(graph[src].x)[edge_index[0]], _coordinate_keys(graph[dst].x)[edge_index[1]]],
        axis=1,
    )


def remap_rows_by_keys(
    ckpt_rows: torch.Tensor,
    ckpt_keys: np.ndarray,
    model_rows: torch.Tensor,
    model_keys: np.ndarray,
) -> tuple[torch.Tensor, int]:
    """Take each model row from the checkpoint row with the same key; keep the model row otherwise."""
    lookup = {key.tobytes(): i for i, key in enumerate(ckpt_keys)}
    model_index, ckpt_index = [], []
    for i, key in enumerate(model_keys):
        j = lookup.get(key.tobytes())
        if j is not None:
            model_index.append(i)
            ckpt_index.append(j)
    rows = model_rows.detach().clone()
    if model_index:
        rows[torch.as_tensor(model_index)] = ckpt_rows[torch.as_tensor(ckpt_index)].to(rows.dtype)
    return rows, len(model_index)


def remap_graph_trainables(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    ckpt_graph: HeteroData,
    model_graph: HeteroData,
    hidden_name: str,
) -> None:
    """Transfer graph-specific trainable parameters by graph position, not by tensor shape.

    Trainable node and edge parameters belong to particular nodes and edges. When
    the graph changes between checkpoint and model (e.g. a uniform mesh refined
    into a stretched one), a tensor of the same shape can describe different
    edges, and one of a different shape can still share most of them. Each row
    is matched by its node coordinates (edges: source and destination); matched
    rows come from the checkpoint and the rest keep the model's initial values.
    Edge parameters are only remapped when the checkpoint uses the current
    destination-sorted layout.
    """
    for key in list(state_dict):
        if key not in model_state_dict:
            continue
        try:
            match = _PROVIDER_KEY.match(key)
            if match:
                layout_key = f"{match['path']}.trainable_layout_version"
                if layout_key in state_dict and int(state_dict[layout_key]) != int(model_state_dict[layout_key]):
                    continue
                dataset = match["dataset"]
                edge_type = {
                    "encoder": (dataset, "to", hidden_name),
                    "decoder": (hidden_name, "to", dataset),
                    "processor": (hidden_name, "to", hidden_name),
                }[match["kind"]]
                ckpt_keys, model_keys = edge_row_keys(ckpt_graph, edge_type), edge_row_keys(model_graph, edge_type)
            else:
                match = _NODE_KEY.match(key)
                if not match:
                    continue
                nodes = match["nodes"]
                ckpt_keys, model_keys = node_row_keys(ckpt_graph, nodes), node_row_keys(model_graph, nodes)
        except (KeyError, AttributeError, TypeError) as error:
            LOGGER.info("Not remapping %s by graph position: %s", key, error)
            continue

        if len(ckpt_keys) != state_dict[key].shape[0] or len(model_keys) != model_state_dict[key].shape[0]:
            LOGGER.info("Not remapping %s: graph and parameter sizes differ", key)
            continue
        if np.array_equal(ckpt_keys, model_keys):
            continue

        state_dict[key], matched = remap_rows_by_keys(state_dict[key], ckpt_keys, model_state_dict[key], model_keys)
        LOGGER.info(
            "Remapped %s by graph position: %d of %d rows from the checkpoint, the rest freshly initialised",
            key,
            matched,
            len(model_keys),
        )


def load_and_prepare_model(
    lightning_checkpoint_path: str,
    graph_path: Path | str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint.
    graph_path : Path | str | None, optional
        path to a graph file. If given, it is used instead of the ``graph_data`` stored in the
        checkpoint (which is then not required). The graph is not checked against the checkpoint
        beyond the presence of the required dataset nodes.

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata.

    """
    from hydra.utils import get_class

    # memory-mapped: only the hyperparameters are needed here, not the weights
    hyper_parameters = torch.load(
        lightning_checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["hyper_parameters"]

    overrides = {}
    if graph_path is not None:
        dataset_names = list(hyper_parameters["data_indices"])
        graph = load_graph_from_file(graph_path)
        fused = uses_fused_dataset_graph(graph, dataset_names)
        validate_loaded_graph(graph, dataset_names if fused else [DEFAULT_DATASET_NAME])
        overrides["graph_data"] = graph
    elif "graph_data" not in hyper_parameters:
        msg = (
            f"The checkpoint {lightning_checkpoint_path} does not contain the hyperparameter 'graph_data'. "
            "Supply a graph with --graph."
        )
        raise ValueError(msg)

    training_method_cls = get_class(hyper_parameters["config"].training.method._target_)

    module = training_method_cls.load_from_checkpoint(lightning_checkpoint_path, weights_only=False, **overrides)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(
    model: torch.nn.Module,
    metadata: dict,
    save_path: Path | str,
    supporting_arrays: dict | None = None,
) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model.
    metadata : dict
        Anemoi Metadata to inject into checkpoint.
    save_path : Path | str
        File path of the anemoi inference checkpoint to write.
    supporting_arrays : dict | None, optional
        Supporting arrays to inject into checkpoint.

    Returns
    -------
    Path
        Path to saved checkpoint.
    """
    inference_filepath = Path(save_path)

    clear_imputer_runtime_state(model)
    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata, supporting_arrays=supporting_arrays)
    return inference_filepath


class LoggingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> str:
        if "anemoi.training" in module:
            msg = (
                f"anemoi-training Pydantic schemas found in model's metadata: "
                f"({module}, {name}) Please review Pydantic schemas to avoid this."
            )
            raise ValueError(msg)
        return super().find_class(module, name)


def check_classes(model: torch.nn.Module) -> None:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    _ = LoggingUnpickler(buffer).load()


class RegisterMigrations(Callback):
    """Callback that register all existing migrations to a checkpoint before storing it."""

    def __init__(self):
        self.migrator = Migrator()

    def on_save_checkpoint(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        checkpoint: dict[str, Any],
    ) -> None:
        self.migrator.register_migrations(checkpoint)
