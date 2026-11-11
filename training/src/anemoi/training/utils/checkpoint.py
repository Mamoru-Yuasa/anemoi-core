# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import io
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch_geometric.data import HeteroData

from anemoi.models.distributed.khop_edges import sort_edge_index_by_dst
from anemoi.models.migrations import Migrator
from anemoi.models.preprocessing.imputer import BaseImputer
from anemoi.training.utils.variables_metadata import extract_variables_metadata_from_checkpoint
from anemoi.utils.checkpoints import save_metadata

chunking_fix_migration = importlib.import_module("anemoi.models.migrations.scripts.1762857428_chunking_fix").migrate
trainable_edge_perm_fix_migration = importlib.import_module(
    "anemoi.models.migrations.scripts.1779202136_trainable_edge_perm_fix",
).migrate

LOGGER = logging.getLogger(__name__)


def clear_imputer_runtime_state(model: torch.nn.Module) -> None:
    """Clear imputer state before serialising a model."""
    for module in model.modules():
        if isinstance(module, BaseImputer):
            module.nan_locations = None
            module.loss_mask_training = None


def _filter_state_dict_size_mismatches(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    for key in list(state_dict):
        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
            LOGGER.info("Skipping loading parameter: %s", key)
            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))

            del state_dict[key]


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


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    from anemoi.training.train.methods.base import BaseTrainingModule

    module = BaseTrainingModule.load_from_checkpoint(lightning_checkpoint_path, weights_only=False)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    clear_imputer_runtime_state(model)
    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath


_DATASET_PARENTS = "encoder|decoder|encoder_graph_provider|decoder_graph_provider|trainable_tensors"


def map_checkpoint_datasets(checkpoint: dict, dataset_map: dict[str, str]) -> dict:
    """Give model datasets the network weights of differently named checkpoint datasets.

    ``dataset_map`` maps a dataset of the model to a dataset of the checkpoint, e.g.
    ``{"gsm": "era5", "msm": "era5"}`` starts the GSM and MSM encoders, decoders and
    graph-specific trainables from those of an ERA5-only checkpoint. One checkpoint dataset
    may feed several model datasets. Only the network is mapped (encoders, decoders, their
    graph providers and trainable node features); the normalisers keep the statistics of the
    model's own datasets. The variable lists are mapped as well, so the usual check that the
    variables agree applies to the mapped pairs.
    """
    if not dataset_map:
        return checkpoint
    state_dict = checkpoint["state_dict"]
    for model_name, ckpt_name in dataset_map.items():
        pattern = re.compile(rf"(\.(?:{_DATASET_PARENTS})\.){re.escape(ckpt_name)}(\.)")
        copied = 0
        for key in [k for k in state_dict if pattern.search(k)]:
            new_key = pattern.sub(rf"\g<1>{model_name}\g<2>", key)
            if new_key != key:
                state_dict[new_key] = state_dict[key]
                copied += 1
        if not copied:
            raise ValueError(f"Checkpoint has no network weights for dataset {ckpt_name!r}")
        LOGGER.info("Mapped %d checkpoint tensors of dataset %r to model dataset %r", copied, ckpt_name, model_name)

    hyper = checkpoint.get("hyper_parameters", {})
    data_indices = hyper.get("data_indices")
    if isinstance(data_indices, dict):
        for model_name, ckpt_name in dataset_map.items():
            data_indices[model_name] = data_indices[ckpt_name]
    dataset_metadata = hyper.get("metadata", {}).get("dataset", {})
    for model_name, ckpt_name in dataset_map.items():
        if ckpt_name in dataset_metadata:
            dataset_metadata[model_name] = dataset_metadata[ckpt_name]
    return checkpoint


def transfer_learning_loading(
    model: torch.nn.Module,
    ckpt_path: Path | str,
    dataset_map: dict[str, str] | None = None,
) -> nn.Module:
    # Load the checkpoint
    # Load to CPU explictly, to avoid loading entire model on GPU initially
    # Modifications to the model occur on cpu,
    # The model will be sent to GPU when trainer.fit() is called
    LOGGER.debug("Loading checkpoint to device: cpu")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # apply chunking migration (fails silently otherwise leading to hard to debug issues)
    # this is due to loading with strict=False, planning to make this more robust in the future
    checkpoint = chunking_fix_migration(checkpoint)

    # Datasets renamed between checkpoint and model (e.g. ERA5 pre-training -> GSM fine-tuning).
    checkpoint = map_checkpoint_datasets(checkpoint, dict(dataset_map or {}))

    # Refresh processor stats from the current dataset if configured.
    model._update_checkpoint_state_dict_for_load(checkpoint)

    state_dict = checkpoint["state_dict"]
    model_state_dict = model.state_dict()

    # Graph-specific trainables follow nodes and edges, not tensor shapes.
    ckpt_graph = checkpoint.get("hyper_parameters", {}).get("graph_data")
    model_graph = getattr(model, "hparams", {}).get("graph_data")
    if ckpt_graph is not None and model_graph is not None:
        hidden_name = model.config.model.model.hidden_nodes_name
        remap_graph_trainables(state_dict, model_state_dict, ckpt_graph, model_graph, hidden_name)

    # Filter out layers with size mismatch
    _filter_state_dict_size_mismatches(state_dict, model_state_dict)

    # Runtime migration: the graph-provider permutation depends on instantiated provider state.
    checkpoint = trainable_edge_perm_fix_migration(checkpoint, model)
    state_dict = checkpoint["state_dict"]

    # Load the filtered state_dict into the model
    model.load_state_dict(state_dict, strict=False)

    ## Needed for data indices check
    data_indices = checkpoint["hyper_parameters"]["data_indices"]

    if isinstance(data_indices, dict):
        # New format: data_indices is always a dict in new code (even for single-dataset)
        LOGGER.info("Loading checkpoint with datasets: %s", list(data_indices.keys()))
        model._ckpt_model_name_to_index = {
            dataset_name: indices.name_to_index for dataset_name, indices in data_indices.items()
        }
    else:
        # Old format: data_indices is a single IndexCollection object (not dict)
        msg = (
            f"Checkpoint at '{ckpt_path}' was created with an older version of anemoi-core "
            "that does not support multi-dataset training. This checkpoint is incompatible "
            "with transfer learning in the current version."
        )
        raise TypeError(msg)

    # Extract variables_metadata for unit compatibility check
    model._ckpt_variables_metadata = extract_variables_metadata_from_checkpoint(
        checkpoint,
        model._ckpt_model_name_to_index,
    )

    return model


def freeze_submodule_by_name(module: nn.Module, target_name: str, base_target: str = "") -> bool:
    """Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze. Examples: "encoder", "encoder.lam".
    base_target : str
        Used for logging to show the full path of the current module being checked. Should not be set by the user.

    Returns
    -------
    bool
        True if the target submodule was found and frozen, False otherwise.
    """
    are_submodules_found = False
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                LOGGER.info("Freezing parameter %s: %s", base_target + name, param.shape)
                param.requires_grad = False
            are_submodules_found = True
        elif target_name.startswith(name + "."):
            new_target = target_name.replace(name + ".", "", 1)
            is_found = freeze_submodule_by_name(child, new_target, base_target=base_target + name + ".")
            are_submodules_found = are_submodules_found or is_found
        else:
            LOGGER.debug("Skipping submodule (looking for %s): %s", base_target + target_name, name)
    return are_submodules_found


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
