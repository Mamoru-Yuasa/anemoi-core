# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Starting renamed datasets from the weights of a checkpoint dataset."""

import pytest
import torch

from anemoi.training.utils.checkpoint import map_checkpoint_datasets


def _checkpoint():
    state_dict = {
        "model.model.encoder.era5.emb_nodes_src.weight": torch.ones(2),
        "model.model.decoder.era5.node_data_extractor.1.weight": torch.full((2,), 2.0),
        "model.model.encoder_graph_provider.era5.trainable.trainable": torch.full((2,), 3.0),
        "model.model.node_attributes.trainable_tensors.era5.trainable": torch.full((2,), 4.0),
        "model.model.node_attributes.latlons_era5": torch.zeros(2),
        "model.pre_processors.era5.processors.normalizer._norm_mul": torch.full((2,), 5.0),
        "model.model.processor.proc.0.lin_key.weight": torch.full((2,), 6.0),
    }
    return {
        "state_dict": state_dict,
        "hyper_parameters": {"data_indices": {"era5": "indices"}, "metadata": {"dataset": {"era5": {"a": 1}}}},
    }


def test_network_weights_are_copied_to_every_mapped_dataset():
    checkpoint = map_checkpoint_datasets(_checkpoint(), {"gsm": "era5", "msm": "era5"})
    state = checkpoint["state_dict"]
    for name in ("gsm", "msm"):
        assert state[f"model.model.encoder.{name}.emb_nodes_src.weight"][0] == 1
        assert state[f"model.model.decoder.{name}.node_data_extractor.1.weight"][0] == 2
        assert state[f"model.model.encoder_graph_provider.{name}.trainable.trainable"][0] == 3
        assert state[f"model.model.node_attributes.trainable_tensors.{name}.trainable"][0] == 4
        # Normalisers and coordinates stay with the model's own datasets.
        assert f"model.pre_processors.{name}.processors.normalizer._norm_mul" not in state
        assert f"model.model.node_attributes.latlons_{name}" not in state
        assert checkpoint["hyper_parameters"]["data_indices"][name] == "indices"
        assert checkpoint["hyper_parameters"]["metadata"]["dataset"][name] == {"a": 1}
    # The shared processor and the original names are left as they are.
    assert "model.model.processor.proc.0.lin_key.weight" in state
    assert "model.model.encoder.era5.emb_nodes_src.weight" in state


def test_unknown_checkpoint_dataset_is_an_error():
    with pytest.raises(ValueError):
        map_checkpoint_datasets(_checkpoint(), {"gsm": "ifs"})


def test_empty_map_leaves_the_checkpoint_unchanged():
    checkpoint = _checkpoint()
    keys = set(checkpoint["state_dict"])
    assert set(map_checkpoint_datasets(checkpoint, {})["state_dict"]) == keys
