# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import datetime

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import MixedCadenceForecaster
from anemoi.training.utils.masks import NoOutputMask
from anemoi.utils.dates import frequency_to_timedelta


def _indices() -> dict[str, IndexCollection]:
    config = DictConfig({"forcing": ["forcing"], "diagnostic": [], "target": []})
    return {name: IndexCollection(config, {"state": 0, "forcing": 1}) for name in ("global", "regional")}


@pytest.mark.parametrize("regional_timestep,regional_outputs", [("3h", 2), ("1h", 6)])
def test_offsets_have_only_observed_dates(regional_timestep: str, regional_outputs: int) -> None:
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": regional_timestep},
        block_timestep="6h",
        multistep_input=2,
        rollout={"maximum": 2},
    )
    assert task.num_output_timesteps == {"global": 1, "regional": regional_outputs}
    # Training loads only the current rollout (one block); the longest rollout adds a second block.
    assert task.get_offsets("global", mode="training") == [datetime.timedelta(hours=hour) for hour in (-6, 0, 6)]
    assert task.get_offsets("global") == [datetime.timedelta(hours=hour) for hour in (-6, 0, 6, 12)]
    assert task.get_output_offsets("regional") == [
        datetime.timedelta(hours=hour) for hour in range(6 // regional_outputs, 7, 6 // regional_outputs)
    ]
    assert datetime.timedelta(hours=3) not in task.get_offsets("global", mode="training")


def test_rollout_uses_latest_native_steps_and_forcing() -> None:
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": "3h"},
        block_timestep="6h",
        multistep_input=2,
        rollout={"maximum": 2},
    )
    indices = _indices()
    batch = {
        "global": torch.zeros((1, 4, 1, 1, 2)),
        "regional": torch.zeros((1, 6, 1, 1, 2)),
    }
    batch["global"][0, :, 0, 0, 0] = torch.tensor([-6, 0, 6, 12])
    batch["regional"][0, :, 0, 0, 0] = torch.tensor([-3, 0, 3, 6, 9, 12])
    batch["global"][0, :, 0, 0, 1] = torch.tensor([-60, 0, 60, 120])
    batch["regional"][0, :, 0, 0, 1] = torch.tensor([-30, 0, 30, 60, 90, 120])

    x = task.get_inputs(batch, indices)
    target = task.get_targets(batch, rollout_step=0)
    assert target["global"][0, :, 0, 0, 0].tolist() == [6]
    assert target["regional"][0, :, 0, 0, 0].tolist() == [3, 6]

    predicted = {
        "global": torch.tensor([600.0]).reshape(1, 1, 1, 1, 1),
        "regional": torch.tensor([300.0, 600.0]).reshape(1, 2, 1, 1, 1),
    }
    next_x = task.advance_input(
        x,
        predicted,
        batch,
        data_indices=indices,
        output_mask={"global": NoOutputMask(), "regional": NoOutputMask()},
    )
    assert next_x["global"][0, :, 0, 0, 0].tolist() == [0, 600]
    assert next_x["regional"][0, :, 0, 0, 0].tolist() == [300, 600]
    assert next_x["global"][0, :, 0, 0, 1].tolist() == [0, 60]
    assert next_x["regional"][0, :, 0, 0, 1].tolist() == [30, 60]


def test_rejects_non_divisor_timestep() -> None:
    with pytest.raises(ValueError, match="divide"):
        MixedCadenceForecaster(dataset_timesteps={"regional": "4h"}, block_timestep="6h")


def test_rollout_curriculum_extends_loaded_blocks() -> None:
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": "3h"},
        block_timestep="6h",
        rollout={"start": 1, "epoch_increment": 1, "maximum": 2},
    )
    assert len(task.steps("training")) == 1
    task.on_train_epoch_end(0)
    assert len(task.steps("training")) == 2
    assert task.get_offsets("regional", mode="training")[-1] == datetime.timedelta(hours=12)
    assert task.training_runtime_state_dict()["rollout"]["step"] == 2



def test_metadata_stores_native_offsets_and_advance_map() -> None:
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": "3h"},
        block_timestep="6h",
        multistep_input=2,
    )
    md = {"metadata_inference": {"dataset_names": ["global", "regional"], "global": {}, "regional": {}}}
    task.fill_metadata(md)

    global_ts = md["metadata_inference"]["global"]["timesteps"]
    regional_ts = md["metadata_inference"]["regional"]["timesteps"]
    assert md["metadata_inference"]["task"] == "mixed-cadence-forecaster"
    assert global_ts["timestep"] == "6h"
    hours = lambda offsets: [frequency_to_timedelta(o) / datetime.timedelta(hours=1) for o in offsets]  # noqa: E731
    assert hours(global_ts["input_offsets"]) == [-6, 0]
    assert hours(global_ts["output_offsets"]) == [6]
    assert hours(regional_ts["output_offsets"]) == [3, 6]
    assert global_ts["rollout_shift"] == regional_ts["rollout_shift"] == "6h"
    # global: t-6h <- t (input 1), t <- t+6h (output 0); regional: both slots from outputs
    assert global_ts["advance_map"] == {"inin": [(1, 0)], "outin": [(0, 1)]}
    assert regional_ts["advance_map"] == {"inin": [], "outin": [(0, 0), (1, 1)]}


def _hours(offsets) -> list[float]:
    return [offset / datetime.timedelta(hours=1) for offset in offsets]


def test_regional_history_can_span_the_block() -> None:
    """F(G[t-6], G[t], R[t-6], R[t-3], R[t]) -> (G[t+6], R[t+3], R[t+6])."""
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": "3h"},
        block_timestep="6h",
        multistep_input={"global": 2, "regional": 3},
    )
    assert _hours(task.get_input_offsets("regional")) == [-6, -3, 0]
    assert task.num_input_timesteps == {"global": 2, "regional": 3}
    assert task._advance_maps["regional"] == {"inin": [(2, 0)], "outin": [(0, 1), (1, 2)]}


def test_explicit_input_offsets_skip_intermediate_regional_dates() -> None:
    """F(G[t-6], G[t], R[t-6], R[t]) -> (G[t+6], R[t+3], R[t+6])."""
    task = MixedCadenceForecaster(
        dataset_timesteps={"global": "6h", "regional": "3h"},
        block_timestep="6h",
        input_offsets={"regional": ["-6h", "0h"]},
    )
    assert _hours(task.get_input_offsets("regional")) == [-6, 0]
    assert _hours(task.get_offsets("regional", mode="training")) == [-6, 0, 3, 6]
    assert task._advance_maps["regional"] == {"inin": [(1, 0)], "outin": [(1, 1)]}

    indices = _indices()
    batch = {
        "global": torch.zeros((1, 3, 1, 1, 2)),
        "regional": torch.zeros((1, 4, 1, 1, 2)),
    }
    batch["global"][0, :, 0, 0, 0] = torch.tensor([-6, 0, 6])
    batch["regional"][0, :, 0, 0, 0] = torch.tensor([-6, 0, 3, 6])
    batch["regional"][0, :, 0, 0, 1] = torch.tensor([-60, 0, 30, 60])
    x = task.get_inputs(batch, indices)
    assert x["regional"][0, :, 0, 0, 0].tolist() == [-6, 0]

    predicted = {
        "global": torch.tensor([600.0]).reshape(1, 1, 1, 1, 1),
        "regional": torch.tensor([300.0, 600.0]).reshape(1, 2, 1, 1, 1),
    }
    next_x = task.advance_input(x, predicted, batch, data_indices=indices)
    assert next_x["regional"][0, :, 0, 0, 0].tolist() == [0, 600]
    assert next_x["regional"][0, :, 0, 0, 1].tolist() == [0, 60]


def test_input_offsets_must_be_reachable_from_the_previous_block() -> None:
    with pytest.raises(ValueError, match="neither an input nor an output"):
        MixedCadenceForecaster(
            dataset_timesteps={"regional": "3h"},
            block_timestep="6h",
            input_offsets={"regional": ["-9h", "0h"]},
        )
