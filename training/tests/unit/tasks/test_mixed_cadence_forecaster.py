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
