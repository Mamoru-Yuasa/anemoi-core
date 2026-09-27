# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Forecast several datasets at their native cadences in a shared block."""

import datetime
from collections.abc import Mapping

import torch

from anemoi.training.tasks.forecaster import BaseForecaster
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


class MixedCadenceForecaster(BaseForecaster):
    """Predict each dataset at its own cadence over one common forecast block.

    For example, ``block_timestep='6h'`` with ``global='6h'`` and
    ``regional='1h'`` produces one global target and six regional targets
    per model call. No unobserved global values are interpolated or loaded.

    The model predicts all outputs in a block together. Its regional outputs
    within a block are not fed back until the next block. Rollout scheduling
    (current length, curriculum, resume state) follows :class:`BaseForecaster`
    with the block as the rollout shift.
    """

    name = "mixed-cadence-forecaster"

    def __init__(
        self,
        dataset_timesteps: Mapping[str, str],
        block_timestep: str,
        multistep_input: int | Mapping[str, int] = 2,
        input_offsets: Mapping[str, list[str]] | None = None,
        rollout: dict | None = None,
        validation_rollout: int | None = None,
    ) -> None:
        if not dataset_timesteps:
            raise ValueError("dataset_timesteps must contain at least one dataset")
        block = frequency_to_timedelta(block_timestep)
        if block.total_seconds() <= 0:
            raise ValueError("block_timestep must be positive")

        self.dataset_timesteps = {
            name: frequency_to_timedelta(value) for name, value in dict(dataset_timesteps).items()
        }
        if isinstance(multistep_input, int):
            input_counts = dict.fromkeys(self.dataset_timesteps, multistep_input)
        else:
            input_counts = dict(multistep_input)
            if set(input_counts) != set(self.dataset_timesteps):
                raise ValueError("multistep_input dataset names must match dataset_timesteps")
        if any(not isinstance(count, int) or count < 1 for count in input_counts.values()):
            raise ValueError("multistep_input must contain positive integers")
        explicit_inputs = dict(input_offsets or {})
        if set(explicit_inputs) - set(self.dataset_timesteps):
            raise ValueError("input_offsets dataset names must be in dataset_timesteps")

        self._output_counts = {}
        self._input_offsets_by_dataset = {}
        self._output_offsets_by_dataset = {}
        for name, timestep in self.dataset_timesteps.items():
            if timestep.total_seconds() <= 0 or block % timestep:
                raise ValueError(f"Dataset {name!r} timestep must divide block_timestep exactly")
            self._output_counts[name] = block // timestep
            self._output_offsets_by_dataset[name] = [(i + 1) * timestep for i in range(self._output_counts[name])]
            if name in explicit_inputs:
                offsets = sorted({frequency_to_timedelta(o) for o in explicit_inputs[name]})
                if not offsets or offsets[-1] != datetime.timedelta(0) or any(o % timestep for o in offsets):
                    raise ValueError(
                        f"input_offsets of {name!r} must end at 0h and be multiples of its timestep, got "
                        f"{explicit_inputs[name]}"
                    )
            else:
                # By default the latest `multistep_input` states at the dataset's own cadence.
                offsets = sorted(-i * timestep for i in range(input_counts[name]))
            self._input_offsets_by_dataset[name] = offsets
        self._input_counts = {name: len(offsets) for name, offsets in self._input_offsets_by_dataset.items()}

        super().__init__(
            input_offsets=[],
            output_offsets=[],
            rollout_shift=block,
            rollout=rollout,
            validation_rollout=validation_rollout,
        )
        if self.rollout.start < 1 or self.rollout.maximum < self.rollout.start:
            raise ValueError("Rollout lengths must be positive and maximum >= start")
        if validation_rollout is not None and validation_rollout < 1:
            raise ValueError("validation_rollout must be positive")
        # Every input of the next block must be an input or an output of this one.
        self._advance_maps = {name: self._advance_map(name) for name in self.dataset_timesteps}

    @property
    def block_timestep(self):
        return self._rollout_shift

    @property
    def num_input_timesteps(self) -> dict[str, int]:
        return self._input_counts.copy()

    @property
    def num_output_timesteps(self) -> dict[str, int]:
        return self._output_counts.copy()

    def get_input_offsets(self, dataset_name: str, **_kwargs) -> list:
        return self._input_offsets_by_dataset[dataset_name]

    def get_output_offsets(self, dataset_name: str, rollout_step: int = 0, **_kwargs) -> list:
        shift = rollout_step * self.block_timestep
        return [offset + shift for offset in self._output_offsets_by_dataset[dataset_name]]

    def get_offsets(self, dataset_name: str, mode: str | None = None) -> list:
        if mode in ("training", "validation"):
            blocks = len(self.steps(mode))
        else:
            blocks = max(self.rollout.maximum, self.validation_rollout or 0)
        offsets = set(self.get_input_offsets(dataset_name))
        for step in range(blocks):
            offsets.update(self.get_output_offsets(dataset_name, rollout_step=step))
        return sorted(offsets)

    # Offsets grow monotonically with the rollout, so a batch loaded for the
    # current rollout is a prefix of the longest one and positions agree.
    def get_batch_input_indices(self, dataset_name: str, **_kwargs) -> list[int]:
        full = self.get_offsets(dataset_name)
        return [full.index(offset) for offset in self.get_input_offsets(dataset_name)]

    def get_batch_output_indices(self, dataset_name: str, rollout_step: int = 0, **_kwargs) -> list[int]:
        full = self.get_offsets(dataset_name)
        return [full.index(offset) for offset in self.get_output_offsets(dataset_name, rollout_step)]

    def get_inputs(self, batch: dict, data_indices: dict, **_kwargs) -> dict:
        return {
            name: values[:, normalize_time_indices(self.get_batch_input_indices(name))][
                ..., data_indices[name].data.input.full
            ]
            for name, values in batch.items()
        }

    def get_targets(self, batch: dict, rollout_step: int = 0, **_kwargs) -> dict:
        return {
            name: values[:, normalize_time_indices(self.get_batch_output_indices(name, rollout_step))]
            for name, values in batch.items()
        }

    def advance_input(
        self,
        x: dict[str, torch.Tensor],
        y_pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        rollout_step: int = 0,
        data_indices: dict | None = None,
        output_mask: dict | None = None,
        grid_shard_slice: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the next block's input from this block's inputs and predictions.

        Each next input slot is filled following the dataset's advance map: from
        a prediction of this block ("outin", prognostic variables, with forcings
        from the batch) or from a current input slot ("inin").
        """
        for name, state in x.items():
            indices = data_indices[name]
            advance_map = self._advance_maps[name]
            output_indices = self.get_batch_output_indices(name, rollout_step)
            new_state = state.clone()
            for old_idx, new_idx in advance_map["inin"]:
                new_state[:, new_idx] = state[:, old_idx]
            for out_idx, new_idx in advance_map["outin"]:
                new_state[:, new_idx, ..., indices.model.input.prognostic] = y_pred[name][
                    :, out_idx, ..., indices.model.output.prognostic
                ]
                truth = batch[name][:, output_indices[out_idx]]
                mask = None if output_mask is None else output_mask[name]
                if mask is not None:
                    if truth.shape[1] == 1 and new_state[:, new_idx].shape[1] != 1:
                        truth = truth.expand(-1, new_state[:, new_idx].shape[1], -1, -1)
                    new_state[:, new_idx] = mask.rollout_boundary(
                        new_state[:, new_idx],
                        truth,
                        indices,
                        grid_shard_slice=None if grid_shard_slice is None else grid_shard_slice[name],
                    )
                new_state[:, new_idx, ..., indices.model.input.forcing] = batch[name][
                    :, output_indices[out_idx], ..., indices.data.input.forcing
                ]
            x[name] = new_state
        return x

    def _advance_map(self, dataset_name: str) -> dict[str, list[tuple[int, int]]]:
        """Map next-block input slots to this block's outputs ("outin") or inputs ("inin")."""
        inputs = self.get_input_offsets(dataset_name)
        out_to_idx = {offset: j for j, offset in enumerate(self.get_output_offsets(dataset_name))}
        in_to_idx = {offset: j for j, offset in enumerate(inputs)}
        advance_map = {"inin": [], "outin": []}
        for new_idx, offset in enumerate(inputs):
            shifted = offset + self.block_timestep
            if shifted in out_to_idx:
                advance_map["outin"].append((out_to_idx[shifted], new_idx))
            elif shifted in in_to_idx:
                advance_map["inin"].append((in_to_idx[shifted], new_idx))
            else:
                raise ValueError(
                    f"Input offset {offset} of {dataset_name!r} is neither an input nor an output one block later"
                )
        return advance_map

    def fill_metadata(self, md_dict: dict) -> None:
        """Write each dataset's native-cadence timesteps.

        Besides the relative date indices (in units of the dataset's own
        timestep), the offset-based keys used by OffsetForecaster checkpoints
        (``input_offsets``, ``output_offsets``, ``rollout_shift`` and
        ``advance_map``) are stored per dataset, so inference can build the
        input window and advance it without task-specific code.
        """
        md_dict["task"] = self.name
        md_dict["metadata_inference"]["task"] = self.name
        for name in md_dict["metadata_inference"]["dataset_names"]:
            timestep = self.dataset_timesteps[name]
            input_offsets = self.get_input_offsets(name)
            output_offsets = self.get_output_offsets(name)
            inputs = [offset // timestep for offset in input_offsets]
            outputs = [offset // timestep for offset in output_offsets]
            md_dict["metadata_inference"][name]["timesteps"] = {
                "relative_date_indices_training": sorted(set(inputs + outputs)),
                "input_relative_date_indices": inputs,
                "output_relative_date_indices": outputs,
                "timestep": frequency_to_string(timestep),
                "input_offsets": [frequency_to_string(offset) for offset in input_offsets],
                "output_offsets": [frequency_to_string(offset) for offset in output_offsets],
                "rollout_shift": frequency_to_string(self.block_timestep),
                "advance_map": self._advance_map(name),
            }
