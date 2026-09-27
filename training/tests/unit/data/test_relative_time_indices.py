# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.relative_time_indices import compute_relative_date_indices


def test_get_sample_normalizes_time_indices_before_dataset_access() -> None:
    """Test that NativeGridDataset.get_sample normalizes time indices to slices before accessing the dataset."""

    class FakeDataset:
        def __init__(self) -> None:
            self.last_index = None

        def __getitem__(self, item: int) -> np.ndarray:
            self.last_index = item
            return np.zeros((3, 2, 4, 5), dtype=np.float32)

    dataset = NativeGridDataset.__new__(NativeGridDataset)
    dataset.data = FakeDataset()

    dataset.get_sample(sequence=0, positions=[4, 5, 6], grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, list)

    dataset.get_sample(sequence=0, positions=slice(4, 7, 1), grid_shard_indices=slice(0, 5))

    time_index = dataset.data.last_index[0]
    assert isinstance(time_index, slice)
    assert (time_index.start, time_index.stop, time_index.step) == (4, 7, 1)


def test_compute_relative_date_indices_uses_per_dataset_offsets() -> None:
    """A coarse reader must never be asked for unavailable 3-hour slots."""
    hour = datetime.timedelta(hours=1)

    class PerDatasetTask:
        def get_offsets(self, dataset_name: str, mode: str) -> list[datetime.timedelta]:
            assert mode == "training"
            return [-6 * hour, 0 * hour, 6 * hour] if dataset_name == "global" else [-3 * hour, 0 * hour, 3 * hour]

    readers = {
        "global": SimpleNamespace(frequency=6 * hour),
        "regional": SimpleNamespace(frequency=3 * hour),
    }
    assert compute_relative_date_indices(PerDatasetTask(), readers, mode="training") == {
        "global": [-1, 0, 1],
        "regional": [-1, 0, 1],
    }


def test_compute_relative_date_indices_preserves_legacy_task_and_rejects_unavailable_dates() -> None:
    hour = datetime.timedelta(hours=1)

    class LegacyTask:
        def get_offsets(self, mode: str) -> list[datetime.timedelta]:
            assert mode == "training"
            return [0 * hour, 3 * hour]

    readers = {"regional": SimpleNamespace(frequency=3 * hour)}
    assert compute_relative_date_indices(LegacyTask(), readers, mode="training") == {"regional": [0, 1]}

    readers["global"] = SimpleNamespace(frequency=6 * hour)
    with pytest.raises(ValueError, match="not compatible"):
        compute_relative_date_indices(LegacyTask(), readers, mode="training")
