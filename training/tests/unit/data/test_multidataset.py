# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pytest_mock import MockFixture

from anemoi.training.data.data_reader import NativeGridDataset
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.utils.seeding import SeedContext
from anemoi.training.utils.seeding import derive_seed


class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(30))  # 15 reference dates
        mock_dataset_a.frequency = "3h"
        mock_dataset_a.num_sequences = 1
        # relative_date_indices=[0,2,6], window=7, valid positions [0..23] at sequence 0
        anchors_a = np.column_stack([np.zeros(24, dtype=np.int64), np.arange(24, dtype=np.int64)])
        mock_dataset_a.compute_anchors.return_value = anchors_a

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {7, 8, 9, 10}
        mock_dataset_b.dates = list(range(30))  # 15 reference dates
        mock_dataset_b.frequency = "3h"
        mock_dataset_b.num_sequences = 1
        # missing {7..10}: exclude positions {1..10}, valid positions [0, 11..23] at sequence 0
        pos_b = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
        anchors_b = np.column_stack([np.zeros(14, dtype=np.int64), pos_b])
        mock_dataset_b.compute_anchors.return_value = anchors_b

        data_readers = {"dataset_a": mock_dataset_a, "dataset_b": mock_dataset_b}
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}  # e.g. f([t, t-6h]) = t+12h

        return MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that valid_date_indices returns a flat range over the valid (sequence, position) anchors."""
        # relative_date_indices are: [0, 2, 6]
        # dataset_a has no missing → valid positions [0..23] at sequence 0
        # dataset_b has missing {7,8,9,10} → valid positions [0, 11..23] at sequence 0
        # intersection: [0, 11..23] → 14 anchors
        expected_positions = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        # valid_date_indices is a flat index range over the anchors array
        valid_indices = multi_dataset.valid_date_indices
        assert np.array_equal(valid_indices, np.arange(len(expected_positions)))

        # the anchors themselves encode the expected positions at sequence 0
        assert np.array_equal(multi_dataset.anchors[:, 1], expected_positions)

    def test_set_epoch_updates_contiguous_relative_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that set_epoch can update the loaded rollout to contiguous relative date indices."""
        multi_dataset.set_epoch(
            2,
            rollout=3,
            relative_date_indices={"dataset_a": [0, 1, 2], "dataset_b": [0, 1, 2]},
        )

        assert multi_dataset.epoch == 2
        assert multi_dataset.rollout == 3
        assert multi_dataset.relative_date_indices == {
            "dataset_a": slice(0, 3, 1),
            "dataset_b": slice(0, 3, 1),
        }
        assert len(multi_dataset.valid_date_indices) > 0

    def test_worker_seed_includes_epoch(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that worker RNG seed changes with epoch while staying shared across worker partitions."""
        mocker.patch("anemoi.training.data.multidataset.get_base_seed", return_value=1000)

        multi_dataset.set_epoch(0)
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)
        seed_epoch_0 = multi_dataset.seed
        assert seed_epoch_0 == derive_seed(1000, SeedContext.DATALOADER, 0)

        multi_dataset.set_epoch(5)
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)
        seed_epoch_5 = multi_dataset.seed
        assert seed_epoch_5 == derive_seed(1000, SeedContext.DATALOADER, 5)

        assert seed_epoch_0 != seed_epoch_5

        multi_dataset.per_worker_init(n_workers=4, worker_id=3)
        assert multi_dataset.seed == seed_epoch_5

    def test_worker_shuffle_repeats_for_same_epoch(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """New workers reproduce the shuffle when the base seed and epoch match."""
        mocker.patch("anemoi.training.data.multidataset.get_base_seed", return_value=1000)
        mocker.patch.object(multi_dataset, "get_sample", side_effect=lambda index: int(index))

        multi_dataset.set_epoch(5)
        multi_dataset.per_worker_init(n_workers=2, worker_id=1)
        uninterrupted_order = list(multi_dataset)

        multi_dataset.per_worker_init(n_workers=2, worker_id=1)
        resumed_order = list(multi_dataset)

        assert resumed_order == uninterrupted_order

    def test_fake_dataloading_reuses_first_batch(
        self,
        multi_dataset: MultiDataset,
        mocker: MockFixture,
    ) -> None:
        """Fake dataloading reads one valid batch and reuses its tensors."""
        multi_dataset.fake_dataloading = True
        get_sample = mocker.patch.object(
            multi_dataset,
            "get_sample",
            side_effect=lambda index: {"dataset_a": torch.tensor([index], dtype=torch.int64)},
        )
        multi_dataset.per_worker_init(n_workers=1, worker_id=0)

        batches = list(multi_dataset)

        assert get_sample.call_count == 1
        assert len(batches) == len(multi_dataset.valid_date_indices)
        assert all(batch is batches[0] for batch in batches)
        assert all(torch.equal(batch["dataset_a"], batches[0]["dataset_a"]) for batch in batches)

    def test_valid_date_indices_empty_dataset(self, multi_dataset: MultiDataset) -> None:
        """Test that MultiDataset raises ValueError when a dataset has no valid anchors."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # Make dataset_b return no valid anchors
        data_readers["dataset_b"].compute_anchors.return_value = np.empty((0, 2), dtype=np.int64)

        # Constructing MultiDataset should raise ValueError
        empty_dataset = data_readers["dataset_b"]
        err_msg = f"No valid anchors found for data reader 'dataset_b': {empty_dataset}"
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset) -> None:
        """Test that MultiDataset raises ValueError when intersection of valid anchors is empty."""
        data_readers = multi_dataset.data_readers
        relative_date_indices = {"dataset_a": [0, 2, 6], "dataset_b": [0, 2, 6]}

        # dataset_a has anchors at positions [0, 1, 2]; dataset_b at [5, 6, 7] — no overlap
        data_readers["dataset_a"].compute_anchors.return_value = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
        data_readers["dataset_b"].compute_anchors.return_value = np.array([[0, 5], [0, 6], [0, 7]], dtype=np.int64)

        with pytest.raises(ValueError, match="No valid anchors found after intersection across all datasets"):
            MultiDataset(data_readers=data_readers, relative_date_indices=relative_date_indices)


def _native_reader(dates: np.ndarray, hours: int, missing: set[int] | None = None) -> NativeGridDataset:
    """Return a NativeGridDataset over an in-memory time axis (no data values)."""
    reader = NativeGridDataset.__new__(NativeGridDataset)
    reader.data = SimpleNamespace(
        dates=dates,
        frequency=datetime.timedelta(hours=hours),
        missing=missing or set(),
    )
    reader.default_sampling = {"stride": 1}
    return reader


def test_mixed_frequency_samples_align_by_date_and_respect_missing_values(mocker: MockFixture) -> None:
    """Reader positions can differ even when they refer to the same anchor date."""
    start = np.datetime64("2020-01-01T00")
    global_reader = _native_reader(start + np.arange(5) * np.timedelta64(6, "h"), 6, missing={3})  # t+18h
    # The regional series begins three hours before the global series.
    regional_reader = _native_reader(start + (np.arange(10) - 1) * np.timedelta64(3, "h"), 3, missing={4})  # t+9h

    dataset = MultiDataset(
        data_readers={"global": global_reader, "regional": regional_reader},
        relative_date_indices={"global": [0, 1], "regional": [0, 1]},
    )

    # t=0 is the sole anchor valid for both windows; its regional position is 1.
    np.testing.assert_array_equal(dataset.reader_anchors["global"], [[0, 0]])
    np.testing.assert_array_equal(dataset.reader_anchors["regional"], [[0, 1]])

    get_global = mocker.patch.object(global_reader, "get_sample")
    get_regional = mocker.patch.object(regional_reader, "get_sample")
    dataset.get_sample(0)
    assert get_global.call_args.args[1] == slice(0, 2, 1)
    assert get_regional.call_args.args[1] == slice(1, 3, 1)


def test_mixed_frequency_supports_hourly_regional_reader(mocker: MockFixture) -> None:
    start = np.datetime64("2020-01-01T00")
    global_reader = _native_reader(start + np.arange(4) * np.timedelta64(6, "h"), 6)
    regional_reader = _native_reader(start + np.arange(19) * np.timedelta64(1, "h"), 1)

    dataset = MultiDataset(
        data_readers={"global": global_reader, "regional": regional_reader},
        relative_date_indices={"global": [0, 1], "regional": [0, 1, 2, 3, 4, 5, 6]},
    )

    np.testing.assert_array_equal(dataset.reader_anchors["global"][:, 1], [0, 1, 2])
    np.testing.assert_array_equal(dataset.reader_anchors["regional"][:, 1], [0, 6, 12])
    get_regional = mocker.patch.object(regional_reader, "get_sample")
    mocker.patch.object(global_reader, "get_sample")
    dataset.get_sample(1)
    assert get_regional.call_args.args[1] == slice(6, 13, 1)


def test_mixed_frequency_rejects_unmarked_date_gaps() -> None:
    """A position step must represent its declared time interval, even without a missing flag."""
    start = np.datetime64("2020-01-01T00")
    global_reader = _native_reader(start + np.arange(3) * np.timedelta64(6, "h"), 6)
    # The 07 UTC timestamp is absent: position 7 is 08 UTC, not 07 UTC.
    hours = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    regional_reader = _native_reader(start + hours * np.timedelta64(1, "h"), 1)

    dataset = MultiDataset(
        data_readers={"global": global_reader, "regional": regional_reader},
        relative_date_indices={"global": [0, 1], "regional": [0, 1]},
    )

    np.testing.assert_array_equal(dataset.reader_anchors["global"][:, 1], [0])
