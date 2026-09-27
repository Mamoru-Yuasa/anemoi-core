# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anemoi.training.data.data_reader import BaseAnemoiReader

LOGGER = logging.getLogger(__name__)


def _intersect_anchor_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the ``(sequence, position)`` rows present in both anchor arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays of shape ``(n, 2)`` of ``(sequence, position)`` anchors.

    Returns
    -------
    np.ndarray
        Array of shape ``(m, 2)`` with the common anchors, sorted.
    """
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    common = np.intersect1d(
        np.ascontiguousarray(a).view(dtype),
        np.ascontiguousarray(b).view(dtype),
    )
    return common.view(a.dtype).reshape(-1, 2)


def compute_valid_anchors(
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
) -> np.ndarray:
    """Return the valid ``(sequence, position)`` anchors shared by all readers.

    An anchor ``(s, p)`` is valid if every reader can sample the positions
    ``p + i`` for all of its relative offsets ``i`` within sequence ``s``.
    Returns the intersection of the valid anchors across all data readers.

    Parameters
    ----------
    data_readers : dict[str, BaseAnemoiReader]
        Mapping of dataset name to data reader.
    relative_date_indices : dict[str, np.ndarray | list[int]]
        Relative offsets (in positions) requested for each reader.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_anchors, 2)`` with the shared ``(sequence, position)``
        anchors.
    """
    intersection: np.ndarray | None = None
    for dataset_name, ds in data_readers.items():
        anchors = ds.compute_anchors(relative_date_indices[dataset_name])

        if len(anchors) == 0:
            msg = f"No valid anchors found for data reader '{dataset_name}': {ds}"
            raise ValueError(msg)

        intersection = anchors if intersection is None else _intersect_anchor_rows(intersection, anchors)

        LOGGER.info("Data reader '%s' has %d valid anchors", dataset_name, len(anchors))

    if intersection is None or len(intersection) == 0:
        msg = "No valid anchors found after intersection across all datasets."
        raise ValueError(msg)

    LOGGER.info("MultiDataset has %d valid anchors after intersection.", len(intersection))

    return intersection


def _exact_window_mask(
    dates: np.ndarray,
    positions: np.ndarray,
    relative_indices: np.ndarray | list[int],
    frequency: datetime.timedelta | np.timedelta64,
) -> np.ndarray:
    """Return which anchor positions have windows whose dates match the requested offsets.

    An integer offset is meaningful only when the indexed dates are exactly the
    requested time offsets. This protects against date arrays with unmarked gaps.
    """
    if isinstance(frequency, datetime.timedelta):
        frequency = np.timedelta64(frequency // datetime.timedelta(microseconds=1), "us")
    anchor_dates = dates[positions]
    mask = np.ones(len(positions), dtype=bool)
    for offset in relative_indices:
        mask &= dates[positions + offset] == anchor_dates + offset * frequency
    return mask


def compute_aligned_anchors(
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
) -> dict[str, np.ndarray]:
    """Return, for every reader, the ``(sequence, position)`` anchors of the shared samples.

    Anchor positions of different readers cannot be intersected directly when
    the readers are single time series: position 1 of a 6-hour series and of a
    3-hour series denote different dates (and even equal-frequency series may
    start at different dates). For such readers the valid anchors are mapped to
    their dates, the dates are intersected, and each reader keeps its own
    position for every shared date. Row ``i`` of every returned array refers to
    the same sample.

    Trajectory readers (several sequences) keep the positional intersection of
    :func:`compute_valid_anchors`, identical for all readers.
    """
    if not data_readers:
        msg = "At least one data reader is required."
        raise ValueError(msg)

    if any(ds.num_sequences != 1 for ds in data_readers.values()):
        anchors = compute_valid_anchors(data_readers, relative_date_indices)
        return dict.fromkeys(data_readers, anchors)

    common_dates = None
    aligned: dict[str, np.ndarray] = {}
    for dataset_name, ds in data_readers.items():
        anchors = ds.compute_anchors(relative_date_indices[dataset_name])
        positions = anchors[:, 1]
        dates = np.asarray(ds.dates)
        if dates.dtype.kind == "M" and len(positions):
            positions = positions[_exact_window_mask(dates, positions, relative_date_indices[dataset_name], ds.frequency)]

        if len(positions) == 0:
            msg = f"No valid anchors found for data reader '{dataset_name}': {ds}"
            raise ValueError(msg)

        valid_dates = dates[positions]
        if len(np.unique(valid_dates)) != len(valid_dates):
            msg = f"Data reader '{dataset_name}' has duplicate valid anchor dates; alignment is ambiguous."
            raise ValueError(msg)

        LOGGER.info("Data reader '%s' has %d valid anchors", dataset_name, len(positions))

        if common_dates is None:
            common_dates = valid_dates
            aligned[dataset_name] = positions
            continue

        common_dates, previous_rows, reader_rows = np.intersect1d(
            common_dates,
            valid_dates,
            assume_unique=True,
            return_indices=True,
        )
        aligned = {name: pos[previous_rows] for name, pos in aligned.items()}
        aligned[dataset_name] = positions[reader_rows]

    if len(common_dates) == 0:
        msg = "No valid anchors found after intersection across all datasets."
        raise ValueError(msg)

    # Keep samples in the order of the first reader's positions.
    order = np.argsort(aligned[next(iter(data_readers))], kind="stable")
    LOGGER.info("MultiDataset has %d valid anchors after intersection.", len(common_dates))
    return {
        name: np.stack([np.zeros(len(pos), dtype=np.int64), pos[order].astype(np.int64)], axis=1)
        for name, pos in aligned.items()
    }


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    relative_indices: np.ndarray | list[int],
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Set of missing indices in the series.
    series_length : int
        Length of the series.
    relative_indices: np.ndarray | list[int]
        Array of relative indices requested at each index i.

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    if isinstance(relative_indices, list):
        relative_indices = np.array(relative_indices)

    usable_indices = np.arange(series_length)

    # Restrict to indices where all relative positions are within bounds
    max_offset = int(max(relative_indices))
    min_offset = int(min(relative_indices))
    usable_indices = usable_indices[(usable_indices + min_offset >= 0) & (usable_indices + max_offset < series_length)]

    # Missing indices
    for i in missing_indices:
        rel_missing = i - relative_indices  # indices which have their relative indices match the missing.
        usable_indices = usable_indices[np.all(usable_indices != rel_missing[:, np.newaxis], axis=0)]

    return usable_indices
