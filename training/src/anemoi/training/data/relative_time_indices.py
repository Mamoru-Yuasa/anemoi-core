# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
from collections.abc import Callable
from inspect import Parameter
from inspect import signature

from anemoi.training.tasks.base import BaseTask
from anemoi.utils.dates import frequency_to_string


def get_dataset_offsets(
    method: Callable[..., list[datetime.timedelta]],
    dataset_name: str,
    **kwargs,
) -> list[datetime.timedelta]:
    """Call a task offset method with dataset_name when it supports that argument."""
    parameters = signature(method).parameters
    accepts_dataset_name = "dataset_name" in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_dataset_name:
        return method(dataset_name=dataset_name, **kwargs)
    return method(**kwargs)


def compute_relative_date_indices(
    task: BaseTask,
    data_readers: dict,
    **kwargs,
) -> dict[str, list[int]]:
    """Compute relative date indices for each dataset based on task offsets."""
    relative_date_indices = {}
    for name, dr in data_readers.items():
        # Older tasks define one set of offsets and accept only ``mode``. Tasks
        # with dataset-specific temporal windows can also accept dataset_name.
        offsets = get_dataset_offsets(task.get_offsets, name, **kwargs)
        if any(o % dr.frequency for o in offsets):
            msg = (
                f"The frequency of `{name}` ({frequency_to_string(dr.frequency)}) is not compatible "
                f"with the task defined offsets ({[frequency_to_string(o) for o in offsets]}). "
                f"Check that the task offsets are compatible with the dataset frequency."
            )
            raise ValueError(msg)
        relative_date_indices[name] = [o // dr.frequency for o in offsets]

    return relative_date_indices
