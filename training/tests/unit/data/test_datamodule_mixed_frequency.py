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

from pytest_mock import MockFixture

from anemoi.training.data.datamodule import AnemoiDatasetsDataModule


def test_statistics_tendencies_use_each_dataset_output_offset(mocker: MockFixture) -> None:
    """Tendency statistics must use each dataset's forecast lead time."""

    class PerDatasetTask:
        def get_output_offsets(self, dataset_name: str) -> list[datetime.timedelta]:
            hours = 6 if dataset_name == "global" else 3
            return [datetime.timedelta(hours=hours)]

    global_reader = mocker.MagicMock()
    global_reader.statistics_tendencies.return_value = {"global": True}
    regional_reader = mocker.MagicMock()
    regional_reader.statistics_tendencies.return_value = {"regional": True}

    module = AnemoiDatasetsDataModule.__new__(AnemoiDatasetsDataModule)
    module.task = PerDatasetTask()
    module.__dict__["ds_train"] = SimpleNamespace(data_readers={"global": global_reader, "regional": regional_reader})

    statistics = module.statistics_tendencies
    global_reader.statistics_tendencies.assert_called_once_with("6h")
    regional_reader.statistics_tendencies.assert_called_once_with("3h")
    assert statistics["global"]["lead_times"] == ["6h"]
    assert statistics["regional"]["lead_times"] == ["3h"]
