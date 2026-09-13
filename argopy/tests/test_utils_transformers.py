import pytest
import logging
import xarray as xr
import numpy as np
from pathlib import Path

from argopy import tutorial
from argopy.errors import InvalidDatasetStructure
from argopy.utils.transformers import (
    fill_variables_not_in_all_datasets,
    drop_variables_not_in_all_datasets,
    merge_param_with_param_adjusted,
    filter_param_by_data_mode,
    split_data_mode,
)

log = logging.getLogger("argopy.tests.utils.transform")


def get_da(name, N_PROF, N_LEVELS):
    return xr.DataArray(
        np.random.rand(N_PROF, N_LEVELS),
        coords={"N_PROF": np.arange(0, N_PROF), "N_LEVELS": np.arange(0, N_LEVELS)},
        name=name,
    )


def get_ds(names, N_PROF=2, N_LEVELS=1):
    d = {}
    for name in names:
        d.update({name: get_da(name, N_PROF, N_LEVELS)})
    return xr.Dataset(d)


def test_drop_variables_not_in_all_datasets():
    # Create a list of dummy datasets:
    ds1 = get_ds(["PRES", "TEMP", "PSAL"], 3, 6)
    ds2 = get_ds(["PRES", "TEMP", "PSAL", "DOXY"], 3, 6)
    # Drop:
    ds_list = drop_variables_not_in_all_datasets([ds1, ds2])
    # Assert:
    assert len(ds_list) == 2
    assert "DOXY" not in ds_list[1]
    for key in ["PRES", "TEMP", "PSAL"]:
        assert key in ds_list[0]
        assert key in ds_list[1]


def test_fill_variables_not_in_all_datasets():
    # Create a list of dummy datasets:
    ds1 = get_ds(["PRES", "TEMP", "PSAL"], 3, 6)
    ds2 = get_ds(["PRES", "TEMP", "PSAL", "DOXY"], 3, 6)
    # Fill:
    ds_list = fill_variables_not_in_all_datasets([ds1, ds2], concat_dim="N_PROF")
    # Assert:
    assert len(ds_list) == 2
    for key in ["PRES", "TEMP", "PSAL", "DOXY"]:
        assert key in ds_list[0]
        assert key in ds_list[1]


class Test_merge_param_with_param_adjusted:

    def _create_ds(self):
        # Create a list of dataset that should be mergeable
        # and cover all parameter presence combination possible
        ds_list = []
        ds = get_ds(["PRES", "DATA_MODE", "TEMP_ADJUSTED"], 2, 1)
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})
        ds_list.append(ds)

        ds = get_ds(
            ["PRES", "DATA_MODE", "TEMP", "TEMP_ADJUSTED", "TEMP_ADJUSTED_QC"], 2, 1
        )
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})
        ds_list.append(ds)

        ds = get_ds(
            [
                "PRES",
                "DATA_MODE",
                "TEMP",
                "TEMP_ADJUSTED",
                "TEMP_ADJUSTED_QC",
                "TEMP_ADJUSTED_ERROR",
            ],
            2,
            1,
        )
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})
        ds_list.append(ds)

        # Now fill data mode with:
        # R only, A only, D only,
        # R & A, R & D, A & D
        ds_list_final = []
        for dd in ds_list:
            for ii in range(0, 6):
                if ii == 0:
                    dd["DATA_MODE"].values = ["R", "R"]
                elif ii == 1:
                    dd["DATA_MODE"].values = ["A", "A"]
                elif ii == 2:
                    dd["DATA_MODE"].values = ["D", "D"]
                elif ii == 3:
                    dd["DATA_MODE"].values = ["R", "A"]
                elif ii == 4:
                    dd["DATA_MODE"].values = ["R", "D"]
                elif ii == 5:
                    dd["DATA_MODE"].values = ["A", "D"]
                ds_list_final.append(dd.copy())

        return ds_list_final

    def test_ds_structure_errors(self):
        ds = get_ds(["PRES", "TEMP", "PSAL"], 3, 6)
        with pytest.raises(InvalidDatasetStructure):
            merge_param_with_param_adjusted(ds, "TEMP", errors="raise")
        assert ds == merge_param_with_param_adjusted(ds, "TEMP", errors="ignore")

        ds = get_ds(["PRES", "TEMP", "TEMP_ADJUSTED"], 3, 6)
        with pytest.raises(InvalidDatasetStructure):
            merge_param_with_param_adjusted(ds, "TEMP", errors="raise")

        ds = get_ds(["PRES", "TEMP", "TEMP_ADJUSTED"], 3, 6)
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})
        with pytest.raises(InvalidDatasetStructure):
            merge_param_with_param_adjusted(ds, "TEMP", errors="raise")
        assert ds == merge_param_with_param_adjusted(ds, "TEMP", errors="ignore")

    def test_for_parameters(self):
        ds_list = self._create_ds()
        for ds in ds_list:
            ds_merged = merge_param_with_param_adjusted(ds, "TEMP", errors="raise")
            assert "TEMP_ADJUSTED" not in ds_merged
            assert "TEMP_ADJUSTED_ERROR" not in ds_merged
            assert "TEMP_ADJUSTED_QC" not in ds_merged


class Test_filter_param_by_data_mode:

    def test_ds_structure_errors(self):
        ds = get_ds(["PRES", "TEMP", "PSAL"], 3, 6)
        with pytest.raises(InvalidDatasetStructure):
            filter_param_by_data_mode(ds, "TEMP", errors="raise")
        assert ds == filter_param_by_data_mode(ds, "TEMP", errors="ignore")

    def test_single_value_filter(self):
        ds = get_ds(["PRES", "TEMP", "TEMP_ADJUSTED", "DATA_MODE"])
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})

        ds["DATA_MODE"].values = ["R", "R"]
        ds = filter_param_by_data_mode(ds, "TEMP", dm=["R"], mask=False)
        assert len(ds["N_POINTS"]) == 2

        ds = filter_param_by_data_mode(ds, "TEMP", dm=["A"], mask=False)
        assert len(ds["N_POINTS"]) == 0

    def test_multiple_value_filter(self):
        ds = get_ds(["PRES", "TEMP", "TEMP_ADJUSTED", "DATA_MODE"])
        ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})

        ds["DATA_MODE"].values = ["R", "R"]
        dsf = filter_param_by_data_mode(ds.copy(), "TEMP", dm=["R", "A"], mask=False)
        assert len(dsf["N_POINTS"]) == 2
        assert np.unique(ds["DATA_MODE"]) == "R"

        ds["DATA_MODE"].values = ["R", "A"]
        dsf = filter_param_by_data_mode(ds.copy(), "TEMP", dm=["A", "D"], mask=False)
        assert len(dsf["N_POINTS"]) == 1
        assert dsf["DATA_MODE"].values == ["A"]

        ds["DATA_MODE"].values = ["R", "R"]
        dsf = filter_param_by_data_mode(ds.copy(), "TEMP", dm=["D", "A"], mask=False)
        assert len(dsf["N_POINTS"]) == 0


def test_split_data_mode():
    ds = get_ds(["PRES", "TEMP", "DOXY" "STATION_PARAMETERS", "PARAMETER_DATA_MODE"])
    ds = ds.stack({"N_POINTS": ["N_PROF", "N_LEVELS"]})
    with pytest.raises(InvalidDatasetStructure):
        split_data_mode(ds)

    host = tutorial.open_dataset("gdac")[0]
    ds = xr.open_dataset(Path(host).joinpath("dac/coriolis/3902131/3902131_Sprof.nc"), engine='argo')
    dss = split_data_mode(ds)
    assert "PARAMETER_DATA_MODE" not in dss
    assert "TEMP_DATA_MODE" in dss
    assert "PRES_DATA_MODE" in dss
    assert "PSAL_DATA_MODE" in dss
    assert "DOXY_DATA_MODE" in dss
