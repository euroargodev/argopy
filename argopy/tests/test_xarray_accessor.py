import os
import pytest
import warnings
import numpy as np
import tempfile
import xarray as xr

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidDatasetStructure, OptionValueError
from utils import requires_gdac, _importorskip, _connectskip
from mocked_http import mocked_server_address


has_gsw, requires_gsw = _importorskip("gsw")
has_nogsw, requires_nogsw = _connectskip(not has_gsw, "that GSW module is NOT installed")


@pytest.fixture(scope="module")
def ds_pts(mocked_httpserver):
    """ Create a dictionary of datasets to be used by tests

        Note that these datasets can be modified by tests, which can affect the behaviour of other tests !
    """
    data = {}
    try:
        for user_mode in ['standard', 'expert']:
            data[user_mode] = (
                ArgoDataFetcher(src="erddap", mode=user_mode, server=mocked_server_address)
                .region([-20, -16., 0, 1, 0, 100., "2004-01-01", "2004-01-31"])
                .load()
                .data
            )
    except Exception as e:
        warnings.warn("Error when fetching tests data: %s" % str(e.args))
        pass

    if "expert" not in data:
        # We don't have what we need for testing, skip this test module:
        pytest.xfail("Could not retrieve erddap data in expert mode")
    elif "standard" not in data:
        # We don't have what we need for testing, skip this test module:
        pytest.xfail("Could not retrieve erddap data in standard mode")
    else:
        return data


def test_point2profile(ds_pts):
    assert "N_PROF" in ds_pts['standard'].argo.point2profile().dims


def test_profile2point(ds_pts):
    with pytest.raises(InvalidDatasetStructure):
        ds_pts['standard'].argo.profile2point()


def test_point2profile2point(ds_pts):
    assert ds_pts['standard'].argo.point2profile().argo.profile2point().equals(ds_pts['standard'])


def test_point2profile_reindexes_N_PROF():
    """point2profile must reset N_PROF to a clean 0..N-1 range (gh #632).

    The profiles are built in UID (WMO/CYCLE) order but the final dataset is
    sorted by time; the N_PROF coordinate used to keep its pre-sort labels,
    leaving e.g. [0, 3, 1, 2] instead of [0, 1, 2, 3].
    """
    N_LEVELS = 3
    # (platform, cycle, time offset in days) — deliberately so that time order
    # differs from UID order, which is what triggers the bug:
    profiles = [(6902, 2, 10), (6902, 1, 30), (6901, 5, 5), (6901, 9, 20)]
    platform, cycle, day, level = [], [], [], []
    for wmo, cyc, t in profiles:
        for lev in range(N_LEVELS):
            platform.append(wmo)
            cycle.append(cyc)
            day.append(t)
            level.append(lev)
    n = len(platform)
    day = np.array(day)
    ds = xr.Dataset(
        {
            "PLATFORM_NUMBER": ("N_POINTS", np.array(platform, "float64")),
            "CYCLE_NUMBER": ("N_POINTS", np.array(cycle, "int64")),
            "DIRECTION": ("N_POINTS", np.array(["A"] * n)),
            "PRES": ("N_POINTS", np.array(level, "float64") + 1),
            # vary with level so TEMP stays a [N_PROF, N_LEVELS] variable, and
            # encode the profile time so association can be checked after sorting
            "TEMP": ("N_POINTS", day.astype("float64") * 10 + np.array(level)),
            "TIME": (
                "N_POINTS",
                np.array(
                    [np.datetime64("2004-01-01") + np.timedelta64(int(t), "D") for t in day]
                ),
            ),
            "LATITUDE": ("N_POINTS", np.ones(n)),
            "LONGITUDE": ("N_POINTS", np.ones(n) * 2),
        },
        coords={"N_POINTS": np.arange(n)},
    )
    ds = ds.argo.cast_types()
    ds.argo._type = "point"

    profile = ds.argo.point2profile()

    # N_PROF is a clean contiguous range, not the permuted pre-sort labels
    np.testing.assert_array_equal(
        profile["N_PROF"].values, np.arange(profile.sizes["N_PROF"])
    )
    # profiles are still ordered by time ...
    assert (np.diff(profile["TIME"].values) >= np.timedelta64(0)).all()
    # ... and each profile's data still travels with it (TEMP encodes the day)
    expected_days = np.sort(np.unique(day))
    np.testing.assert_array_equal(
        profile["TEMP"].isel(N_LEVELS=0).values, expected_days * 10
    )


class Test_interp_std_levels:
    def test_interpolation(self, ds_pts):
        """Run with success"""
        ds = ds_pts["standard"].argo.point2profile()
        assert "PRES_INTERPOLATED" in ds.argo.interp_std_levels([20, 30, 40, 50]).dims

    def test_interpolation_expert(self, ds_pts):
        """Run with success"""
        ds = ds_pts["expert"].argo.point2profile()
        assert "PRES_INTERPOLATED" in ds.argo.interp_std_levels([20, 30, 40, 50]).dims

    def test_std_error(self, ds_pts):
        """Try to interpolate on a wrong axis"""
        ds = ds_pts["standard"].argo.point2profile()
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels([100, 20, 30, 40, 50])
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels([-20, 20, 30, 40, 50])
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels(12)


class Test_groupby_pressure_bins:
    def test_groupby_ds_type(self, ds_pts):
        """Run with success for standard/expert mode and point/profile"""
        for user_mode, this in ds_pts.items():
            for format in ["point", "profile"]:
                if format == 'profile':
                    that = this.argo.point2profile()
                else:
                    that = this.copy()
                bins = np.arange(0.0, np.max(that["PRES"]) + 10.0, 10.0)
                assert "STD_PRES_BINS" in that.argo.groupby_pressure_bins(bins).coords

    def test_bins_error(self, ds_pts):
        """Try to groupby over invalid bins """
        ds = ds_pts["standard"]
        with pytest.raises(ValueError):
            ds.argo.groupby_pressure_bins([100, 20, 30, 40, 50])  # un-sorted
        with pytest.raises(ValueError):
            ds.argo.groupby_pressure_bins([-20, 20, 30, 40, 50])  # Negative values

    def test_axis_error(self, ds_pts):
        """Try to group by using invalid pressure axis """
        ds = ds_pts["standard"]
        bins = np.arange(0.0, np.max(ds["PRES"]) + 10.0, 10.0)
        with pytest.raises(ValueError):
            ds.argo.groupby_pressure_bins(bins, axis='invalid')

    def test_empty_result(self, ds_pts):
        """Try to groupby over bins without data"""
        ds = ds_pts["standard"]
        with pytest.warns(Warning):
            out = ds.argo.groupby_pressure_bins([10000, 20000])
        assert out == None

    def test_all_select(self, ds_pts):
        ds = ds_pts["standard"]
        bins = np.arange(0.0, np.max(ds["PRES"]) + 10.0, 10.0)
        for select in ["shallow", "deep", "middle", "random", "min", "max", "mean", "median"]:
            assert "STD_PRES_BINS" in ds.argo.groupby_pressure_bins(bins).coords


class Test_teos10:

    @requires_nogsw
    def test_gsw_not_available(self, ds_pts):
        # Make sure we raise an error when GSW is not available
        for key, this in ds_pts.items():
            that = this.copy()  # To avoid modifying the original dataset
            with pytest.raises(ModuleNotFoundError):
                that.argo.teos10()

    @requires_gsw
    def test_teos10_variables_default(self, ds_pts):
        """Add default new set of TEOS10 variables"""
        for key, this in ds_pts.items():
            for format in ["point", "profile"]:
                that = this.copy()  # To avoid modifying the original dataset
                if format == "profile":
                    that = that.argo.point2profile()
                that = that.argo.teos10()
                assert "SA" in that.variables
                assert "CT" in that.variables

    @requires_gsw
    def test_teos10_variables_single(self, ds_pts):
        """Add a single TEOS10 variables"""
        for key, this in ds_pts.items():
            for format in ["point", "profile"]:
                that = this.copy()  # To avoid modifying the original dataset
                if format == "profile":
                    that = that.argo.point2profile()
                that = that.argo.teos10(["PV"])
                assert "PV" in that.variables

    @requires_gsw
    def test_teos10_opt_variables_single(self, ds_pts):
        """Add a single TEOS10 optional variables"""
        for key, this in ds_pts.items():
            for format in ["point", "profile"]:
                that = this.copy()  # To avoid modifying the original dataset
                if format == "profile":
                    that = that.argo.point2profile()
                that = that.argo.teos10(["SOUND_SPEED"])
                assert "SOUND_SPEED" in that.variables

    @requires_gsw
    def test_teos10_variables_inplace(self, ds_pts):
        """Compute all default variables to a new dataset"""
        for key, this in ds_pts.items():
            ds = this.argo.teos10(inplace=False)  # So "SA" must be in 'ds' but not in 'this'
            assert "SA" in ds.variables
            assert "SA" not in this.variables

    @requires_gsw
    def test_teos10_invalid_variable(self, ds_pts):
        """Try to add an invalid variable"""
        for key, this in ds_pts.items():
            for format in ["point", "profile"]:
                that = this.copy()  # To avoid modifying the original dataset
                if format == "profile":
                    that = that.argo.point2profile()
                with pytest.raises(ValueError):
                    that.argo.teos10(["InvalidVariable"])


@requires_gsw
@requires_gdac
class Test_create_float_source:
    local_gdac = argopy.tutorial.open_dataset("gdac")[0]

    def is_valid_mdata(self, this_mdata):
        """Validate structure of the output dataset """
        check = []
        # Check for dimensions:
        check.append(argopy.utils.is_list_equal(['m', 'n'], list(this_mdata.dims)))
        # Check for coordinates:
        check.append(argopy.utils.is_list_equal(['m', 'n'], list(this_mdata.coords)))
        # Check for data variables:
        check.append(np.all(
            [v in this_mdata.data_vars for v in ['PRES', 'TEMP', 'PTMP', 'SAL', 'DATES', 'LAT', 'LONG', 'PROFILE_NO']]))
        check.append(np.all(
            [argopy.utils.is_list_equal(['n'], this_mdata[v].dims) for v in ['LONG', 'LAT', 'DATES', 'PROFILE_NO']
             if v in this_mdata.data_vars]))
        check.append(np.all(
            [argopy.utils.is_list_equal(['m', 'n'], this_mdata[v].dims) for v in ['PRES', 'TEMP', 'SAL', 'PTMP'] if
             v in this_mdata.data_vars]))
        return np.all(check)

    def test_error_user_mode(self):
        with argopy.set_options(gdac=self.local_gdac):
            with pytest.raises(InvalidDatasetStructure):
                ds = ArgoDataFetcher(src="gdac", mode='standard').float([6901929, 2901623]).load().data
                ds.argo.create_float_source()

    def test_opt_force(self):
        with argopy.set_options(gdac=self.local_gdac):
            expert_ds = ArgoDataFetcher(src="gdac", mode='expert').float([2901623]).load().data

            with pytest.raises(OptionValueError):
                expert_ds.argo.create_float_source(force='dummy')

            ds_float_source = expert_ds.argo.create_float_source(path=None, force='default')
            assert np.all([k in np.unique(expert_ds['PLATFORM_NUMBER']) for k in ds_float_source.keys()])
            assert np.all([isinstance(ds_float_source[k], xr.Dataset) for k in ds_float_source.keys()])
            assert np.all([self.is_valid_mdata(ds_float_source[k]) for k in ds_float_source.keys()])

            ds_float_source = expert_ds.argo.create_float_source(path=None, force='raw')
            assert np.all([k in np.unique(expert_ds['PLATFORM_NUMBER']) for k in ds_float_source.keys()])
            assert np.all([isinstance(ds_float_source[k], xr.Dataset) for k in ds_float_source.keys()])
            assert np.all([self.is_valid_mdata(ds_float_source[k]) for k in ds_float_source.keys()])

    def test_filecreate(self):
        with argopy.set_options(gdac=self.local_gdac):
            expert_ds = ArgoDataFetcher(src="gdac", mode='expert').float([6901929, 2901623]).load().data

            N_file = len(np.unique(expert_ds['PLATFORM_NUMBER']))
            with tempfile.TemporaryDirectory() as folder_output:
                expert_ds.argo.create_float_source(path=folder_output)
                assert len(os.listdir(folder_output)) == N_file
