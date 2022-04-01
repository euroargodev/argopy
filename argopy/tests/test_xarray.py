import os
import pytest
import warnings
import numpy as np
import tempfile
import xarray as xr

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidDatasetStructure, OptionValueError
from utils import requires_connected_erddap_phy, requires_localftp, _importorskip, _connectskip

has_gsw, requires_gsw = _importorskip("gsw")
has_nogsw, requires_nogsw = _connectskip(not has_gsw, "missing GSW")


@pytest.fixture(scope="module")
@requires_connected_erddap_phy
def ds_pts():
    """ Create a dictionary of datasets to be used by tests

        Note that these datasets can be modified by tests, which can affect the behaviour of other tests !
    """
    data = {}
    try:
        for user_mode in ['standard', 'expert']:
            data[user_mode] = (
                ArgoDataFetcher(src="erddap", mode=user_mode)
                .region([-75, -55, 30.0, 40.0, 0, 100.0, "2011-01-01", "2011-01-15"])
                .load()
                .data
            )
    except Exception as e:
        warnings.warn("Error when fetching tests data: %s" % str(e.args))
        pass

    if "expert" not in data or "standard" not in data:
        # We don't have what we need for testing, skip this test module:
        pytest.xfail("Could not retrieve erddap data in both standard and expert mode")
    else:
        return data


@requires_connected_erddap_phy
def test_point2profile(ds_pts):
    assert "N_PROF" in ds_pts['standard'].argo.point2profile().dims


@requires_connected_erddap_phy
def test_profile2point(ds_pts):
    with pytest.raises(InvalidDatasetStructure):
        ds_pts['standard'].argo.profile2point()


@requires_connected_erddap_phy
def test_point2profile2point(ds_pts):
    assert ds_pts['standard'].argo.point2profile().argo.profile2point().equals(ds_pts['standard'])


@requires_connected_erddap_phy
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


@requires_connected_erddap_phy
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


@requires_connected_erddap_phy
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
@requires_localftp
class Test_create_float_source:
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]

    def is_valid_mdata(self, this_mdata):
        """Validate structure of the output dataset """
        check = []
        # Check for dimensions:
        check.append(argopy.utilities.is_list_equal(['m', 'n'], list(this_mdata.dims)))
        # Check for coordinates:
        check.append(argopy.utilities.is_list_equal(['m', 'n'], list(this_mdata.coords)))
        # Check for data variables:
        check.append(np.all(
            [v in this_mdata.data_vars for v in ['PRES', 'TEMP', 'PTMP', 'SAL', 'DATES', 'LAT', 'LONG', 'PROFILE_NO']]))
        check.append(np.all(
            [argopy.utilities.is_list_equal(['n'], this_mdata[v].dims) for v in ['LONG', 'LAT', 'DATES', 'PROFILE_NO']
             if v in this_mdata.data_vars]))
        check.append(np.all(
            [argopy.utilities.is_list_equal(['m', 'n'], this_mdata[v].dims) for v in ['PRES', 'TEMP', 'SAL', 'PTMP'] if
             v in this_mdata.data_vars]))
        return np.all(check)

    def test_error_user_mode(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(InvalidDatasetStructure):
                ds = ArgoDataFetcher(src="localftp", mode='standard').float([6901929, 2901623]).load().data
                ds.argo.create_float_source()

    def test_opt_force(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            expert_ds = ArgoDataFetcher(src="localftp", mode='expert').float([2901623]).load().data

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
        with argopy.set_options(local_ftp=self.local_ftp):
            expert_ds = ArgoDataFetcher(src="localftp", mode='expert').float([6901929, 2901623]).load().data

            N_file = len(np.unique(expert_ds['PLATFORM_NUMBER']))
            with tempfile.TemporaryDirectory() as folder_output:
                expert_ds.argo.create_float_source(path=folder_output)
                assert len(os.listdir(folder_output)) == N_file
