import pytest
import warnings

from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidDatasetStructure
from . import requires_connected_erddap_phy


@pytest.fixture(scope="module")
@requires_connected_erddap_phy
def ds_pts():
    """ Create a dictionary of datasets to be used by tests

        Note that these datasets can be modified by tests, which can affect the behaviour of other tests !
    """
    data = {}
    try:
        # raise ValueError('toto')
        for user_mode in ['standard', 'expert']:
            data[user_mode] = (
                ArgoDataFetcher(src="erddap", mode=user_mode)
                .region([-75, -55, 30.0, 40.0, 0, 100.0, "2011-01-01", "2011-01-15"])
                .to_xarray()
            )
    except Exception as e:
        warnings.warn("Error when fetching tests data: %s" % str(e.args))
        pass

    if "expert" not in data or "standard" not in data:
        # We don't have what we need for testing, skip this test module:
        pytest.skip("Tests data not available")
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
    
    def test_points_error(self, ds_pts):
        """Try to interpolate points, not profiles"""
        ds = ds_pts["standard"]
        with pytest.raises(InvalidDatasetStructure):
            ds.argo.interp_std_levels([20, 30, 40, 50])    

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
class Test_teos10:
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

    def test_teos10_variables_single(self, ds_pts):
        """Add a single TEOS10 variables"""
        for key, this in ds_pts.items():
            for format in ["point", "profile"]:
                that = this.copy()  # To avoid modifying the original dataset
                if format == "profile":
                    that = that.argo.point2profile()
                that = that.argo.teos10(["PV"])
                assert "PV" in that.variables

    def test_teos10_variables_inplace(self, ds_pts):
        """Compute all default variables to a new dataset"""
        for key, this in ds_pts.items():
            ds = this.argo.teos10(inplace=False)  # So "SA" must be in 'ds' but not in 'this'
            assert "SA" in ds.variables
            assert "SA" not in this.variables
