import pytest
import unittest

from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import InvalidDatasetStructure, ErddapServerError

from argopy.utilities import list_available_data_src, isconnected, erddap_ds_exists
AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()
if CONNECTED:
    DSEXISTS = erddap_ds_exists(ds="ArgoFloats")
else:
    DSEXISTS = False


@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_point2profile():
    try:
        ds = ArgoDataFetcher(src='erddap')\
                    .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                    .to_xarray()
        assert 'N_PROF' in ds.argo.point2profile().dims
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass


@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_profile2point():
    try:
        ds = ArgoDataFetcher(src='erddap')\
                    .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                    .to_xarray()
        with pytest.raises(InvalidDatasetStructure):
            ds.argo.profile2point()
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass


@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
def test_point2profile2point():
    try:
        ds_pts = ArgoDataFetcher(src='erddap') \
            .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15']) \
            .to_xarray()
        assert ds_pts.argo.point2profile().argo.profile2point().equals(ds_pts)
    except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
        pass


@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
class test_interp_std_levels(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_data(self):
        # Fetch real data to test interpolation
        try:
            self.ds_pts_standard = ArgoDataFetcher(src='erddap', mode='standard')\
                .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                .to_xarray()
            self.ds_pts_expert = ArgoDataFetcher(src='erddap', mode='expert')\
                .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
                .to_xarray()
        except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
            pass
        except ValueError:  # Catches value error for incorrect standard levels as inputs
            pass

    def test_interpolation(self):
        # Run with success:
        ds = self.ds_pts_standard.argo.point2profile()
        assert 'PRES_INTERPOLATED' in ds.argo.interp_std_levels(
            [20, 30, 40, 50]).dims

    def test_points_error(self):
        # Try to interpolate points, not profiles
        ds = self.ds_pts_standard
        with pytest.raises(InvalidDatasetStructure):
            ds.argo.interp_std_levels([20, 30, 40, 50])

    def test_mode_error(self):
        # Try to interpolate expert data
        ds = self.ds_pts_expert.argo.point2profile()
        with pytest.raises(InvalidDatasetStructure):
            ds.argo.interp_std_levels([20, 30, 40, 50]).dims

    def test_std_error(self):
        # Try to interpolate on a wrong axis
        ds = self.ds_pts_standard.argo.point2profile()
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels([100, 20, 30, 40, 50])
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels([-20, 20, 30, 40, 50])
        with pytest.raises(ValueError):
            ds.argo.interp_std_levels(12)


@unittest.skipUnless('erddap' in AVAILABLE_SOURCES, "requires erddap data fetcher")
@unittest.skipUnless(CONNECTED, "erddap requires an internet connection")
@unittest.skipUnless(DSEXISTS, "erddap requires a valid core Argo dataset from Ifremer server")
class test_teos10(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def init_data(self):
        # Fetch real data to test interpolation
        try:
            # self.ds_pts_standard = ArgoDataFetcher(src='erddap', mode='standard')\
            #     .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
            #     .to_xarray()
            # self.ds_pts_expert = ArgoDataFetcher(src='erddap', mode='expert')\
            #     .region([-75, -55, 30., 40., 0, 100., '2011-01-01', '2011-01-15'])\
            #     .to_xarray()
            self.ds_pts_standard = ArgoDataFetcher(src='erddap', mode='standard')\
                .float(2901623)\
                .to_xarray()
            self.ds_pts_expert = ArgoDataFetcher(src='erddap', mode='expert')\
                .float(2901623)\
                .to_xarray()
        except ErddapServerError:  # Test is passed when something goes wrong because of the erddap server, not our fault !
            pass
        except ValueError:  # Catches value error for incorrect standard levels as inputs
            pass

    def test_teos10_variables_default(self):
        ds_list = [self.ds_pts_standard, self.ds_pts_expert]
        for this in ds_list:
            for format in ['point', 'profile']:
                that = this
                if format == 'profile':
                    that = that.argo.point2profile()
                that = that.argo.teos10()
                assert 'SA' in that.variables
                assert 'CT' in that.variables

    def test_teos10_variables_single(self):
        ds_list = [self.ds_pts_standard, self.ds_pts_expert]
        for this in ds_list:
            for format in ['point', 'profile']:
                that = this
                if format == 'profile':
                    that = that.argo.point2profile()
                that = that.argo.teos10(['PV'])
                assert 'PV' in that.variables

    def test_teos10_variables_inplace(self):
        ds_list = [self.ds_pts_standard, self.ds_pts_expert]
        for this in ds_list:
            ds = this.argo.teos10(inplace=False)
            assert 'SA' in ds.variables
            assert 'SA' not in this.variables
