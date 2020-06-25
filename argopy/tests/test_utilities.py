import os
import io
import pytest
import unittest
import argopy
import xarray as xr
import numpy as np
import requests
import shutil
from argopy.utilities import load_dict, mapp_dict, list_multiprofile_file_variables, \
    isconnected, erddap_ds_exists, open_etopo1, list_available_data_src, linear_interpolation_remap
from argopy import DataFetcher as ArgoDataFetcher

AVAILABLE_SOURCES = list_available_data_src()
CONNECTED = isconnected()


def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def test_invalid_dictionnary():
    with pytest.raises(ValueError):
        load_dict("invalid_dictionnary")


def test_invalid_dictionnary_key():
    d = load_dict('profilers')
    assert mapp_dict(d, "invalid_key") == "Unknown"


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())


def test_show_versions():
    f = io.StringIO()
    argopy.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()


def test_isconnected():
    assert isinstance(isconnected(), bool)
    assert isconnected(host="http://dummyhost") is False


def test_erddap_ds_exists():
    assert isinstance(erddap_ds_exists(ds='ArgoFloats'), bool)
    assert erddap_ds_exists(ds='DummyDS') is False


@unittest.skipUnless('localftp' in AVAILABLE_SOURCES, "requires localftp data fetcher")
def test_clear_cache():
    # Fetch data to cache:
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    testcachedir = os.path.expanduser(os.path.join("~", ".argopytest_tmp"))
    with argopy.set_options(cachedir=testcachedir, local_ftp=ftproot):
        ArgoDataFetcher(src='localftp').profile(2902696, 12).to_xarray()
    # Then clean it:
    argopy.clear_cache()
    assert os.path.isdir(testcachedir) is False

# We disable this test because the server has not responded over a week (May 29th)
# @unittest.skipUnless(CONNECTED, "open_etopo1 requires an internet connection")
# def test_open_etopo1():
#     try:
#         ds = open_etopo1([-80, -79, 20, 21], res='l')
#         assert isinstance(ds, xr.DataArray) is True
#     except requests.HTTPError:  # not our fault
#         pass


class test_linear_interpolation_remap(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def create_data(self):
        # create fake data to test interpolation:
        temp = np.random.rand(200, 100)
        pres = np.sort(np.floor(np.zeros(
            [200, 100])+np.linspace(50, 950, 100)+np.random.randint(-5, 5, [200, 100])))
        self.dsfake = xr.Dataset({'TEMP': (['N_PROF', 'N_LEVELS'], temp),
                                  'PRES': (['N_PROF', 'N_LEVELS'],  pres)},
                                 coords={'N_PROF': ('N_PROF', range(200)),
                                         'N_LEVELS': ('N_LEVELS', range(100)),
                                         'Z_LEVELS': ('Z_LEVELS', np.arange(100, 900, 20))})

    def test_interpolation(self):
        # Run it with success:
        dsi = linear_interpolation_remap(
            self.dsfake.PRES, self.dsfake['TEMP'], self.dsfake['Z_LEVELS'], z_dim='N_LEVELS', z_regridded_dim='Z_LEVELS')
        assert 'remapped' in dsi.dims

    def test_error_zdim(self):
        # Test error:
        # catches error from _regular_interp linked to z_dim
        with pytest.raises(RuntimeError):
            dsi = linear_interpolation_remap(
                self.dsfake.PRES, self.dsfake['TEMP'], self.dsfake['Z_LEVELS'], z_regridded_dim='Z_LEVELS')

    def test_error_ds(self):
        # Test error:
        # catches error from linear_interpolation_remap linked to datatype
        with pytest.raises(ValueError):
            dsi = linear_interpolation_remap(
                self.dsfake.PRES, self.dsfake, self.dsfake['Z_LEVELS'], z_dim='N_LEVELS', z_regridded_dim='Z_LEVELS')
