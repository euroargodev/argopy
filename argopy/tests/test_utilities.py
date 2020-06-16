import io
import pytest
import unittest
import argopy
import xarray as xr
import requests
from argopy.utilities import load_dict, mapp_dict, list_multiprofile_file_variables, \
    isconnected, erddap_ds_exists, open_etopo1

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


@unittest.skipUnless(CONNECTED, "open_etopo1 requires an internet connection")
def test_open_etopo1():
    try:
        ds = open_etopo1([-80, -79, 20, 21], res='l')
        assert isinstance(ds, xr.DataArray) is True
    except requests.HTTPError:  # not our fault
        pass
