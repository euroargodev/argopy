import os
import pytest
import argopy
from argopy.options import OPTIONS


def test_invalid_option():
    with pytest.raises(ValueError):
        argopy.set_options(not_a_valid_options=True)


def test_data_src():
    with pytest.raises(ValueError):
        argopy.set_options(src="invalid_src")
    with argopy.set_options(src="erddap"):
        assert OPTIONS["src"]


def test_dataset():
    with pytest.raises(ValueError):
        argopy.set_options(dataset="invalid_ds")
    with argopy.set_options(dataset="phy"):
        assert OPTIONS["dataset"]
    with argopy.set_options(dataset="bgc"):
        assert OPTIONS["dataset"]
    with argopy.set_options(dataset="ref"):
        assert OPTIONS["dataset"]


def test_usermode():
    with pytest.raises(ValueError):
        argopy.set_options(mode="invalid_mode")
    with argopy.set_options(mode="standard"):
        assert OPTIONS["mode"]
    with argopy.set_options(mode="expert"):
        assert OPTIONS["mode"]


def test_local_ftp():
    with pytest.raises(ValueError):
        argopy.set_options(local_ftp="invalid_path")
    with argopy.set_options(local_ftp=os.path.expanduser("~")):
        assert OPTIONS["local_ftp"]
