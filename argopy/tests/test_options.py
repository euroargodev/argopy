import os
import pytest
import warnings
import argopy
from argopy.options import OPTIONS
from argopy.errors import OptionValueError, FtpPathError, ErddapPathError
from utils import requires_gdac, create_read_only_folder
from mocked_http import mocked_httpserver, mocked_server_address
import logging


log = logging.getLogger("argopy.tests.options")


def test_invalid_option_name():
    with pytest.raises(ValueError):
        argopy.set_options(not_a_valid_options=True)


def test_opt_src():
    with pytest.raises(OptionValueError):
        argopy.set_options(src="invalid_src")
    with argopy.set_options(src="erddap"):
        assert OPTIONS["src"]


@requires_gdac
def test_opt_gdac_ftp():
    with pytest.raises(FtpPathError):
        argopy.set_options(ftp="invalid_path")

    local_ftp = argopy.tutorial.open_dataset("gdac")[0]
    with argopy.set_options(ftp=local_ftp):
        assert OPTIONS["ftp"] == local_ftp


def test_opt_ifremer_erddap(mocked_httpserver):
    with pytest.raises(ErddapPathError):
        argopy.set_options(erddap="invalid_path")

    with argopy.set_options(erddap=mocked_server_address):
        assert OPTIONS["erddap"] == mocked_server_address


def test_opt_dataset():
    with pytest.raises(OptionValueError):
        argopy.set_options(dataset="invalid_ds")
    with argopy.set_options(dataset="phy"):
        assert OPTIONS["dataset"] == "phy"
    with argopy.set_options(dataset="bgc"):
        assert OPTIONS["dataset"] == "bgc"
    with argopy.set_options(dataset="ref"):
        assert OPTIONS["dataset"] == "ref"


# @pytest.mark.skipif(True, reason="Need to be debugged for Windows support")
def test_opt_invalid_cachedir():
    # Cachedir is created if not exist.
    # OptionValueError is raised when it's not writable
    folder_name = "read_only_folder"
    create_read_only_folder(folder_name)
    warnings.warn(os.stat(folder_name).st_mode)
    warnings.warn(os.access(folder_name, os.W_OK))
    with pytest.raises(OptionValueError):
        argopy.set_options(cachedir=folder_name)
    os.rmdir(folder_name)


def test_opt_cachedir():
    with argopy.set_options(cachedir=os.path.expanduser("~")):
        assert OPTIONS["cachedir"]


def test_opt_cache_expiration():
    with pytest.raises(OptionValueError):
        argopy.set_options(cache_expiration="dummy")
    with pytest.raises(OptionValueError):
        argopy.set_options(cache_expiration=-3600)
    with argopy.set_options(cache_expiration=60):
        assert OPTIONS["cache_expiration"]


def test_opt_mode():
    with pytest.raises(OptionValueError):
        argopy.set_options(mode="invalid_mode")
    with argopy.set_options(mode="standard"):
        assert OPTIONS["mode"]
    with argopy.set_options(mode="expert"):
        assert OPTIONS["mode"]


def test_opt_api_timeout():
    with pytest.raises(OptionValueError):
        argopy.set_options(api_timeout='toto')
    with pytest.raises(OptionValueError):
        argopy.set_options(api_timeout=-12)


def test_opt_trust_env():
    with pytest.raises(ValueError):
        argopy.set_options(trust_env='toto')
    with pytest.raises(ValueError):
        argopy.set_options(trust_env=0)
