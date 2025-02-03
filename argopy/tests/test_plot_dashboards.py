"""
This file covers the argopy.plot.dashboards submodule
"""
import pytest
import logging

import argopy
from argopy.errors import InvalidDashboard
from utils import (
    requires_connection,
    requires_ipython,
    has_ipython,
    create_temp_folder,
)
from mocked_http import mocked_httpserver, mocked_server_address

if has_ipython:
    import IPython

log = logging.getLogger("argopy.tests.plot.dashboards")


@pytest.mark.parametrize("board_type", ["invalid", "op", "ocean-ops", "coriolis"], indirect=False)
def test_invalid_dashboard(board_type):
    # Test types without 'base'
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(type=board_type, url_only=True)


@pytest.mark.parametrize("board_type", ["op", "ocean-ops", "coriolis"], indirect=False)
def test_invalid_dashboard_profile(board_type):
    # Test types without 'cyc'
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(6902755, 12, type=board_type, url_only=True)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "argovis", "bgc"], indirect=False)
def test_valid_dashboard(board_type):
    # Test types with 'base'
    assert isinstance(argopy.dashboard(type=board_type, url_only=True), str)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "argovis", "op", "ocean-ops", "bgc"], indirect=False)
def test_valid_dashboard_float(board_type, mocked_httpserver):
    # Test types with 'wmo' (should be all)
    with argopy.set_options(server=mocked_server_address):
        assert isinstance(argopy.dashboard(6901929, type=board_type, url_only=True), str)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "argovis", "bgc"], indirect=False)
def test_valid_dashboard_profile(board_type, mocked_httpserver):
    # Test types with 'cyc'
    with create_temp_folder() as cachedir:
        with argopy.set_options(cachedir=cachedir, server=mocked_server_address):
            assert isinstance(argopy.dashboard(5904797, 12, type=board_type, url_only=True), str)


@requires_ipython
@pytest.mark.parametrize("opts", [{}, {'wmo': 6901929}, {'wmo': 6901929, 'cyc': 3}],
                         ids=['', 'WMO', 'WMO, CYC'],
                         indirect=False)
def test_valid_dashboard_ipython_output(opts, mocked_httpserver):
    with create_temp_folder() as cachedir:
        with argopy.set_options(cachedir=cachedir, server=mocked_server_address):
            dsh = argopy.dashboard(**opts)
            assert isinstance(dsh, IPython.lib.display.IFrame)
