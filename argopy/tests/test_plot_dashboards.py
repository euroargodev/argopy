"""
This file covers the argopy.plot.dashboards submodule
"""
import pytest
import logging

import argopy
from argopy.errors import InvalidDashboard
from utils import (
    requires_gdac,
    requires_localftp,
    requires_connection,
    requires_matplotlib,
    requires_ipython,
    requires_cartopy,
    has_matplotlib,
    has_seaborn,
    has_cartopy,
    has_ipython,
    has_ipywidgets,
)

if has_ipython:
    import IPython

log = logging.getLogger("argopy.tests.plot.dashboards")


@pytest.mark.parametrize("board_type", ["invalid", "argovis", "op", "ocean-ops", "coriolis"], indirect=False)
def test_invalid_dashboard(board_type):
    # Test types without 'base'
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(type=board_type, url_only=True)


@pytest.mark.parametrize("board_type", ["op", "ocean-ops", "coriolis"], indirect=False)
def test_invalid_dashboard_profile(board_type):
    # Test types without 'cyc'
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(6902755, 12, type=board_type, url_only=True)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "bgc"], indirect=False)
def test_valid_dashboard(board_type):
    # Test types with 'base'
    assert isinstance(argopy.dashboard(type=board_type, url_only=True), str)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "argovis", "op", "ocean-ops", "bgc"], indirect=False)
def test_valid_dashboard_float(board_type):
    # Test types with 'wmo' (should be all)
    assert isinstance(argopy.dashboard(5904797, type=board_type, url_only=True), str)


@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "argovis", "bgc"], indirect=False)
def test_valid_dashboard_profile(board_type):
    # Test types with 'cyc'
    assert isinstance(argopy.dashboard(5904797, 12, type=board_type, url_only=True), str)


@requires_ipython
@requires_connection
@pytest.mark.parametrize("opts", [{}, {'wmo': 5904797}, {'wmo': 5904797, 'cyc': 3}],
                         ids=['', 'WMO', 'WMO, CYC'],
                         indirect=False)
def test_valid_dashboard_ipython_output(opts):
    dsh = argopy.dashboard(**opts)
    assert isinstance(dsh, IPython.lib.display.IFrame)

