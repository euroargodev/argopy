import pytest

import argopy
from argopy.errors import InvalidDashboard
from . import requires_connection


@requires_connection
def test_invalid_dashboard():
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(wmo=5904797, type='invalid_service')


@requires_connection
def test_valid_dashboard():
    import IPython
    dsh = argopy.dashboard(wmo=5904797)
    assert isinstance(dsh, IPython.lib.display.IFrame)
