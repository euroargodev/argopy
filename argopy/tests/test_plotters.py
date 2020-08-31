import pytest
import unittest

import argopy
from argopy.errors import InvalidDashboard

from argopy.utilities import isconnected
CONNECTED = isconnected()


@unittest.skipUnless(CONNECTED, "notebook dashboard requires an internet connection")
def test_invalid_dashboard():
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(wmo=5904797, type='invalid_service')


@unittest.skipUnless(CONNECTED, "notebook dashboard requires an internet connection")
def test_valid_dashboard():
    import IPython
    dsh = argopy.dashboard(wmo=5904797)
    assert isinstance(dsh, IPython.lib.display.IFrame)
