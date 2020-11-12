import pytest

import argopy
from argopy.errors import InvalidDashboard
from . import requires_connection, requires_mpl, has_mpl
from argopy.plotters import discrete_coloring


if has_mpl:
    import matplotlib.colors as mcolors


@requires_connection
def test_invalid_dashboard():
    with pytest.raises(InvalidDashboard):
        argopy.dashboard(wmo=5904797, type='invalid_service')


@requires_connection
def test_valid_dashboard():
    import IPython
    dsh = argopy.dashboard(wmo=5904797)
    assert isinstance(dsh, IPython.lib.display.IFrame)


@requires_mpl
def test_discrete_coloring():
    dc = discrete_coloring()
    assert isinstance(dc.cmap, mcolors.LinearSegmentedColormap)