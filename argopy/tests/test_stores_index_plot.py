import pytest
import logging
import importlib

import argopy
from argopy.stores import ArgoIndex

from utils import (
    requires_gdac,
    requires_matplotlib,
    requires_cartopy,
    has_matplotlib,
    has_cartopy,
)

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

log = logging.getLogger("argopy.tests.indexstores.plot")
argopy.clear_cache()


"""
Select GDAC host to be use for plot accessor test 
"""
VALID_HOST = argopy.tutorial.open_dataset("gdac")[0]  # Use local files
# 'http1': mocked_server_address,  # Use the mocked http server
# 'http2': 'https://data-argo.ifremer.fr',
# 'ftp': "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)

"""
List WMO to be tested, one for each mission
"""
VALID_WMO = [13857, 3902131]


@requires_gdac
@requires_matplotlib
class Test_IndexStore_PlotAccessor:

    @requires_cartopy
    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_plot_trajectory(self, wmo):
        fig, ax = ArgoIndex(host=VALID_HOST, cache=True).query.wmo(wmo).plot.trajectory()
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)
        mpl.pyplot.close(fig)

    @pytest.mark.parametrize(
        "by", ['dac'], indirect=False, ids=[f"by='{w}'" for w in ['dac']]
    )
    @pytest.mark.parametrize(
        "index", [True, False], indirect=False, ids=[f"index={c}" for c in [True, False]]
    )
    def test_plot_bar(self, by, index):
        idx = ArgoIndex(host=VALID_HOST, cache=True)

        fig, ax = idx.plot.bar(by=by, index=index)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)
        mpl.pyplot.close(fig)


    def test_plot_bar_errors(self):
        idx = ArgoIndex(host=VALID_HOST, cache=True)

        with pytest.raises(ValueError):
            idx.plot.bar(by="NOT_A_VALID_PARAM")
