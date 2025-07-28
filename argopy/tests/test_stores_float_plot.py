import pytest
import logging

import argopy
from argopy.stores import ArgoFloat

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

log = logging.getLogger("argopy.tests.floatstore.plot")
argopy.clear_cache()

skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")
skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")


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


@skip_offline
@requires_gdac
@requires_matplotlib
class Test_FloatStore_PlotAccessor:

    @requires_cartopy
    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_plot_trajectory(self, wmo):
        fig, ax, hdl = ArgoFloat(wmo, host=VALID_HOST, cache=True).plot.trajectory()
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)
        mpl.pyplot.close(fig)

    def test_plot_map_errors(self):
        af = ArgoFloat(13857, host=VALID_HOST, cache=True)
        with pytest.raises(ValueError):
            af.plot.map("NOT_A_CORE_PARAM", ds="prof")
        with pytest.raises(ValueError):
            af.plot.map("NOT_A_BGC_PARAM", ds="Sprof")
        with pytest.raises(ValueError):
            af.plot.map("DOXY", ds="prof")

    @requires_cartopy
    @pytest.mark.parametrize(
        "wmo", [VALID_WMO[0]], indirect=False, ids=[f"wmo={w}" for w in [VALID_WMO[0]]]
    )
    def test_plot_map(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        fig, ax, hdl = af.plot.map("TEMP", ds="prof")
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)
        mpl.pyplot.close(fig)

    def test_plot_scatter_errors(self):
        af = ArgoFloat(13857, host=VALID_HOST, cache=True)
        with pytest.raises(ValueError):
            af.plot.scatter("NOT_A_CORE_PARAM", ds="prof")
        with pytest.raises(ValueError):
            af.plot.scatter("NOT_A_BGC_PARAM", ds="Sprof")
        with pytest.raises(ValueError):
            af.plot.scatter("DOXY", ds="prof")

    @pytest.mark.parametrize(
        "wmo", [VALID_WMO[0]], indirect=False, ids=[f"wmo={w}" for w in [VALID_WMO[0]]]
    )
    @pytest.mark.parametrize(
        "cbar", [True, False], indirect=False, ids=[f"cbar={c}" for c in [True, False]]
    )
    def test_plot_scatter(self, wmo, cbar):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)

        if cbar:
            fig, ax, m, cbar = af.plot.scatter("TEMP", ds="prof", cbar=cbar)
        else:
            fig, ax, m = af.plot.scatter("TEMP", ds="prof", cbar=cbar)

        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)
        mpl.pyplot.close(fig)
