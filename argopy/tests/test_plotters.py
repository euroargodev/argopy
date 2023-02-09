"""
This file covers the plotters module
We test plotting functions from IndexFetcher and DataFetcher
"""
import pytest
import logging
from typing import Callable

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
from ..plot import bar_plot, plot_trajectory, open_sat_altim_report, scatter_map
from argopy import DataFetcher as ArgoDataFetcher

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

if has_ipython:
    import IPython

log = logging.getLogger("argopy.tests.plot")


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

@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "eric", "bgc"], indirect=False)
def test_valid_dashboard(board_type):
    # Test types with 'base'
    assert isinstance(argopy.dashboard(type=board_type, url_only=True), str)

@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "eric", "argovis", "op", "ocean-ops", "bgc"], indirect=False)
def test_valid_dashboard_float(board_type):
    # Test types with 'wmo' (should be all)
    assert isinstance(argopy.dashboard(5904797, type=board_type, url_only=True), str)

@pytest.mark.parametrize("board_type", ["data", "meta", "ea", "eric", "argovis", "bgc"], indirect=False)
def test_valid_dashboard_profile(board_type):
    # Test types with 'cyc'
    assert isinstance(argopy.dashboard(5904797, 12, type=board_type, url_only=True), str)


@requires_ipython
@requires_connection
def test_valid_dashboard_ipython_output():

    dsh = argopy.dashboard()
    assert isinstance(dsh, IPython.lib.display.IFrame)

    dsh = argopy.dashboard(wmo=5904797)
    assert isinstance(dsh, IPython.lib.display.IFrame)

    dsh = argopy.dashboard(wmo=5904797, cyc=3)
    assert isinstance(dsh, IPython.lib.display.IFrame)


@requires_connection
@pytest.mark.parametrize("WMOs", [2901623, [2901623, 6901929]],
                         ids=['For unique WMO', 'For a list of WMOs'],
                         indirect=False)
@pytest.mark.parametrize("embed", ['dropdown', 'slide', 'list'],
                         indirect=False)
def test_open_sat_altim_report(WMOs, embed):
    dsh = open_sat_altim_report(WMO=WMOs, embed=embed)

    if not has_ipython:
        assert isinstance(dsh, dict)

    elif embed == 'dropdown' and has_ipywidgets:
        assert isinstance(dsh, Callable)
        assert isinstance(dsh(2901623), IPython.display.Image)

    elif embed == 'slide' and has_ipywidgets:
        assert isinstance(dsh, Callable)

    elif embed == 'list':
        assert dsh is None

@requires_gdac
@requires_matplotlib
class Test_plot_trajectory:
    src = "gdac"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        # "float": [[2901623], [2901623, 6901929, 5906072]],
        # "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0, 0., 10.],
            [-60, -40, 40.0, 60.0, 0., 10., "2007-08-01", "2007-09-01"],
        ],
    }

    opts = [(ws, wc, lg) for ws in [False, has_seaborn] for wc in [False, has_cartopy] for lg in [True, False]]
    opts_ids = ["with_seaborn=%s, with_cartopy=%s, add_legend=%s" % (opt[0], opt[1], opt[2]) for opt in opts]

    def __test_traj_plot(self, df, opts):
        with_seaborn, with_cartopy, with_legend = opts
        fig, ax = plot_trajectory(
            df, with_seaborn=with_seaborn, with_cartopy=with_cartopy, add_legend=with_legend
        )
        assert isinstance(fig, mpl.figure.Figure)

        expected_ax_type = (
            cartopy.mpl.geoaxes.GeoAxesSubplot
            if has_cartopy and with_cartopy
            else mpl.axes.Axes
        )
        assert isinstance(ax, expected_ax_type)

        expected_lg_type = mpl.legend.Legend if with_legend else type(None)
        assert isinstance(ax.get_legend(), expected_lg_type)

        mpl.pyplot.close(fig)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_region(self, opts):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg).load()
                self.__test_traj_plot(loader.index, opts)


@requires_gdac
@requires_matplotlib
class Test_bar_plot:
    src = "gdac"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        # "float": [[2901623], [2901623, 6901929, 5906072]],
        # "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0, 0., 10.],
            [-60, -40, 40.0, 60.0, 0., 10., "2007-08-01", "2007-09-01"],
        ],
    }

    opts = [(ws, by) for ws in [False, has_seaborn] for by in ['institution', 'profiler']]
    opts_ids = ["with_seaborn=%s, by=%s" % (opt[0], opt[1]) for opt in opts]

    def __test_bar_plot(self, df, opts):
        with_seaborn, by = opts
        fig, ax = bar_plot(df, by=by, with_seaborn=with_seaborn)
        assert isinstance(fig, mpl.figure.Figure)
        mpl.pyplot.close(fig)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_region(self, opts):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg)
                self.__test_bar_plot(loader.to_index(full=True), opts)


@requires_gdac
@requires_matplotlib
@requires_cartopy
class Test_scatter_map:
    src = "gdac"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        # "float": [[2901623], [2901623, 6901929, 5906072]],
        # "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0, 0., 10.],
            [-60, -40, 40.0, 60.0, 0., 10., "2007-08-01", "2007-09-01"],
        ],
    }

    opts = [(traj, lg) for traj in [True, False] for lg in [True, False]]
    opts_ids = ["traj=%s, legend=%s" % (opt[0], opt[1]) for opt in opts]

    def __test(self, data, axis, opts):
        traj, legend = opts
        fig, ax = scatter_map(
            data, x=axis[0], y=axis[1], hue=axis[2], traj=traj, traj_axis=axis[2], legend=legend
        )
        assert isinstance(fig, mpl.figure.Figure)

        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)

        expected_lg_type = mpl.legend.Legend if legend else type(None)
        assert isinstance(ax.get_legend(), expected_lg_type)

        mpl.pyplot.close(fig)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataset_of_points(self, opts):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg).load()
                self.__test(loader.data, ('LONGITUDE', 'LATITUDE', 'PLATFORM_NUMBER'), opts)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataset_of_profiles(self, opts):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg).load()
                self.__test(loader.data.argo.point2profile(), ('LONGITUDE', 'LATITUDE', 'PLATFORM_NUMBER'), opts)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataframe_of_index(self, opts):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg).load()
                self.__test(loader.index, ('longitude', 'latitude', 'wmo'), opts)
