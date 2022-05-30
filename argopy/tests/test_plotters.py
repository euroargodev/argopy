"""
This file covers the plotters module
We test plotting functions from IndexFetcher and DataFetcher
"""
import pytest
import importlib
import logging

import argopy
from argopy.errors import InvalidDashboard
from utils import (
    requires_gdac,
    requires_localftp,
    requires_connection,
    requires_matplotlib,
    requires_ipython,
    has_matplotlib,
    has_seaborn,
    has_cartopy,
    has_ipython,
)
from ..plot import bar_plot, plot_trajectory, open_sat_altim_report
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy import DataFetcher as ArgoDataFetcher

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

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
    import IPython

    dsh = argopy.dashboard()
    assert isinstance(dsh, IPython.lib.display.IFrame)

    dsh = argopy.dashboard(wmo=5904797)
    assert isinstance(dsh, IPython.lib.display.IFrame)

    dsh = argopy.dashboard(wmo=5904797, cyc=3)
    assert isinstance(dsh, IPython.lib.display.IFrame)


@requires_connection
def test_open_sat_altim_report():
    if has_ipython:
        import IPython

    dsh = open_sat_altim_report(WMO=5904797, embed='dropdown')
    if has_ipython:
        assert isinstance(dsh(5904797), IPython.display.Image)
    else:
        assert isinstance(dsh, dict)


@requires_gdac
@requires_matplotlib
class Test_index_plot:
    src = "gdac"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        "float": [[2901623], [2901623, 6901929, 5906072]],
        "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0],
            [-60, -40, 40.0, 60.0, "2007-08-01", "2007-09-01"],
        ],
    }

    def __test_traj_plot(self, df):
        for ws in [False, has_seaborn]:
            for wc in [False, has_cartopy]:
                for legend in [True, False]:
                    fig, ax = plot_trajectory(
                        df, with_seaborn=ws, with_cartopy=wc, add_legend=legend
                    )
                    assert isinstance(fig, mpl.figure.Figure)

                    expected_ax_type = (
                        cartopy.mpl.geoaxes.GeoAxesSubplot
                        if has_cartopy and wc
                        else mpl.axes.Axes
                    )
                    assert isinstance(ax, expected_ax_type)

                    expected_lg_type = mpl.legend.Legend if legend else type(None)
                    assert isinstance(ax.get_legend(), expected_lg_type)

                    mpl.pyplot.close(fig)

    def __test_bar_plot(self, df):
        for ws in [False, has_seaborn]:
            for by in [
                "institution",
                "profiler",
                "ocean"
            ]:
                fig, ax = bar_plot(df, by=by, with_seaborn=ws)
                assert isinstance(fig, mpl.figure.Figure)
                mpl.pyplot.close(fig)

    def test_traj_plot_region(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoIndexFetcher().region(arg).load()
                self.__test_traj_plot(loader.index)

    def test_traj_plot_float(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["float"]:
                loader = ArgoIndexFetcher().float(arg).load()
                self.__test_traj_plot(loader.index)

    def test_traj_plot_profile(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["profile"]:
                loader = ArgoIndexFetcher().profile(*arg).load()
                self.__test_traj_plot(loader.index)

    def test_bar_plot_region(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoIndexFetcher().region(arg).load()
                self.__test_bar_plot(loader.index)

    def test_bar_plot_float(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["float"]:
                loader = ArgoIndexFetcher().float(arg).load()
                self.__test_bar_plot(loader.index)

    def test_bar_plot_profile(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["profile"]:
                loader = ArgoIndexFetcher().profile(*arg).load()
                self.__test_bar_plot(loader.index)


@requires_gdac
@requires_matplotlib
class Test_data_plot:
    src = "gdac"
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    requests = {
        "float": [[2901623], [2901623, 6901929, 5906072]],
        "profile": [[2901623, 12], [6901929, [5, 45]]],
        "region": [
            [-60, -40, 40.0, 60.0, 0., 10.],
            [-60, -40, 40.0, 60.0, 0., 10., "2007-08-01", "2007-09-01"],
        ],
    }

    def __test_traj_plot(self, df):
        for ws in [False, has_seaborn]:
            for wc in [False, has_cartopy]:
                for legend in [True, False]:
                    fig, ax = plot_trajectory(
                        df, with_seaborn=ws, with_cartopy=wc, add_legend=legend
                    )
                    assert isinstance(fig, mpl.figure.Figure)

                    expected_ax_type = (
                        cartopy.mpl.geoaxes.GeoAxesSubplot
                        if has_cartopy and wc
                        else mpl.axes.Axes
                    )
                    assert isinstance(ax, expected_ax_type)

                    expected_lg_type = mpl.legend.Legend if legend else type(None)
                    assert isinstance(ax.get_legend(), expected_lg_type)

                    mpl.pyplot.close(fig)

    def __test_bar_plot(self, df):
        for ws in [False, has_seaborn]:
            for by in [
                "institution",
                "profiler"
            ]:
                fig, ax = bar_plot(df, by=by, with_seaborn=ws)
                assert isinstance(fig, mpl.figure.Figure)
                mpl.pyplot.close(fig)

    def test_traj_plot_region(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg).load()
                self.__test_traj_plot(loader.index)

    def test_traj_plot_float(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["float"]:
                loader = ArgoDataFetcher().float(arg).load()
                self.__test_traj_plot(loader.index)

    def test_traj_plot_profile(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["profile"]:
                loader = ArgoDataFetcher().profile(*arg).load()
                self.__test_traj_plot(loader.index)

    def test_bar_plot_region(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg)
                self.__test_bar_plot(loader.to_index(full=True))

    def test_bar_plot_float(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["float"]:
                loader = ArgoDataFetcher().float(arg)
                self.__test_bar_plot(loader.to_index(full=True))

    def test_bar_plot_profile(self):
        with argopy.set_options(src=self.src, ftp=self.local_ftp):
            for arg in self.requests["profile"]:
                loader = ArgoDataFetcher().profile(*arg)
                self.__test_bar_plot(loader.to_index(full=True))
