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
    requires_cartopy,
    has_matplotlib,
    has_seaborn,
    has_cartopy,
    has_ipython,
)
from ..plot import scatter_map
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy import DataFetcher as ArgoDataFetcher

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

log = logging.getLogger("argopy.tests.plot")


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
