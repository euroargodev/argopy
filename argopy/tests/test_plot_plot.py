"""
This file covers the argopy.plot.plot submodule
"""
import pytest
import logging
from typing import Callable

import argopy
from utils import (
    requires_gdac,
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

from argopy.plot import bar_plot, plot_trajectory, open_sat_altim_report, scatter_map
from argopy.errors import InvalidDatasetStructure
from argopy import DataFetcher as ArgoDataFetcher
from mocked_http import mocked_server_address

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

if has_ipython:
    import IPython


log = logging.getLogger("argopy.tests.plot.plot")
argopy.clear_cache()


class Test_open_sat_altim_report:
    WMOs = [2901623, [2901623, 6901929]]

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    @pytest.mark.parametrize("WMOs", WMOs,
                             ids=['For unique WMO', 'For a list of WMOs'],
                             indirect=False)
    @pytest.mark.parametrize("embed", ['dropdown', 'slide', 'list', None],
                             indirect=False)
    def test_open_sat_altim_report(self, WMOs, embed):
        if has_ipython:
            import IPython

        dsh = open_sat_altim_report(WMO=WMOs, embed=embed, api_server=mocked_server_address)

        if has_ipython and embed is not None:

            if has_ipywidgets:

                if embed == "dropdown":
                    assert isinstance(dsh, Callable)
                    assert isinstance(dsh(2901623), IPython.display.Image)

                if embed == "slide":
                    assert isinstance(dsh, Callable)

            else:
                assert dsh is None

        else:
            assert isinstance(dsh, dict)

    @requires_ipython
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            open_sat_altim_report(WMO=self.WMOs[0], embed='dummy_method', api_server=mocked_server_address)


@requires_gdac
@requires_matplotlib
class Test_plot_trajectory:
    src = "gdac"
    local_gdac = argopy.tutorial.open_dataset("gdac")[0]
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
        with argopy.set_options(src=self.src, gdac=self.local_gdac):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher(cache=True).region(arg).load()
                self.__test_traj_plot(loader.index, opts)


@requires_gdac
@requires_matplotlib
class Test_bar_plot:
    src = "gdac"
    local_gdac = argopy.tutorial.open_dataset("gdac")[0]
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
        with argopy.set_options(src=self.src, gdac=self.local_gdac):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher().region(arg)
                self.__test_bar_plot(loader.to_index(full=True), opts)


@requires_gdac
@requires_matplotlib
@requires_cartopy
class Test_scatter_map:
    src = "gdac"
    local_gdac = argopy.tutorial.open_dataset("gdac")[0]
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
        fig, ax, hdl = scatter_map(
            data, x=axis[0], y=axis[1], hue=axis[2], traj=traj, traj_axis=axis[2], legend=legend
        )
        assert isinstance(fig, mpl.figure.Figure)

        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)

        assert isinstance(hdl, dict)

        expected_lg_type = mpl.legend.Legend if legend else type(None)
        assert isinstance(ax.get_legend(), expected_lg_type)

        mpl.pyplot.close(fig)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataset_of_points(self, opts):
        with argopy.set_options(src=self.src, gdac=self.local_gdac):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher(cache=True).region(arg).load()
                with pytest.raises(InvalidDatasetStructure):
                    self.__test(loader.data, (None, None, None), opts)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataset_of_profiles(self, opts):
        with argopy.set_options(src=self.src, gdac=self.local_gdac):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher(cache=True).region(arg).load()
                dsp = loader.data.argo.point2profile()
                # with pytest.warns(UserWarning):
                #     self.__test(dsp, (None, None, None), opts)
                self.__test(dsp.isel(N_LEVELS=0), (None, None, None), opts)

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_with_a_dataframe_of_index(self, opts):
        with argopy.set_options(src=self.src, gdac=self.local_gdac):
            for arg in self.requests["region"]:
                loader = ArgoDataFetcher(cache=True).region(arg).load()
                self.__test(loader.index, (None, None, None), opts)
