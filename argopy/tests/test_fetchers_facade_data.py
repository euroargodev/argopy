import pandas as pd
import xarray as xr

import pytest

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    InvalidFetcherAccessPoint,
    InvalidFetcher,
    OptionValueError,
)
from argopy.utils import is_list_of_strings
from utils import (
    requires_fetcher,
    requires_connection,
    requires_connected_erddap_phy,
    requires_gdac,
    requires_connected_gdac,
    requires_connected_argovis,
    requires_ipython,
    safe_to_server_errors,
    requires_matplotlib,
    has_matplotlib,
    has_seaborn,
    has_cartopy,
    has_ipython,
)


if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

if has_ipython:
    import IPython

skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


@requires_gdac
class Test_Facade:

    # Use the first valid data source:
    src = 'gdac'
    src_opts = {'gdac': argopy.tutorial.open_dataset("gdac")[0]}

    def __get_fetcher(self, empty: bool = False, pt: str = 'profile'):
        f = ArgoDataFetcher(src=self.src, **self.src_opts)
        # f.valid_access_points[0]

        if pt == 'float':
            if not empty:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).float(2901623)
            else:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).float(12)

        if pt == 'profile':
            if not empty:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).profile(2901623, 12)
            else:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).profile(12, 1200)

        if pt == 'region':
            if not empty:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).region([-60, -55, 40.0, 45.0, 0.0, 10.0,
                                                                "2007-08-01", "2007-09-01"])
            else:
                return f, ArgoDataFetcher(src=self.src, **self.src_opts).region([-60, -55, 40.0, 45.0, 99.92, 99.99,
                                                                "2007-08-01", "2007-08-01"])

    def test_invalid_mode(self):
        with pytest.raises(OptionValueError):
            ArgoDataFetcher(src=self.src, mode='invalid').to_xarray()

    def test_invalid_source(self):
        with pytest.raises(OptionValueError):
            ArgoDataFetcher(src="invalid").to_xarray()

    def test_invalid_dataset(self):
        with pytest.raises(OptionValueError):
            ArgoDataFetcher(src=self.src, ds='invalid')

    def test_invalid_accesspoint(self):
        with pytest.raises(InvalidFetcherAccessPoint):
            self.__get_fetcher()[0].invalid_accesspoint.to_xarray()

    def test_no_uri(self):
        with pytest.raises(InvalidFetcherAccessPoint):
            self.__get_fetcher()[0].uri

    def test_to_xarray(self):
        assert isinstance(self.__get_fetcher()[1].to_xarray(), xr.Dataset)
        with pytest.raises(InvalidFetcher):
            assert self.__get_fetcher()[0].to_xarray()

    def test_to_dataframe(self):
        assert isinstance(self.__get_fetcher()[1].to_dataframe(), pd.core.frame.DataFrame)
        with pytest.raises(InvalidFetcher):
            assert self.__get_fetcher()[0].to_dataframe()

    params = [(p, c) for p in [True, False] for c in [False]]
    ids_params = ["full=%s, coriolis_id=%s" % (p[0], p[1]) for p in params]
    @pytest.mark.parametrize("params", params,
                             indirect=False,
                             ids=ids_params)
    def test_to_index(self, params):
        full, coriolis_id = params
        assert isinstance(self.__get_fetcher()[1].to_index(full=full, coriolis_id=coriolis_id), pd.core.frame.DataFrame)

    params = [(p, c) for p in [True, False] for c in [True]]
    ids_params = ["full=%s, coriolis_id=%s" % (p[0], p[1]) for p in params]
    @pytest.mark.parametrize("params", params,
                             indirect=False,
                             ids=ids_params)
    @requires_connection
    def test_to_index_coriolis(self, params):
        full, coriolis_id = params
        assert isinstance(self.__get_fetcher()[1].to_index(full=full, coriolis_id=coriolis_id), pd.core.frame.DataFrame)

    def test_load(self):
        f, fetcher = self.__get_fetcher(pt='float')

        fetcher.load()
        assert is_list_of_strings(fetcher.uri)
        assert isinstance(fetcher.data, xr.Dataset)
        assert isinstance(fetcher.index, pd.core.frame.DataFrame)

        # Change the access point:
        new_fetcher = f.profile(fetcher._AccessPoint_data['wmo'], 1)
        new_fetcher.load()
        assert is_list_of_strings(new_fetcher.uri)
        assert isinstance(new_fetcher.data, xr.Dataset)
        assert isinstance(new_fetcher.index, pd.core.frame.DataFrame)

    @requires_matplotlib
    def test_plot_trajectory(self):
        f, fetcher = self.__get_fetcher(pt='float')
        fig, ax = fetcher.plot(ptype='trajectory',
                               with_seaborn=has_seaborn,
                               with_cartopy=has_cartopy)
        assert isinstance(fig, mpl.figure.Figure)
        expected_ax_type = (
            cartopy.mpl.geoaxes.GeoAxesSubplot
            if has_cartopy
            else mpl.axes.Axes
        )
        assert isinstance(ax, expected_ax_type)
        assert isinstance(ax.get_legend(), mpl.legend.Legend)
        mpl.pyplot.close(fig)

    @requires_matplotlib
    @pytest.mark.parametrize("by", ["dac", "profiler"], indirect=False)
    def test_plot_bar(self, by):
        f, fetcher = self.__get_fetcher(pt='float')
        fig, ax = fetcher.plot(ptype=by, with_seaborn=has_seaborn)
        assert isinstance(fig, mpl.figure.Figure)
        mpl.pyplot.close(fig)

    @requires_matplotlib
    def test_plot_invalid(self):
        f, fetcher = self.__get_fetcher(pt='float')
        with pytest.raises(ValueError):
            fetcher.plot(ptype='invalid_cat')

    @requires_matplotlib
    def test_plot_qc_altimetry(self):
        f, fetcher = self.__get_fetcher(pt='float')
        dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
        if has_ipython:
            assert isinstance(dsh(0), IPython.display.Image)
        else:
            assert isinstance(dsh, dict)

    def test_domain(self):
        f, fetcher = self.__get_fetcher(pt='float')
        fetcher.domain

    def test_dashboard(self):
        f, fetcher = self.__get_fetcher(pt='float')
        assert isinstance(fetcher.dashboard(url_only=True), str)

        f, fetcher = self.__get_fetcher(pt='profile')
        assert isinstance(fetcher.dashboard(url_only=True), str)

        with pytest.warns(UserWarning):
            f, fetcher = self.__get_fetcher(pt='region')
            fetcher.dashboard(url_only=True)
