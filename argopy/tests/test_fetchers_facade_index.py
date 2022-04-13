import pandas as pd
import pytest
import importlib

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import InvalidFetcherAccessPoint, InvalidFetcher
from utils import (
    # AVAILABLE_INDEX_SOURCES,
    requires_fetcher_index,
    requires_connected_erddap_index,
    requires_localftp_index,
    requires_connected_gdac,
    requires_connection,
    requires_ipython,
    safe_to_server_errors,
    ci_erddap_index,
    requires_matplotlib,
    has_matplotlib,
    has_seaborn,
    has_cartopy
)


if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy


ERDDAP_TIMEOUT = 3 * 60
skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


@requires_connection
class Test_Facade:
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    src = 'localftp'
    # src = list(AVAILABLE_INDEX_SOURCES.keys())[0]

    def __get_fetcher(self, empty: bool = False, pt: str = 'profile'):
        f = ArgoIndexFetcher(src=self.src)
        # f.valid_access_points[0]

        if pt == 'float':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src).float(2901623)
            else:
                return f, ArgoIndexFetcher(src=self.src).float(12)

        if pt == 'profile':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src).profile(2901623, 12)
            else:
                return f, ArgoIndexFetcher(src=self.src).profile(12, 1200)

        if pt == 'region':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src).region([-60, -55, 40.0, 45.0, 0.0, 10.0,
                                                                "2007-08-01", "2007-09-01"])
            else:
                return f, ArgoIndexFetcher(src=self.src).region([-60, -55, 40.0, 45.0, 99.92, 99.99,
                                                                "2007-08-01", "2007-08-01"])

    def test_invalid_fetcher(self):
        with pytest.raises(InvalidFetcher):
            ArgoIndexFetcher(src="invalid_fetcher").to_xarray()

    @requires_fetcher_index
    def test_invalid_accesspoint(self):
        # Use the first valid data source
        with pytest.raises(InvalidFetcherAccessPoint):
            ArgoIndexFetcher(
                src=self.src
            ).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
        with pytest.raises(InvalidFetcherAccessPoint):
            ArgoIndexFetcher(
                src=self.src
            ).to_xarray()  # Can't get data if access point not defined first

    @requires_fetcher_index
    def test_invalid_dataset(self):
        with pytest.raises(ValueError):
            ArgoIndexFetcher(src=self.src, ds='dummy_ds')

    @requires_matplotlib
    def test_plot(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            f, fetcher = self.__get_fetcher(pt='float')

            # Test 'trajectory'
            for ws in [False, has_seaborn]:
                for wc in [False, has_cartopy]:
                    for legend in [True, False]:
                        fig, ax = fetcher.plot(ptype='trajectory', with_seaborn=ws, with_cartopy=wc, add_legend=legend)
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

            # Test 'dac', 'profiler'
            for ws in [False, has_seaborn]:
                for by in [
                    "dac",
                    "profiler"
                ]:
                    fig, ax = fetcher.plot(ptype=by, with_seaborn=ws)
                    assert isinstance(fig, mpl.figure.Figure)
                    mpl.pyplot.close(fig)

            # Test 'qc_altimetry'
            if importlib.util.find_spec('IPython') is not None:
                import IPython
                dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
                assert isinstance(dsh(0), IPython.display.Image)

            # Test invalid plot
            with pytest.raises(ValueError):
                fetcher.plot(ptype='invalid_cat', with_seaborn=ws)

    @requires_ipython
    @requires_matplotlib
    def test_plot_qc_altimetry(self):
        import IPython
        with argopy.set_options(local_ftp=self.local_ftp):
            f, fetcher = self.__get_fetcher(pt='float')
            dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
            assert isinstance(dsh(0), IPython.display.Image)


"""
The following tests are not necessary, since index fetching is tested from each index fetcher tests 
"""
@requires_connection
@requires_fetcher_index
class OFFTest_IndexFetching:
    """ Test main API facade for all available index fetching backends """

    local_ftp = argopy.tutorial.open_dataset("localftp")[0]

    # todo Determine the list of output format to test
    # what else beyond .to_xarray() ?

    fetcher_opts = {}

    # Define API entry point options to tests:
    # These should be available online and with the argopy-data dummy gdac ftp
    args = {}
    # args["float"] = [[2901623], [6901929, 2901623]]
    # args["region"] = [
    #     [-100., -95., -35., -30.],
    #     [-100., -95., -35., -30., "2020-01-01", "2020-01-15"],
    # ]
    # args["profile"] = [[2901623, 2], [6901929, [5, 45]]]
    args["float"] = [[13857]]
    args["profile"] = [[13857, 90], [13857, [90, 91]]]
    args["region"] = [
        [-20, -16., 0, 1.],
        [-20, -16., 0, 1., "1997-07-01", "1997-09-01"],
    ]

    def __test_float(self, bk, **ftc_opts):
        """ Test float index fetching for a given backend """
        for arg in self.args["float"]:
            options = {**self.fetcher_opts, **ftc_opts}
            f = ArgoIndexFetcher(src=bk, **options).float(arg)
            f.load()
            assert isinstance(f.index, pd.core.frame.DataFrame)

    def __test_profile(self, bk, **ftc_opts):
        """ Test profile index fetching for a given backend """
        for arg in self.args["profile"]:
            options = {**self.fetcher_opts, **ftc_opts}
            f = ArgoIndexFetcher(src=bk, **options).profile(*arg)
            f.load()
            assert isinstance(f.index, pd.core.frame.DataFrame)

    def __test_region(self, bk, **ftc_opts):
        """ Test float index fetching for a given backend """
        for arg in self.args["region"]:
            options = {**self.fetcher_opts, **ftc_opts}
            f = ArgoIndexFetcher(src=bk, **options).region(arg)
            f.load()
            assert isinstance(f.index, pd.core.frame.DataFrame)

    @requires_localftp_index
    def test_float_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_float("localftp")

    @skip_for_debug
    @requires_connected_gdac
    def test_float_gdac(self):
        self.__test_float("gdac", N_RECORDS=100)

    @ci_erddap_index
    @requires_connected_erddap_index
    @safe_to_server_errors
    def test_float_erddap(self):
        with argopy.set_options(api_timeout=ERDDAP_TIMEOUT):
            self.__test_float("erddap")

    @requires_localftp_index
    def test_profile_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_profile("localftp")

    @skip_for_debug
    @requires_connected_gdac
    def test_profile_gdac(self):
        self.__test_profile("gdac", N_RECORDS=100)

    @requires_localftp_index
    def test_region_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_region("localftp")

    @ci_erddap_index
    @requires_connected_erddap_index
    def test_region_erddap(self):
        with argopy.set_options(api_timeout=ERDDAP_TIMEOUT):
            self.__test_region("erddap")

    @skip_for_debug
    @requires_connected_gdac
    def test_region_gdac(self):
        self.__test_region("gdac", N_RECORDS=100)
