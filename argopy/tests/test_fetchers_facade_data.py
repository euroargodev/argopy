import numpy as np
import pandas as pd
import xarray as xr

import pytest

import argopy
from argopy import DataFetcher as ArgoDataFetcher
from argopy.errors import (
    InvalidFetcherAccessPoint,
    InvalidFetcher
)
from argopy.utilities import is_list_of_strings
from utils import (
    requires_fetcher,
    requires_connected_erddap_phy,
    requires_localftp,
    # requires_gdac,
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

skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


@requires_localftp
class Test_Facade:

    # Use the first valid data source:
    # src = list(AVAILABLE_SOURCES.keys())[0]
    local_ftp = argopy.tutorial.open_dataset("localftp")[0]
    src = 'localftp'

    def __get_fetcher(self, empty: bool = False, pt: str = 'profile'):
        f = ArgoDataFetcher(src=self.src)
        # f.valid_access_points[0]

        if pt == 'float':
            if not empty:
                return f, ArgoDataFetcher(src=self.src).float(2901623)
            else:
                return f, ArgoDataFetcher(src=self.src).float(12)

        if pt == 'profile':
            if not empty:
                return f, ArgoDataFetcher(src=self.src).profile(2901623, 12)
            else:
                return f, ArgoDataFetcher(src=self.src).profile(12, 1200)

        if pt == 'region':
            if not empty:
                return f, ArgoDataFetcher(src=self.src).region([-60, -55, 40.0, 45.0, 0.0, 10.0,
                                                                "2007-08-01", "2007-09-01"])
            else:
                return f, ArgoDataFetcher(src=self.src).region([-60, -55, 40.0, 45.0, 99.92, 99.99,
                                                                "2007-08-01", "2007-08-01"])

    def test_invalid_fetcher(self):
        with pytest.raises(InvalidFetcher):
            ArgoDataFetcher(src="invalid_fetcher").to_xarray()

    def test_invalid_accesspoint(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(InvalidFetcherAccessPoint):
                self.__get_fetcher()[0].invalid_accesspoint.to_xarray()

    def test_invalid_dataset(self):
        with pytest.raises(ValueError):
            ArgoDataFetcher(src=self.src, ds='dummy_ds')

    def test_warnings(self):
        with pytest.warns(UserWarning):
            ArgoDataFetcher(src='erddap', ds='bgc', mode='standard')

    def test_no_uri(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            with pytest.raises(InvalidFetcherAccessPoint):
                self.__get_fetcher()[0].uri

    def test_to_xarray(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            assert isinstance(self.__get_fetcher()[1].to_xarray(), xr.Dataset)
            with pytest.raises(InvalidFetcher):
                assert self.__get_fetcher()[0].to_xarray()

    def test_to_dataframe(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            assert isinstance(self.__get_fetcher()[1].to_dataframe(), pd.core.frame.DataFrame)
            with pytest.raises(InvalidFetcher):
                assert self.__get_fetcher()[0].to_dataframe()

    def test_to_index(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            assert isinstance(self.__get_fetcher()[1].to_index(), pd.core.frame.DataFrame)
            assert isinstance(self.__get_fetcher()[1].to_index(full=True), pd.core.frame.DataFrame)
            assert isinstance(self.__get_fetcher()[1].to_index(full=False, coriolis_id=True), pd.core.frame.DataFrame)

    def test_load(self):
        with argopy.set_options(local_ftp=self.local_ftp):
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

            with pytest.raises(ValueError):
                fetcher.plot(ptype='invalid_cat', with_seaborn=ws)

    @requires_matplotlib
    def test_plot_qc_altimetry(self):
        if has_ipython:
            import IPython

        with argopy.set_options(local_ftp=self.local_ftp):
            f, fetcher = self.__get_fetcher(pt='float')
            dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
            if has_ipython:
                assert isinstance(dsh(0), IPython.display.Image)
            else:
                assert isinstance(dsh, dict)

    # @requires_ipython
    # @requires_matplotlib
    # def test_plot_qc_altimetry(self):
    #     import IPython
    #     with argopy.set_options(local_ftp=self.local_ftp):
    #         f, fetcher = self.__get_fetcher(pt='float')
    #
    #         # Test 'qc_altimetry'
    #         dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
    #         assert isinstance(dsh(0), IPython.display.Image)


"""
The following tests are not necessary, since data fetching is tested from each data fetcher tests 
"""
@requires_fetcher
class Test_DataFetching:
    """ Test main API facade for all available fetching backends and default dataset """

    local_ftp = argopy.tutorial.open_dataset("localftp")[0]

    # todo Determine the list of output format to test
    # what else beyond .to_xarray() ?

    fetcher_opts = {}

    mode = ["standard", "expert"]
    # mode = ["standard"]

    # Define API entry point options to tests:
    # args = {}
    # args["float"] = [[2901623], [2901623, 6901929]]
    # args["profile"] = [[2901623, 12], [2901623, np.arange(12, 14)], [6901929, [1, 6]]]
    # args["region"] = [
    #     [12.181, 13.459, -40.416, -39.444, 0.0, 1014.0],
    #     [12.181, 17.459, -40.416, -34.444, 0.0, 2014.0, '2008-06-07', '2008-09-06'],
    # ]
    args = {}
    args["float"] = [[4901079]]
    args["profile"] = [[4901079, 135], [4901079, [138, 139]]]
    args["region"] = [
        [-50., -45., 36., 37., 0., 100.],
        [-50., -45., 36., 37., 0., 100., "2008-01-01", "2008-02-15"],
    ]


    def test_profile_from_float(self):
        with pytest.raises(TypeError):
            ArgoDataFetcher(src='erddap').float(self.args["float"][0], CYC=12)

    def __assert_fetcher(self, f):
        # Standard loading of measurements:
        f.load()
        assert is_list_of_strings(f.uri)
        assert isinstance(f.data, xr.Dataset)
        assert isinstance(f.index, pd.core.frame.DataFrame)

        # Only test specific output structures:
        # f.to_xarray()
        # f.to_dataframe()
        # f.to_index(full=False)

    def __test_float(self, bk, **ftc_opts):
        """ Test float for a given backend """
        for arg in self.args["float"]:
            for mode in self.mode:
                options = {**self.fetcher_opts, **ftc_opts}
                f = ArgoDataFetcher(src=bk, mode=mode, **options).float(arg)
                self.__assert_fetcher(f)

    def __test_profile(self, bk, **ftc_opts):
        """ Test profile for a given backend """
        for arg in self.args["profile"]:
            for mode in self.mode:
                options = {**self.fetcher_opts, **ftc_opts}
                f = ArgoDataFetcher(src=bk, mode=mode, **options).profile(*arg)
                self.__assert_fetcher(f)

    def __test_region(self, bk, **ftc_opts):
        """ Test region for a given backend """
        for arg in self.args["region"]:
            for mode in self.mode:
                options = {**self.fetcher_opts, **ftc_opts}
                f = ArgoDataFetcher(src=bk, mode=mode, **options).region(arg)
                self.__assert_fetcher(f)

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_float_erddap(self):
        self.__test_float("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_float_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_float("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_float_argovis(self):
        self.__test_float("argovis")

    # @requires_connected_gdac
    @safe_to_server_errors
    def test_float_gdac(self):
        # self.__test_float("gdac", N_RECORDS=100)
        with argopy.set_options(ftp=self.local_ftp):
            self.__test_float("gdac")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_profile_erddap(self):
        self.__test_profile("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_profile_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_profile("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_profile_argovis(self):
        self.__test_profile("argovis")

    # @requires_connected_gdac
    @safe_to_server_errors
    def test_profile_gdac(self):
        with argopy.set_options(ftp=self.local_ftp):
            self.__test_profile("gdac")

    @requires_connected_erddap_phy
    @safe_to_server_errors
    def test_region_erddap(self):
        self.__test_region("erddap")

    @requires_localftp
    @safe_to_server_errors
    def test_region_localftp(self):
        with argopy.set_options(local_ftp=self.local_ftp):
            self.__test_region("localftp")

    @requires_connected_argovis
    @safe_to_server_errors
    def test_region_argovis(self):
        self.__test_region("argovis")

    # @requires_connected_gdac
    @safe_to_server_errors
    def test_region_gdac(self):
        # self.__test_region("gdac", N_RECORDS=100)
        with argopy.set_options(ftp=self.local_ftp):
            self.__test_region("gdac")
