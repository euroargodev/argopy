import pytest
import importlib

import argopy
from argopy import IndexFetcher as ArgoIndexFetcher
from argopy.errors import InvalidFetcherAccessPoint, OptionValueError
from utils import (
    # AVAILABLE_INDEX_SOURCES,
    requires_fetcher_index,
    requires_connected_erddap_index,
    requires_connected_gdac,
    requires_connection,
    requires_ipython,
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


skip_for_debug = pytest.mark.skipif(True, reason="Taking too long !")


@requires_connection
class Test_Facade:
    local_ftp = argopy.tutorial.open_dataset("gdac")[0]
    src = 'gdac'
    src_opts = {'ftp': local_ftp}

    def __get_fetcher(self, empty: bool = False, pt: str = 'profile'):
        f = ArgoIndexFetcher(src=self.src, **self.src_opts)

        if pt == 'float':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).float(2901623)
            else:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).float(12)

        if pt == 'profile':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).profile(2901623, 12)
            else:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).profile(12, 1200)

        if pt == 'region':
            if not empty:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).region([-60, -55, 40.0, 45.0, 0.0, 10.0,
                                                                "2007-08-01", "2007-09-01"])
            else:
                return f, ArgoIndexFetcher(src=self.src, **self.src_opts).region([-60, -55, 40.0, 45.0, 99.92, 99.99,
                                                                "2007-08-01", "2007-08-01"])

    def test_invalid_fetcher(self):
        with pytest.raises(OptionValueError):
            ArgoIndexFetcher(src="invalid_fetcher").to_xarray()

    @requires_fetcher_index
    def test_invalid_accesspoint(self):
        # Use the first valid data source
        with pytest.raises(InvalidFetcherAccessPoint):
            ArgoIndexFetcher(
                src=self.src, **self.src_opts
            ).invalid_accesspoint.to_xarray()  # Can't get data if access point not defined first
        with pytest.raises(InvalidFetcherAccessPoint):
            ArgoIndexFetcher(
                src=self.src, **self.src_opts
            ).to_xarray()  # Can't get data if access point not defined first

    @requires_fetcher_index
    def test_invalid_dataset(self):
        with pytest.raises(ValueError):
            ArgoIndexFetcher(src=self.src, ds='dummy_ds', **self.src_opts)

    @requires_matplotlib
    def test_plot(self):
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
        f, fetcher = self.__get_fetcher(pt='float')
        dsh = fetcher.plot(ptype='qc_altimetry', embed='slide')
        assert isinstance(dsh(0), IPython.display.Image)
