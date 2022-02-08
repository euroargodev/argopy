import os
import io
import pytest
import tempfile
import xarray as xr
import pandas as pd
import numpy as np
import types

import argopy
from argopy.utilities import (
    load_dict,
    mapp_dict,
    list_multiprofile_file_variables,
    check_localftp,
    isconnected,
    isAPIconnected,
    erddap_ds_exists,
    linear_interpolation_remap,
    Chunker,
    is_box,
    is_list_of_strings,
    format_oneline, is_indexbox,
    check_wmo, is_wmo,
    wmo2box,
    modified_environ,
    wrap_longitude,
    toYearFraction, YearFraction_to_datetime,
    TopoFetcher
)
from argopy.errors import InvalidFetcherAccessPoint, FtpPathError
from argopy import DataFetcher as ArgoDataFetcher
from . import requires_connection, requires_localftp, safe_to_server_errors


def test_invalid_dictionnary():
    with pytest.raises(ValueError):
        load_dict("invalid_dictionnary")


def test_invalid_dictionnary_key():
    d = load_dict("profilers")
    assert mapp_dict(d, "invalid_key") == "Unknown"


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())


def test_check_localftp():
    assert check_localftp("dummy_path", errors='ignore') is False
    with pytest.raises(FtpPathError):
        check_localftp("dummy_path", errors='raise')
    with pytest.warns(UserWarning):
        assert check_localftp("dummy_path", errors='warn') is False


def test_show_versions():
    f = io.StringIO()
    argopy.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()


def test_isconnected():
    assert isinstance(isconnected(), bool)
    assert isconnected(host="http://dummyhost") is False


def test_isAPIconnected():
    assert isinstance(isAPIconnected(src="erddap", data=True), bool)
    assert isinstance(isAPIconnected(src="erddap", data=False), bool)


def test_erddap_ds_exists():
    assert isinstance(erddap_ds_exists(ds="ArgoFloats"), bool)
    assert erddap_ds_exists(ds="DummyDS") is False


# todo : Implement tests for utilities functions: badge, fetch_status and monitor_status


@requires_connection
@requires_localftp
def test_clear_cache():
    ftproot, flist = argopy.tutorial.open_dataset("localftp")
    with tempfile.TemporaryDirectory() as cachedir:
        with argopy.set_options(cachedir=cachedir, local_ftp=ftproot):
            loader = ArgoDataFetcher(src="localftp", cache=True).profile(2902696, 12)
            loader.to_xarray()  # 1st call to load from source and save in memory
            loader.to_xarray()  # 2nd call to load from memory and save in cache
            argopy.clear_cache()
            assert os.path.exists(cachedir) is True
            assert len(os.listdir(cachedir)) == 0


# We disable this test because the server has not responded over a week (May 29th)
# @unittest.skipUnless(CONNECTED, "open_etopo1 requires an internet connection")
# def test_open_etopo1():
#     try:
#         ds = open_etopo1([-80, -79, 20, 21], res='l')
#         assert isinstance(ds, xr.DataArray) is True
#     except requests.HTTPError:  # not our fault
#         pass


class Test_linear_interpolation_remap:
    @pytest.fixture(autouse=True)
    def create_data(self):
        # create fake data to test interpolation:
        temp = np.random.rand(200, 100)
        pres = np.sort(
            np.floor(
                np.zeros([200, 100])
                + np.linspace(50, 950, 100)
                + np.random.randint(-5, 5, [200, 100])
            )
        )
        self.dsfake = xr.Dataset(
            {
                "TEMP": (["N_PROF", "N_LEVELS"], temp),
                "PRES": (["N_PROF", "N_LEVELS"], pres),
            },
            coords={
                "N_PROF": ("N_PROF", range(200)),
                "N_LEVELS": ("N_LEVELS", range(100)),
                "Z_LEVELS": ("Z_LEVELS", np.arange(100, 900, 20)),
            },
        )

    def test_interpolation(self):
        # Run it with success:
        dsi = linear_interpolation_remap(
            self.dsfake["PRES"],
            self.dsfake["TEMP"],
            self.dsfake["Z_LEVELS"],
            z_dim="N_LEVELS",
            z_regridded_dim="Z_LEVELS",
        )
        assert "remapped" in dsi.dims

    def test_interpolation_1d(self):
        # Run it with success:
        dsi = linear_interpolation_remap(
            self.dsfake["PRES"].isel(N_PROF=0),
            self.dsfake["TEMP"].isel(N_PROF=0),
            self.dsfake["Z_LEVELS"],
            z_regridded_dim="Z_LEVELS",
        )
        assert "remapped" in dsi.dims

    def test_error_zdim(self):
        # Test error:
        # catches error from _regular_interp linked to z_dim
        with pytest.raises(RuntimeError):
            linear_interpolation_remap(
                self.dsfake["PRES"],
                self.dsfake["TEMP"],
                self.dsfake["Z_LEVELS"],
                z_regridded_dim="Z_LEVELS",
            )

    def test_error_ds(self):
        # Test error:
        # catches error from linear_interpolation_remap linked to datatype
        with pytest.raises(ValueError):
            linear_interpolation_remap(
                self.dsfake["PRES"],
                self.dsfake,
                self.dsfake["Z_LEVELS"],
                z_dim="N_LEVELS",
                z_regridded_dim="Z_LEVELS",
            )


class Test_Chunker:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.WMO = [
            6902766,
            6902772,
            6902914,
            6902746,
            6902916,
            6902915,
            6902757,
            6902771,
        ]
        self.BOX3d = [0, 20, 40, 60, 0, 1000]
        self.BOX4d = [0, 20, 40, 60, 0, 1000, "2001-01", "2001-6"]

    def test_InvalidFetcherAccessPoint(self):
        with pytest.raises(InvalidFetcherAccessPoint):
            Chunker({"invalid": self.WMO})

    def test_invalid_chunks(self):
        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunks='toto')

    def test_invalid_chunksize(self):
        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunksize='toto')

    def test_chunk_wmo(self):
        C = Chunker({"wmo": self.WMO})
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )

        C = Chunker({"wmo": self.WMO}, chunks="auto")
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )

        C = Chunker({"wmo": self.WMO}, chunks={"wmo": 1})
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )
        assert len(C.fit_transform()) == 1

        with pytest.raises(ValueError):
            Chunker({"wmo": self.WMO}, chunks=["wmo", 1])

        C = Chunker({"wmo": self.WMO})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )

    def test_chunk_box3d(self):
        C = Chunker({"box": self.BOX3d})
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX3d}, chunks="auto")
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 12, "lat": 1, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lat": 1, "dpt": 1}, chunksize={"lon": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][1] - chunks[0][0] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 1, "lat": 12, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lon": 1, "dpt": 1}, chunksize={"lat": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][3] - chunks[0][2] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 1, "lat": 1, "dpt": 12})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lon": 1, "lat": 1}, chunksize={"dpt": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][5] - chunks[0][4] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 4, "lat": 2, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2 * 4

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 2, "lat": 3, "dpt": 4})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2 * 3 * 4

        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunks=["lon", 1])

        C = Chunker({"box": self.BOX3d})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )

    def test_chunk_box4d(self):
        C = Chunker({"box": self.BOX4d})
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX4d}, chunks="auto")
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 2, "lat": 1, "dpt": 1, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lat": 1, "dpt": 1, "time": 1},
            chunksize={"lon": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][1] - chunks[0][0] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 2, "dpt": 1, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "dpt": 1, "time": 1},
            chunksize={"lat": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][3] - chunks[0][2] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 1, "dpt": 2, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "lat": 1, "time": 1},
            chunksize={"dpt": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][5] - chunks[0][4] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 1, "dpt": 1, "time": 2}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "lat": 1, "dpt": 1},
            chunksize={"time": 5},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert np.timedelta64(
            pd.to_datetime(chunks[0][7]) - pd.to_datetime(chunks[0][6]), "D"
        ) <= np.timedelta64(5, "D")

        with pytest.raises(ValueError):
            Chunker({"box": self.BOX4d}, chunks=["lon", 1])

        C = Chunker({"box": self.BOX4d})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )


class Test_is_box:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.BOX3d = [0, 20, 40, 60, 0, 1000]
        self.BOX4d = [0, 20, 40, 60, 0, 1000, "2001-01", "2001-6"]

    def test_box_ok(self):
        assert is_box(self.BOX3d)
        assert is_box(self.BOX4d)

    def test_box_notok(self):
        for box in [[], list(range(0, 12))]:
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_num(self):
        for i in [0, 1, 2, 3, 4, 5]:
            box = self.BOX3d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_range(self):
        for i in [0, 1, 2, 3, 4, 5]:
            box = self.BOX3d
            box[i] = -1000
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_str(self):
        for i in [6, 7]:
            box = self.BOX4d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")


class Test_is_indexbox:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.BOX2d = [0, 20, 40, 60]
        self.BOX3d = [0, 20, 40, 60, "2001-01", "2001-6"]

    def test_box_ok(self):
        assert is_indexbox(self.BOX2d)
        assert is_indexbox(self.BOX3d)

    def test_box_notok(self):
        for box in [[], list(range(0, 12))]:
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_num(self):
        for i in [0, 1, 2, 3]:
            box = self.BOX2d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_range(self):
        for i in [0, 1, 2, 3]:
            box = self.BOX2d
            box[i] = -1000
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_str(self):
        for i in [4, 5]:
            box = self.BOX3d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")


def test_format_oneline():
    s = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore"
    assert isinstance(format_oneline(s), str)
    assert isinstance(format_oneline(s[0:5]), str)
    s = format_oneline(s, max_width=12)
    assert isinstance(s, str) and len(s) == 12


def test_is_wmo():
    assert is_wmo(12345)
    assert is_wmo([12345])
    assert is_wmo([12345, 1234567])
    with pytest.raises(ValueError):
        is_wmo(1234, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(-1234, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(1234.12, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(12345.7, errors="raise")
    assert not is_wmo(12, errors="silent")
    assert not is_wmo(-12, errors="silent")
    assert not is_wmo(1234.12, errors="silent")
    assert not is_wmo(12345.7, errors="silent")


def test_check_wmo():
    assert check_wmo(12345) == [12345]
    assert check_wmo([1234567]) == [1234567]
    assert check_wmo([12345, 1234567]) == [12345, 1234567]
    assert check_wmo(np.array((12345, 1234567), dtype='int')) == [12345, 1234567]


def test_modified_environ():
    os.environ["DUMMY_ENV_ARGOPY"] = 'initial'
    with modified_environ(DUMMY_ENV_ARGOPY='toto'):
        assert os.environ['DUMMY_ENV_ARGOPY'] == 'toto'
    assert os.environ['DUMMY_ENV_ARGOPY'] == 'initial'
    os.environ.pop('DUMMY_ENV_ARGOPY')


def test_wmo2box():
    with pytest.raises(ValueError):
        wmo2box(12)
    with pytest.raises(ValueError):
        wmo2box(8000)
    with pytest.raises(ValueError):
        wmo2box(2000)

    def complete_box(b):
        b2 = b.copy()
        b2.insert(4, 0.)
        b2.insert(5, 10000.)
        return b2

    assert is_box(complete_box(wmo2box(1212)))
    assert is_box(complete_box(wmo2box(3324)))
    assert is_box(complete_box(wmo2box(5402)))
    assert is_box(complete_box(wmo2box(7501)))


def test_wrap_longitude():
    assert wrap_longitude(np.array([-20])) == 340
    assert wrap_longitude(np.array([40])) == 40
    assert np.all(np.equal(wrap_longitude(np.array([340, 20])), np.array([340, 380])))


def test_toYearFraction():
    assert toYearFraction(pd.to_datetime('202001010000')) == 2020
    assert toYearFraction(pd.to_datetime('202001010000', utc=True)) == 2020
    assert toYearFraction(pd.to_datetime('202001010000')+pd.offsets.DateOffset(years=1)) == 2021


def test_YearFraction_to_datetime():
    assert YearFraction_to_datetime(2020) == pd.to_datetime('202001010000')
    assert YearFraction_to_datetime(2020+1) == pd.to_datetime('202101010000')


class Test_TopoFetcher():

    @safe_to_server_errors
    def test_fetching(self):
        box = [81, 123, -67, -54]
        fetcher = TopoFetcher(box, ds='gebco', stride=[10, 10], cache=True)
        ds = fetcher.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert 'elevation' in ds.data_vars

    @safe_to_server_errors
    def test_fetching_cached(self):
        box = [81, 123, -67, -54]
        fetcher = TopoFetcher(box, ds='gebco', stride=[10, 10], cache=False)
        ds = fetcher.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert 'elevation' in ds.data_vars
