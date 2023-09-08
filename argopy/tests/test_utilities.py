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
    check_gdac_path,
    isconnected,
    urlhaskeyword,
    isalive,
    isAPIconnected,
    erddap_ds_exists,
    linear_interpolation_remap,
    Chunker,
    format_oneline,
    wmo2box,
    modified_environ,
    wrap_longitude,
    toYearFraction, YearFraction_to_datetime,
    argo_split_path,
    Registry,
    float_wmo,
    get_coriolis_profile_id,
    get_ea_profile_page,
)
from argopy.utils import (
    is_box,
    is_list_of_strings,
)
from argopy.errors import InvalidFetcherAccessPoint, FtpPathError
from argopy import DataFetcher as ArgoDataFetcher
from utils import (
    requires_connection,
    requires_erddap,
    requires_gdac,
)
from mocked_http import mocked_httpserver, mocked_server_address


def test_invalid_dictionnary():
    with pytest.raises(ValueError):
        load_dict("invalid_dictionnary")


def test_invalid_dictionnary_key():
    d = load_dict("profilers")
    assert mapp_dict(d, "invalid_key") == "Unknown"


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())


def test_check_gdac_path():
    assert check_gdac_path("dummy_path", errors='ignore') is False
    with pytest.raises(FtpPathError):
        check_gdac_path("dummy_path", errors='raise')
    with pytest.warns(UserWarning):
        assert check_gdac_path("dummy_path", errors='warn') is False


@pytest.mark.parametrize("conda", [False, True],
                         indirect=False,
                         ids=["conda=%s" % str(p) for p in [False, True]])
def test_show_versions(conda):
    f = io.StringIO()
    argopy.show_versions(file=f, conda=conda)
    assert "SYSTEM" in f.getvalue()


def test_isconnected(mocked_httpserver):
    assert isinstance(isconnected(host=mocked_server_address), bool)
    assert isconnected(host="http://dummyhost") is False


def test_urlhaskeyword(mocked_httpserver):
    url = "https://api.ifremer.fr/argopy/data/ARGO-FULL.json"
    url.replace("https://api.ifremer.fr", mocked_server_address)
    assert isinstance(urlhaskeyword(url, "label"), bool)


params = [mocked_server_address,
          {"url": mocked_server_address + "/argopy/data/ARGO-FULL.json", "keyword": "label"}
          ]
params_ids = ["url is a %s" % str(type(p)) for p in params]
@pytest.mark.parametrize("params", params, indirect=False, ids=params_ids)
def test_isalive(params, mocked_httpserver):
    assert isinstance(isalive(params), bool)


@requires_erddap
@pytest.mark.parametrize("data", [True, False], indirect=False, ids=["data=%s" % t for t in [True, False]])
def test_isAPIconnected(data, mocked_httpserver):
    with argopy.set_options(erddap=mocked_server_address):
        assert isinstance(isAPIconnected(src="erddap", data=data), bool)


def test_erddap_ds_exists(mocked_httpserver):
    with argopy.set_options(erddap=mocked_server_address):
        assert isinstance(erddap_ds_exists(ds="ArgoFloats"), bool)
        assert erddap_ds_exists(ds="DummyDS") is False

# todo : Implement tests for utilities functions: badge, fetch_status and monitor_status


@requires_gdac
def test_clear_cache():
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    with tempfile.TemporaryDirectory() as cachedir:
        with argopy.set_options(cachedir=cachedir):
            loader = ArgoDataFetcher(src="gdac", ftp=ftproot, cache=True).profile(2902696, 12)
            loader.to_xarray()
            argopy.clear_cache()
            assert os.path.exists(cachedir) is True
            assert len(os.listdir(cachedir)) == 0


@requires_gdac
def test_lscache():
    ftproot, flist = argopy.tutorial.open_dataset("gdac")
    with tempfile.TemporaryDirectory() as cachedir:
        with argopy.set_options(cachedir=cachedir):
            loader = ArgoDataFetcher(src="gdac", ftp=ftproot, cache=True).profile(2902696, 12)
            loader.to_xarray()
            result = argopy.utilities.lscache(cache_path=cachedir, prt=True)
            assert isinstance(result, str)

            result = argopy.utilities.lscache(cache_path=cachedir, prt=False)
            assert isinstance(result, pd.DataFrame)



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


def test_format_oneline():
    s = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore"
    assert isinstance(format_oneline(s), str)
    assert isinstance(format_oneline(s[0:5]), str)
    s = format_oneline(s, max_width=12)
    assert isinstance(s, str) and len(s) == 12


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


class Test_argo_split_path:
    #############
    # UTILITIES #
    #############
    # src = "https://data-argo.ifremer.fr/dac"
    src = argopy.tutorial.open_dataset("gdac")[0] + "/dac"
    list_of_files = [
        src + "/bodc/6901929/6901929_prof.nc",  # core / multi-profile
        src + "/coriolis/3902131/3902131_Sprof.nc",  # bgc / synthetic multi-profile

        src + "/meds/4901079/profiles/D4901079_110.nc",  # core / mono-profile / Delayed
        src + "/aoml/13857/profiles/R13857_001.nc",  # core / mono-profile / Real

        src + "/coriolis/3902131/profiles/SD3902131_001.nc",  # bgc / synthetic mono-profile / Delayed
        src + "/coriolis/3902131/profiles/SD3902131_001D.nc",  # bgc / synthetic mono-profile / Delayed / Descent
        src + "/coriolis/6903247/profiles/SR6903247_134.nc",  # bgc / synthetic mono-profile / Real
        src + "/coriolis/6903247/profiles/SR6903247_134D.nc",  # bgc / synthetic mono-profile / Real / Descent

        src + "/coriolis/3902131/profiles/BR3902131_001.nc",  # bgc / mono-profile / Real
        src + "/coriolis/3902131/profiles/BR3902131_001D.nc",  # bgc / mono-profile / Real / Descent

        src + "/aoml/5900446/5900446_Dtraj.nc",  # traj / Delayed
        src + "/csio/2902696/2902696_Rtraj.nc",  # traj / Real

        src + "/coriolis/3902131/3902131_BRtraj.nc",  # bgc / traj / Real
        # src + "/coriolis/6903247/6903247_BRtraj.nc",  # bgc / traj / Real

        src + "/incois/2902269/2902269_tech.nc",  # technical
        # src + "/nmdis/2901623/2901623_tech.nc",  # technical

        src + "/jma/4902252/4902252_meta.nc",  # meta-data
        # src + "/coriolis/1900857/1900857_meta.nc",  # meta-data
    ]
    list_of_files = [f.replace("/", os.path.sep) for f in list_of_files]

    #########
    # TESTS #
    #########

    @pytest.mark.parametrize("file", list_of_files,
                             indirect=False)
    def test_argo_split_path(self, file):
        desc = argo_split_path(file)
        assert isinstance(desc, dict)
        for key in ['origin', 'path', 'name', 'type', 'extension', 'wmo', 'dac']:
            assert key in desc


class Test_float_wmo():

    def test_init(self):
        assert isinstance(float_wmo(2901746), float_wmo)
        assert isinstance(float_wmo(float_wmo(2901746)), float_wmo)

    def test_isvalid(self):
        assert float_wmo(2901746).isvalid
        assert not float_wmo(12, errors='ignore').isvalid

    def test_ppt(self):
        assert isinstance(str(float_wmo(2901746)), str)
        assert isinstance(repr(float_wmo(2901746)), str)

    def test_comparisons(self):
        assert float_wmo(2901746) == float_wmo(2901746)
        assert float_wmo(2901746) != float_wmo(2901745)
        assert float_wmo(2901746) >= float_wmo(2901746)
        assert float_wmo(2901746) > float_wmo(2901745)
        assert float_wmo(2901746) <= float_wmo(2901746)
        assert float_wmo(2901746) < float_wmo(2901747)

    def test_hashable(self):
        assert isinstance(hash(float_wmo(2901746)), int)


class Test_Registry():

    opts = [(None, 'str'), (['hello', 'world'], str), (None, float_wmo), ([2901746, 4902252], float_wmo)]
    opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_init(self, opts):
        assert isinstance(Registry(opts[0], dtype=opts[1]), Registry)

    opts = [(['hello', 'world'], str), ([2901746, 4902252], float_wmo)]
    opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_commit(self, opts):
        R = Registry(dtype=opts[1])
        R.commit(opts[0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_append(self, opts):
        R = Registry(dtype=opts[1])
        R.append(opts[0][0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_extend(self, opts):
        R = Registry(dtype=opts[1])
        R.append(opts[0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_insert(self, opts):
        R = Registry(opts[0][0], dtype=opts[1])
        R.insert(0, opts[0][-1])
        assert R[0] == opts[0][-1]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_remove(self, opts):
        R = Registry(opts[0], dtype=opts[1])
        R.remove(opts[0][0])
        assert opts[0][0] not in R

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_copy(self, opts):
        R = Registry(opts[0], dtype=opts[1])
        assert R == R.copy()

    bad_opts = [(['hello', 12], str), ([2901746, 1], float_wmo)]
    bad_opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", bad_opts, indirect=False, ids=bad_opts_ids)
    def test_invalid_dtype(self, opts):
        with pytest.raises(ValueError):
            Registry(opts[0][0], dtype=opts[1], invalid='raise').commit(opts[0][-1])
        with pytest.warns(UserWarning):
            Registry(opts[0][0], dtype=opts[1], invalid='warn').commit(opts[0][-1])
        # Raise nothing:
        Registry(opts[0][0], dtype=opts[1], invalid='ignore').commit(opts[0][-1])


@pytest.mark.parametrize("params", [[6901929, None], [6901929, 12]], indirect=False, ids=['float', 'profile'])
def test_get_coriolis_profile_id(params, mocked_httpserver):
    with argopy.set_options(cachedir=tempfile.mkdtemp()):
        assert isinstance(get_coriolis_profile_id(params[0], params[1], api_server=mocked_server_address), pd.core.frame.DataFrame)

@pytest.mark.parametrize("params", [[6901929, None], [6901929, 12]], indirect=False, ids=['float', 'profile'])
def test_get_ea_profile_page(params, mocked_httpserver):
    with argopy.set_options(cachedir=tempfile.mkdtemp()):
        assert is_list_of_strings(get_ea_profile_page(params[0], params[1], api_server=mocked_server_address))
