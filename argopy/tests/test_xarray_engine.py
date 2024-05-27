import os
import pytest
import xarray as xr
import logging
import warnings
import argopy
from argopy.utils.format import argo_split_path

log = logging.getLogger("argopy.tests.xarray.engine")


def print_desc(desc):
    txt = [desc["type"]]
    if "direction" in desc:
        txt.append(desc["direction"])

    if "data_mode" in desc:
        txt.append(desc["data_mode"])

    return ", ".join(txt)


class Test_Argo_Engine:
    host = argopy.tutorial.open_dataset("gdac")[0]
    src = host + "/dac"
    list_of_files = [
        src + "/bodc/6901929/6901929_prof.nc",  # core / multi-profile
        src + "/coriolis/3902131/3902131_Sprof.nc",  # bgc / synthetic multi-profile
        src + "/meds/4901079/profiles/D4901079_110.nc",  # core / mono-profile / Delayed
        src + "/aoml/13857/profiles/R13857_001.nc",  # core / mono-profile / Real
        src
        + "/coriolis/3902131/profiles/SD3902131_001.nc",  # bgc / synthetic mono-profile / Delayed
        src
        + "/coriolis/3902131/profiles/SD3902131_001D.nc",  # bgc / synthetic mono-profile / Delayed / Descent
        src
        + "/coriolis/6903247/profiles/SR6903247_134.nc",  # bgc / synthetic mono-profile / Real
        src
        + "/coriolis/6903247/profiles/SR6903247_134D.nc",  # bgc / synthetic mono-profile / Real / Descent
        src
        + "/coriolis/3902131/profiles/BR3902131_001.nc",  # bgc / mono-profile / Real
        src
        + "/coriolis/3902131/profiles/BR3902131_001D.nc",  # bgc / mono-profile / Real / Descent
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

    list_of_files_desc = [print_desc(argo_split_path(f)) for f in list_of_files]
    # list_of_files_desc = [f for f in list_of_files]

    #############
    # UTILITIES #
    #############

    def how_many_casted(self, this_ds):
        """Return the length of non casted variables in ds"""
        # check which variables are not casted:
        l = []
        for iv, v in enumerate(this_ds.variables):
            if this_ds[v].dtype == "O":
                l.append(v)
        # log.debug("%i/%i variables not casted properly.\n%s" % (len(l), len(this_ds.variables), l))
        return len(l), len(this_ds.variables)

    #########
    # TESTS #
    #########

    @pytest.mark.parametrize(
        "file", list_of_files, indirect=False, ids=list_of_files_desc
    )
    def test_read(self, file, mocked_httpserver):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "invalid value encountered in cast", RuntimeWarning
            )

            # No Argo enfine:
            # ds1 = xr.open_dataset(file)
            # n1, N1, l1 = self.how_many_casted(ds1)

            # With Argo engine:
            ds = xr.open_dataset(file, engine="argo")
            n, N = self.how_many_casted(ds)
            assert n == 0
