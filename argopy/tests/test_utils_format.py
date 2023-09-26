import os
import pytest
import argopy
from argopy.utils.format import format_oneline, argo_split_path


def test_format_oneline():
    s = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore"
    assert isinstance(format_oneline(s), str)
    assert isinstance(format_oneline(s[0:5]), str)
    s = format_oneline(s, max_width=12)
    assert isinstance(s, str) and len(s) == 12


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
