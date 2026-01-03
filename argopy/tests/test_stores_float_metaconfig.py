import pytest
import logging
import numpy as np
import pandas as pd

import argopy
from argopy.stores import ArgoFloat
from argopy.stores.float.implementations.online.meta_config import (
    ConfigParameters as config_on,
)
from argopy.stores.float.implementations.offline.meta_config import (
    ConfigParameters as config_off,
)

from utils import (
    requires_gdac,
)

log = logging.getLogger("argopy.tests.floatstore.config")
argopy.clear_cache()

skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")
skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")

"""
Select a GDAC host to be use for config extension tests
"""
VALID_HOST = argopy.tutorial.open_dataset("gdac")[0]  # Use local files
# 'http1': mocked_server_address,  # Use the mocked http server
# 'http2': 'https://data-argo.ifremer.fr',
# 'ftp': "MOCKFTP",  # keyword to use a fake/mocked ftp server (running on localhost)

"""
List WMO to be tested, one for each mission
"""
VALID_WMO = [13857, 3902131]

@skip_offline
@requires_gdac
class Test_FloatStore_Config_Offline:
    config = config_off

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_missions(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        log.debug(type(cp._metadata))

        assert isinstance(cp.n_missions, int)
        assert isinstance(cp.missions, list)
        assert all(isinstance(item, int) for item in cp.missions)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_params(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)

        assert isinstance(cp.n_params, int)
        assert cp.n_params == len(cp)
        assert all(isinstance(item, str) for item in cp.parameters)

        assert isinstance(cp.cycles, dict)
        assert all(isinstance(key, int) for key in cp.cycles.keys())
        assert all(isinstance(val, int) for val in cp.cycles.values())

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_cycles(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)

        assert isinstance(cp.cycles, dict)
        assert all(isinstance(key, int) for key in cp.cycles.keys())
        assert all(isinstance(val, int) for val in cp.cycles.values())

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_get(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        a_param = np.random.choice(cp.parameters, 1)[0]
        assert isinstance(cp[a_param], list)
        assert isinstance(cp[a_param, 1:2], list)
        assert isinstance(cp[a_param, 1], int | float | str | bool)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_for_cycles(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        a_param = np.random.choice(cp.parameters, 1)[0]
        cyc = cp.cycles[np.random.choice(cp.missions, 1)[0]]
        assert isinstance(
            cp.for_cycles(a_param, cycle_numbers=cyc), int | float | str | bool
        )

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_to_dataframe(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        n = (cp.n_params, cp.n_missions)

        for missions in [
            None,
            np.random.choice(cp.missions, 1)[0],
            np.random.choice(cp.missions, 2),
        ]:
            nm = n[1] if missions is None else (missions := np.unique(missions)).size
            df = cp.to_dataframe(missions=missions)
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (n[0], 3 + nm)

@skip_online
@requires_gdac
class Test_FloatStore_Config_Online:
    config = config_on

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_missions(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        log.debug(type(cp._metadata))

        assert isinstance(cp.n_missions, int)
        assert isinstance(cp.missions, list)
        assert all(isinstance(item, int) for item in cp.missions)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_params(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)

        assert isinstance(cp.n_params, int)
        assert cp.n_params == len(cp)
        assert all(isinstance(item, str) for item in cp.parameters)

        assert isinstance(cp.cycles, dict)
        assert all(isinstance(key, int) for key in cp.cycles.keys())
        assert all(isinstance(val, int) for val in cp.cycles.values())

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_cycles(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)

        assert isinstance(cp.cycles, dict)
        assert all(isinstance(key, int) for key in cp.cycles.keys())
        assert all(isinstance(val, int) for val in cp.cycles.values())

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_get(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        a_param = np.random.choice(cp.parameters, 1)[0]
        assert isinstance(cp[a_param], list)
        assert isinstance(cp[a_param, 1:2], list)
        assert isinstance(cp[a_param, 1], int | float | str | bool)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_for_cycles(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        a_param = np.random.choice(cp.parameters, 1)[0]
        cyc = cp.cycles[np.random.choice(cp.missions, 1)[0]]
        assert isinstance(
            cp.for_cycles(a_param, cycle_numbers=cyc), int | float | str | bool
        )

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_to_dataframe(self, wmo):
        af = ArgoFloat(wmo, host=VALID_HOST, cache=True)
        cp = self.config(af)
        n = (cp.n_params, cp.n_missions)

        for missions in [
            None,
            np.random.choice(cp.missions, 1)[0],
            np.random.choice(cp.missions, 2),
        ]:
            nm = n[1] if missions is None else (missions := np.unique(missions)).size
            df = cp.to_dataframe(missions=missions)
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (n[0], 3 + nm)
