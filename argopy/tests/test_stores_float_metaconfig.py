import pytest
import logging
import numpy as np
import pandas as pd
import tempfile

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
from mocked_http import mocked_httpserver, mocked_server_address

log = logging.getLogger("argopy.tests.floatstore.config")
argopy.clear_cache()

skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")
skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")

"""
List WMO to be tested, one for each mission
"""
VALID_WMO = [13857, 3902131]


class FloatStore_Config_Proto:

    def setup_class(self):
        self.cachedir = tempfile.mkdtemp()

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_missions(self, wmo, config, mocked_httpserver):
        # log.debug(type(config._metadata))
        assert isinstance(config.n_missions, int)
        assert isinstance(config.missions, list)
        assert all(isinstance(item, int) for item in config.missions)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_params(self, wmo, config, mocked_httpserver):
        assert isinstance(config.n_params, int)
        assert config.n_params == len(config)
        assert all(isinstance(item, str) for item in config.parameters)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_cycles(self, wmo, config, mocked_httpserver):
        assert isinstance(config.cycles, dict)
        assert all(isinstance(key, int) for key in config.cycles.keys())
        assert all(isinstance(val, int) for val in config.cycles.values())

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_get(self, wmo, config, mocked_httpserver):
        a_param = np.random.choice(config.parameters, 1)[0]
        assert isinstance(config[a_param], list)
        assert isinstance(config[a_param, 1:2], list)
        assert isinstance(config[a_param, 1], int | float | str | bool)

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_for_cycles(self, wmo, config, mocked_httpserver):
        a_param = np.random.choice(config.parameters, 1)[0]
        cyc = config.cycles[np.random.choice(config.missions, 1)[0]]
        assert isinstance(
            config.for_cycles(a_param, cycle_numbers=cyc), int | float | str | bool
        )

    @pytest.mark.parametrize(
        "wmo", VALID_WMO, indirect=False, ids=[f"wmo={w}" for w in VALID_WMO]
    )
    def test_to_dataframe(self, wmo, config, mocked_httpserver):
        n = (config.n_params, config.n_missions)

        for missions in [
            None,
            np.random.choice(config.missions, 1)[0],
            np.random.choice(config.missions, 2),
        ]:
            nm = n[1] if missions is None else (missions := np.unique(missions)).size
            df = config.to_dataframe(missions=missions)
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (n[0], 3 + nm)


@skip_offline
@requires_gdac
class Test_FloatStore_Config_Offline(FloatStore_Config_Proto):
    af: dict[int, ArgoFloat] = {}

    # Define a fixture for an offline ArgoFloat instance (local store)
    @pytest.fixture
    def argo_float(self, wmo):
        if wmo not in self.af:
            self.af[wmo] = ArgoFloat(
                wmo,
                host=argopy.tutorial.open_dataset("gdac")[0],
                cache=True,
                cachedir=self.cachedir,
            )
        return self.af[wmo]

    # Define a fixture for the config extension
    @pytest.fixture
    def config(self, argo_float):
        return config_off(argo_float)


@skip_online
class Test_FloatStore_Config_Online(FloatStore_Config_Proto):
    af: dict[int, ArgoFloat] = {}

    # Define a fixture for an online ArgoFloat instance (but using mocked http)
    @pytest.fixture
    def argo_float(self, wmo, mocked_httpserver):
        if wmo not in self.af:
            self.af[wmo] = ArgoFloat(
                wmo,
                host=mocked_server_address,
                cache=True,
                cachedir=self.cachedir,
                eafleetmonitoring_server=mocked_server_address,
            )
        return self.af[wmo]

    # Define a fixture for the config extension
    @pytest.fixture
    def config(self, argo_float):
        return config_on(argo_float)
