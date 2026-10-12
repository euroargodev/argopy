import pytest
import logging

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver


log = logging.getLogger("argopy.tests.extensions.canyonmed")
USE_MOCKED_SERVER = True


@pytest.fixture
def fetcher():
    defaults_args = {"src": 'erddap',
                     "cache": False,
                     "ds": 'bgc',
                     "params": "DOXY",
                     "measured": "DOXY",
                     }
    if USE_MOCKED_SERVER:
        defaults_args['server'] = mocked_server_address

    return DataFetcher(**defaults_args).profile(5903248, 34)

@pytest.mark.parametrize("what", [
    None,
    'PO4',
    ], indirect=False)
def test_predict(fetcher, what, mocked_erddapserver):
    ds = fetcher.to_xarray()
    ds = ds.argo.canyon_med.predict(what)
    assert "CANYON-MED" in ds.attrs['Processing_history']
