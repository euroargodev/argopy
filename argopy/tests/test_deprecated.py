import logging
import pytest
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver

import argopy
from utils import (
    requires_erddap,
)


log = logging.getLogger("argopy.tests.deprecated")


def test_deprecated_option_dataset():
    with pytest.deprecated_call():
        argopy.set_options(dataset='phy')


def test_deprecated_option_ftp():
    with pytest.deprecated_call():
        argopy.set_options(ftp='https://data-argo.ifremer.fr')


def test_deprecated_fetcher_argument_ftp():
    with pytest.deprecated_call():
        argopy.DataFetcher(src='gdac', ftp='https://data-argo.ifremer.fr')
