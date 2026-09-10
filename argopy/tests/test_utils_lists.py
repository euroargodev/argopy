import subprocess
import sys

import pytest

from argopy.utils.checkers import is_list_of_strings
from argopy.utils.lists import list_multiprofile_file_variables, shortcut2gdac


def test_AvailableDataSources_lazy_on_import():
    # Importing argopy must not build the data sources list (which would import
    # every data fetcher). The module-level AVAILABLE_DATA_SOURCES instance must
    # still have an unbuilt (None) _sources right after a fresh import.
    # Run in a clean interpreter so the singleton isn't built by another test.
    code = (
        "import argopy.fetchers as f; assert f.AVAILABLE_DATA_SOURCES._sources is None"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())

shortcuts = {None: dict, 'ftp': str, 'https://data-argo.ifremer.fr': str}

@pytest.mark.parametrize("short", shortcuts.items(),
                         indirect=False,
                         ids=["host=%s" % p for p in shortcuts.keys()])
def test_shortcut2gdac(short):
    assert isinstance(shortcut2gdac(short[0]), short[1])
