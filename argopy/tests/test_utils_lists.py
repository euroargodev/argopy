import pytest
from argopy.utils.checkers import is_list_of_strings
from argopy.utils.lists import list_multiprofile_file_variables, shortcut2gdac


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())

shortcuts = {None: dict, 'ftp': str, 'https://data-argo.ifremer.fr': str}

@pytest.mark.parametrize("short", shortcuts.items(),
                         indirect=False,
                         ids=["host=%s" % p for p in shortcuts.keys()])
def test_shortcut2gdac(short):
    assert isinstance(shortcut2gdac(short[0]), short[1])
