# import pytest
from argopy.utils.checkers import is_list_of_strings
from argopy.utils.lists import list_multiprofile_file_variables


def test_list_multiprofile_file_variables():
    assert is_list_of_strings(list_multiprofile_file_variables())
