import pytest
import argopy
from utils import requires_connection


def test_invalid_dataset():
    with pytest.raises(ValueError):
        argopy.tutorial.open_dataset('invalid_dataset')


@requires_connection
def test_localftp_dataset():
    ftproot, flist = argopy.tutorial.open_dataset('localftp')
    assert isinstance(ftproot, str)
    assert isinstance(flist, list)


@requires_connection
def test_weekly_index_dataset():
    rpath, txtfile = argopy.tutorial.open_dataset('weekly_index_prof')
    assert isinstance(txtfile, str)


@requires_connection
def test_global_index_dataset():
    rpath, txtfile = argopy.tutorial.open_dataset('global_index_prof')
    assert isinstance(txtfile, str)
