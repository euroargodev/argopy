import os
import pytest
import io
import argopy
from argopy.utils.locals import modified_environ


@pytest.mark.parametrize("conda", [False, True],
                         indirect=False,
                         ids=["conda=%s" % str(p) for p in [False, True]])
def test_show_versions(conda):
    f = io.StringIO()
    argopy.show_versions(file=f, conda=conda)
    assert "SYSTEM" in f.getvalue()


def test_modified_environ():
    os.environ["DUMMY_ENV_ARGOPY"] = 'initial'
    with modified_environ(DUMMY_ENV_ARGOPY='toto'):
        assert os.environ['DUMMY_ENV_ARGOPY'] == 'toto'
    assert os.environ['DUMMY_ENV_ARGOPY'] == 'initial'
    os.environ.pop('DUMMY_ENV_ARGOPY')
