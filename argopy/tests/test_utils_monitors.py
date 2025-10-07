import pytest
IPython = pytest.importorskip("IPython", reason="Requires 'IPython'")

from argopy.utils.monitors import badge, fetch_status, monitor_status
from utils import has_ipywidgets, requires_ipywidgets


@pytest.mark.parametrize("insert", [False, True], indirect=False, ids=["insert=%s" % str(i) for i in [False, True]])
def test_badge(insert):
    b = badge(label="label", message="message", color="green", insert=insert)
    if not insert:
        assert isinstance(b, str)
    else:
        assert isinstance(b, IPython.core.display.Image)


def test_fetch_status():
    fs = fetch_status()
    results = fs.fetch()
    assert isinstance(results, dict)
    assert isinstance(fs.text, str)
    assert isinstance(fs.html, str)


@requires_ipywidgets
def test_monitor_status():
    ms = monitor_status()
    assert ms.runner in ['notebook', 'terminal', 'standard', False]
    assert isinstance(ms.content, str)
