import sys
import os
import logging
import shutil
import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from argopy.tests.helpers.mocked_ftp import mocked_ftpserver
from argopy.tests.helpers.mocked_http import mocked_httpserver


log = logging.getLogger("argopy.tests.conftests")


def pytest_sessionstart(session):
    log.debug("Starting tests session")
    log.debug("Initial session state: %s" % session)
    pass


def pytest_sessionfinish(session, exitstatus):
    try:
        shutil.rmtree(os.getenv('FTP_HOME'))
    except:
        pass
    log.debug("Ending tests session")
    log.debug("Final session state: %s" % session)
    pass

@pytest.fixture(autouse=True)
def _resource_tracker(request):
    """Track resource usage around every test to catch what leaks."""
    try:
        import resource
        import threading
    except:
        print("Can't use _resource_tracker")

    proc = f"/proc/{os.getpid()}"

    def _fds():
        try:
            return len(os.listdir(f"{proc}/fd"))
        except Exception:
            return -1

    def _rss():
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            return -1

    def _threads():
        return threading.active_count()

    before = dict(fds=_fds(), rss=_rss(), threads=_threads())
    yield
    after = dict(fds=_fds(), rss=_rss(), threads=_threads())

    delta_fds = after["fds"] - before["fds"]
    delta_threads = after["threads"] - before["threads"]
    delta_rss = after["rss"] - before["rss"]

    # Only log when something looks wrong
    if delta_fds > 10 or delta_threads > 2 or delta_rss > 500_000:
        print(
            f"\n[RESOURCE LEAK] {request.node.nodeid}\n"
            f"  FDs:     {before['fds']} → {after['fds']} (Δ{delta_fds:+d})\n"
            f"  Threads: {before['threads']} → {after['threads']} (Δ{delta_threads:+d})\n"
            f"  RSS kb:  {before['rss']} → {after['rss']} (Δ{delta_rss:+d})\n"
        )
