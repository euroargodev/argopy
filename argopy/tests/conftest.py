import sys
import os
import logging
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from mocked_ftp import mocked_ftpserver
from mocked_http import mocked_httpserver


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