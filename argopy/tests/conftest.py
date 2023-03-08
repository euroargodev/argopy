import sys
import os
import logging
from argopy import tutorial
import pytest
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
log = logging.getLogger("argopy.tests.conftests")

def pytest_sessionstart(session):
    log.debug("Starting tests session")
    log.debug("Initial session state: %s" % session)
    pass

@pytest.fixture(scope="module", autouse=True)
def mocked_ftpserver(ftpserver):
    os.environ['FTP_USER'] = 'janedow'
    os.environ['FTP_PASS'] = 'please'
    os.environ['FTP_HOME'] = tempfile.mkdtemp()
    os.environ['FTP_PORT'] = '31175'

    # Set up the ftp server with the tutorial repo GDAC content:
    ftproot, flist = tutorial.open_dataset('localftp')
    for f in flist:
        ftpserver.put_files({"src": f,
                             "dest": f.replace(ftproot, ".")},
                            style="url", anon=True)

    #
    # ftp_login_data = ftpserver.get_login_data()
    MOCKFTP = ftpserver.get_login_data(style="url", anon=True)
    pytest.MOCKFTP = MOCKFTP
    log.debug("Mocked GDAC ftp server located at: %s" % MOCKFTP)

    yield ftpserver

def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(os.getenv('FTP_HOME'))
    log.debug("Ending tests session")
    log.debug("Final session state: %s" % session)
    pass