import os
import pytest
import logging
import tempfile
from argopy import tutorial


log = logging.getLogger("argopy.tests")


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
    log.debug("Mocked GDAC ftp server ready (%s)" % MOCKFTP)

    yield ftpserver

