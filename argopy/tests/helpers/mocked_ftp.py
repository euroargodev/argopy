import os
import pytest
import logging
import tempfile
from argopy import tutorial


log = logging.getLogger("argopy.tests.mocked_ftp")


@pytest.fixture(scope="module", autouse=True)
def mocked_ftpserver(ftpserver):
    """https://github.com/oz123/pytest-localftpserver"""
    os.environ['FTP_USER'] = 'janedow'
    os.environ['FTP_PASS'] = 'please'
    os.environ['FTP_HOME'] = tempfile.mkdtemp()
    # os.environ['FTP_PORT'] = '31175'  # Let this be chosen automatically

    # Set up the ftp server with the tutorial repo GDAC content:
    ftproot, flist = tutorial.open_dataset('gdac')
    for f in flist:
        ftpserver.put_files({"src": f,
                             "dest": f.replace(ftproot, ".")},
                            style="url", anon=True)

    #
    ftp_login_data = ftpserver.get_login_data()
    # log.info(ftp_login_data)
    os.environ['FTP_HOST'] = ftp_login_data['host']
    os.environ['FTP_PORT'] = str(ftp_login_data['port'])
    MOCKFTP = ftpserver.get_login_data(style="url", anon=True)
    pytest.MOCKFTP = MOCKFTP
    log.info("Mocked GDAC ftp server up and ready at %s, serving %i files" % (MOCKFTP, len(flist)))

    # Run test
    yield ftpserver

    # Teardown
    log.info("Teardown mocked GDAC ftp")
    ftpserver.stop()
