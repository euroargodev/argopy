import os
import pytest
import logging
import tempfile
from pathlib import Path
from argopy import tutorial


log = logging.getLogger("argopy.tests.mocked_ftp")


@pytest.fixture(scope="module")
def mocked_ftpserver(ftpserver):
    """https://github.com/oz123/pytest-localftpserver"""
    os.environ['FTP_USER'] = 'janedow'
    os.environ['FTP_PASS'] = 'please'
    os.environ['FTP_HOME'] = tempfile.mkdtemp()
    # os.environ['FTP_PORT'] = '31175'  # Let this be chosen automatically

    # Serve the tutorial GDAC tree over anonymous FTP *without* copying it:
    # point the server's (empty) anon_root at the read-only tutorial tree via a
    # directory symlink. pyftpdlib resolves realpath(root) to the tutorial dir,
    # so every served file resolves inside the root and passes validpath().
    # (Symlinking individual files would fail: their realpath escapes the root.)
    ftproot, flist = tutorial.open_dataset('gdac')
    anon_root = Path(ftpserver.anon_root)
    if anon_root.is_symlink():
        anon_root.unlink()
    elif anon_root.exists():
        anon_root.rmdir()  # freshly-created empty dir from the plugin
    anon_root.symlink_to(ftproot, target_is_directory=True)

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

    # Teardown: restore an empty dir so the plugin's rmtree(anon_root) can't
    # follow the symlink into the real tutorial data, then stop the server.
    if anon_root.is_symlink():
        anon_root.unlink()
        anon_root.mkdir()
    log.info("Teardown mocked GDAC ftp")
    ftpserver.stop()
