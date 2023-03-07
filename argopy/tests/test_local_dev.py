import os
from shutil import copyfile
import logging
from .. import tutorial
from argopy import DataFetcher as ArgoDataFetcher


log = logging.getLogger("argopy.tests.dev")

ftproot, flist = tutorial.open_dataset('localftp')
log.debug("ftproot: %s" % ftproot)
os.environ['FTP_USER'] = 'janedow'
os.environ['FTP_PASS'] = 'please'
os.environ['FTP_HOME'] = ftproot
os.environ['FTP_PORT'] = '31175'


def your_code_retrieving_files(host, port, file_paths):
    log.debug("host: %s" % host)

def test_your_code_retrieving_files(ftpserver):
    log.debug("get_login_data: %s" % ftpserver.get_login_data())
    log.debug("get_login_data: %s" % ftpserver.get_login_data(style="url", anon=True))
    log.debug("ftpserver.anon_root:%s:" % ftpserver.anon_root)

    # fetcher = ArgoDataFetcher(src='gdac').float(13857)
    # log.debug(fetcher.uri)
    fetcher = ArgoDataFetcher(src='gdac', ftp='localhost:31175').float(13857)
    log.debug(fetcher.uri)

    # log.debug(list(ftpserver.get_file_contents(flist[0])))
    # ftpserver.put_files("test_folder/test_file", style="url", anon=True)


    # dest_path = os.path.join(ftpserver.anon_root, flist[0])
    # copyfile(flist[0], dest_path)
    # your_code_retrieving_files(host="localhost",
    #                            port=ftpserver.server_port,
    #                            file_paths=[{"remote": flist[0],
    #                                         "local": "testfile_downloaded.txt"
    #                                         }])
    # with open(flist[0]) as original, open("testfile_downloaded.txt") as downloaded:
    #     assert original.read() == downloaded.read()