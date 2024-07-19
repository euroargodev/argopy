"""

Useful for documentation, to play with and to test argopy

Data files are hosted on companion repository: http://www.github.com/euroargodev/argopy-data

```
import argopy
ftproot, flist = argopy.tutorial.open_dataset('gdac')
ftproot, flist = argopy.tutorial.open_dataset('weekly_index_prof')
ftproot, flist = argopy.tutorial.open_dataset('global_index_prof')

# To force a new download of the data repo:
argopy.tutorial.repodata().download(overwrite=True)
```
"""

import os
from zipfile import ZipFile
from urllib.request import urlretrieve
import shutil


_DEFAULT_CACHE_DIR = os.path.expanduser(os.path.sep.join(["~", ".argopy_tutorial_data"]))


def open_dataset(name: str) -> tuple:
    """ Open a dataset from the argopy online data repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Refresh dataset with:
    ```
    argopy.tutorial.repodata().download(overwrite=True)
    ```

    Parameters
    ----------
    name: str
        Name of the dataset to load or get information for. It can be one of the following:

            - ``gdac``: A small subset of the GDAC ftp.
            - ``weekly_index_prof``: The weekly profile index file
            - ``global_index_prof``: The global profile index file

    Returns
    -------
    path: str
        Root path to files
    files: list(str) or str
        List of files with the requested dataset

    """
    if name == 'gdac':
        gdacftp = sample_ftp()
        gdacftp.download(overwrite=False)
        return gdacftp.rootpath, gdacftp.ls()

    elif name == 'weekly_index_prof':
        gdacftp = sample_ftp()
        gdacftp.download(overwrite=False)
        flist = gdacftp.ls()
        ifile = [f for f in flist if 'ar_index_this_week_prof.txt' in f][0]
        return gdacftp.rootpath, ifile

    elif name == 'global_index_prof':
        gdacftp = sample_ftp()
        gdacftp.download(overwrite=False)
        flist = gdacftp.ls()
        ifile = [f for f in flist if 'ar_index_global_prof.txt' in f][0]
        return gdacftp.rootpath, ifile

    else:
        raise ValueError("Unknown tutorial dataset ('%s')" % name)


class repodata():
    """ Helper class for the local copy of the repository data """
    def __init__(self, path: str = _DEFAULT_CACHE_DIR):
        self.localpath = os.path.expanduser(path)
        self.downloaded = os.path.isdir(self.localpath)
        pass

    @property
    def rootpath(self):
        if self.downloaded:
            return self.localpath
        else:
            raise FileNotFoundError("Local repository data not found at '%s'.\n "
                                    "Try a: 'argopy.tutorial.repodata().download()' first." % self.localpath)

    def download(self, repo: str = 'argopy-data', branch: str = 'master', overwrite: bool = True):
        """ Download an euroargodev repository

            Override the destination folder !
        """
        if self.downloaded:
            if overwrite:
                shutil.rmtree(self.localpath)
            else:
                # warnings.warn("Data already downloaded !")
                return self.rootpath
        # else:
            # warnings.warn("Downloading repo data ...")

        zipurl = "https://github.com/euroargodev/%s/archive/%s.zip" % (repo, branch)
        localzipfile = self.localpath + ".zip"

        # Download zip file:
        urlretrieve(zipurl, localzipfile)  # nosec B310 because protocol cannot be modified

        # Expand zip file to a temporary location:
        _tempo_dir = self.localpath + "_master"
        with ZipFile(localzipfile, 'r') as zipObj:
            zipObj.extractall(path=_tempo_dir)  # Extract all the contents of zip file

        # Move the repo dir to final destination:
        shutil.move(os.path.join(_tempo_dir, "%s-%s" % (repo, branch)), self.localpath)

        # Delete temporary location and zip file:
        os.rmdir(_tempo_dir)
        os.remove(localzipfile)

        return self.rootpath


class sample_ftp(repodata):
    """ Helper class for the local GDAC ftp folder """

    @property
    def rootpath(self):
        if os.path.isdir(self.localpath):
            return os.path.join(self.localpath, 'ftp')
        else:
            raise FileNotFoundError("Local repository data not found at '%s'.\n "
                                    "Try a: 'argopy.tutorial.repodata().download()' first." % self.localpath)

    def ls(self):
        """ Return the list of files in the local_work local GDAC ftp sample """
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.rootpath):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        return listOfFiles
