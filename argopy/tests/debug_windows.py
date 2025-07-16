#!/usr/bin/env python

import argopy
import fsspec

if __name__ == "__main__":

    argopy.tutorial.open_dataset('gdac')

    p = argopy.tutorial.open_dataset('gdac')[0]
    print(p)
    # UBUNTU:
    # /home/runner/.argopy_tutorial_data/ftp
    # WINDOWS:
    # C:\Users\runneradmin\.argopy_tutorial_data\ftp

    fs=fsspec.filesystem('dir', fs=fsspec.filesystem('file'), path=p)

    print(fs)
    # UBUNTU:
    # DirFileSystem(path='/home/runner/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x7f443b470f10>)
    # WINDOWS:
    # DirFileSystem(path='C:/Users/runneradmin/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x000001BD2C02B790>)

    print(fs.info('dac/aoml/13857/13857_meta.nc'))
    # UBUNTU:
    # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752650272.842394, 'islink': False, 'mode': 33188, 'uid': 1001, 'gid': 118, 'mtime': 1752650272.842394, 'ino': 320798, 'nlink': 1}
    # WINDOWS:
    # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752651012.0670562, 'islink': False, 'mode': 33206, 'uid': 0, 'gid': 0, 'mtime': 1752651012.0670562, 'ino': 844424930409953, 'nlink': 1}

    try:
        print(argopy.utils.dirfs_relpath(fs, 'dac/aoml/13857/13857_meta.nc'))
        # UBUNTU:
        # ----------------------------------------
        # fs.path= /home/runner/.argopy_tutorial_data/ftp
        # prefix= /home/runner/.argopy_tutorial_data/ftp/
        # fs.fs.sep= /
        # fs.path.startswith(fs.fs.sep)= True
        # path.startswith(fs.fs.sep)= False
        # prefix= home/runner/.argopy_tutorial_data/ftp/
        # path= dac/aoml/13857/13857_meta.nc
        # path.startswith(prefix)= False

        # /home/runner/.argopy_tutorial_data/ftp
        # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752650972.7143445, 'islink': False, 'mode': 33188, 'uid': 1001, 'gid': 118, 'mtime': 1752650972.7143445, 'ino': 272885, 'nlink': 1}
        # /home/runner/.argopy_tutorial_data/ftp
        # DirFileSystem(path='/home/runner/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x7f719e25a050>)
        # ----------------------------------------

        # WINDOWS:
        # ----------------------------------------
        # fs.path= C:/Users/runneradmin/.argopy_tutorial_data/ftp
        # prefix= C:/Users/runneradmin/.argopy_tutorial_data/ftp/
        # fs.fs.sep= /
        # fs.path.startswith(fs.fs.sep)= False
        # path.startswith(fs.fs.sep)= False
        # prefix= C:/Users/runneradmin/.argopy_tutorial_data/ftp/
        # path= dac/aoml/13857/13857_meta.nc
        # path.startswith(prefix)= False


    except Exception as e:
        print(e)
        pass

    # p = argopy.tutorial.open_dataset('gdac')[0]
    # print(p)
    # fs=argopy.gdacfs(p)
    # print(fs.info('dac/aoml/13857/13857_meta.nc'))
    #
    # p=argopy.tutorial.open_dataset('gdac')[0]
    # print(p)
    # fs=argopy.gdacfs(p)
    # print(fs.fs)
    # try:
    #     print(argopy.utils.dirfs_relpath(fs.fs,'dac/aoml/13857/13857_meta.nc'))
    # except Exception as e:
    #     print(e)
    #     pass