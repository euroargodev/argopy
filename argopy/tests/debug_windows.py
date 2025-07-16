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
    # DirFileSystem(path='D:/home/runner/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x0000013A50A7A790>)

    print(fs.info('dac/aoml/13857/13857_meta.nc'))
    # UBUNTU:
    # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752650272.842394, 'islink': False, 'mode': 33188, 'uid': 1001, 'gid': 118, 'mtime': 1752650272.842394, 'ino': 320798, 'nlink': 1}
    # WINDOWS:
    # FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:/home/runner/.argopy_tutorial_data/ftp/dac/aoml/13857/13857_meta.nc'

    try:
        print(argopy.utils.dirfs_relpath(fs, 'dac/aoml/13857/13857_meta.nc'))
    except Exception as e:
        print(e)
        pass

    p = argopy.tutorial.open_dataset('gdac')[0]
    print(p)
    fs=argopy.gdacfs(p)
    print(fs.info('dac/aoml/13857/13857_meta.nc'))

    p=argopy.tutorial.open_dataset('gdac')[0]
    print(p)
    fs=argopy.gdacfs(p)
    print(fs.fs)
    try:
        print(argopy.utils.dirfs_relpath(fs.fs,'dac/aoml/13857/13857_meta.nc'))
    except Exception as e:
        print(e)
        pass