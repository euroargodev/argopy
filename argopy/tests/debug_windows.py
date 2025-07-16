#!/usr/bin/env python

import argopy
import fsspec

if __name__ == "__main__":

    argopy.tutorial.open_dataset('gdac')

    fs=fsspec.filesystem('dir', fs=fsspec.filesystem('file'), path='/home/runner/.argopy_tutorial_data/ftp')
    print(fs)
    print(fs.info('dac/aoml/13857/13857_meta.nc'))
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