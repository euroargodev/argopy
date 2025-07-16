#!/usr/bin/env python
# ================================== FAILURES ===================================
# _____________ Test_Gdacfs.test_implementation[host='c', no cache] _____________
#
# self = <argopy.tests.test_stores_fs_gdac.Test_Gdacfs object at 0x0000021AEE3DC290>
# mocked_httpserver = 'http://127.0.0.1:9898'
# store_maker = <argopy.stores.implementations.local.filestore object at 0x0000021A8520C110>
#
#     @pytest.mark.parametrize(
#         "store_maker", scenarios, indirect=True, ids=scenarios_ids
#     )
#     def test_implementation(self, mocked_httpserver, store_maker):
# >       self.assert_fs(store_maker)
#
# argopy\tests\test_stores_fs_gdac.py:118:
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
# argopy\tests\test_stores_fs_gdac.py:109: in assert_fs
#     assert fs.info("dac/aoml/13857/13857_meta.nc")['size'] == 25352
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# argopy\stores\spec.py:89: in info
#     return self.fs.info(path, *args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# C:\Users\runneradmin\micromamba\envs\argopy-tests\Lib\site-packages\fsspec\implementations\dirfs.py:242: in info
#     info["name"] = self._relpath(info["name"])
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#
# self = DirFileSystem(path='C:/Users/runneradmin/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x0000021AE829B110>)
# path = 'C:/Users/runneradmin/.argopy_tutorial_data/ftp/dac/aoml/13857/13857_meta.nc'
#
#     def _relpath(self, path):
#         if isinstance(path, str):
#             if not self.path:
#                 return path
#             # We need to account for S3FileSystem returning paths that do not
#             # start with a '/'
#             if path == self.path or (
#                 self.path.startswith(self.fs.sep) and path == self.path[1:]
#             ):
#                 return ""
#             prefix = self.path + self.fs.sep
#             if self.path.startswith(self.fs.sep) and not path.startswith(self.fs.sep):
#                 prefix = prefix[1:]
# >           assert path.startswith(prefix)
#                    ^^^^^^^^^^^^^^^^^^^^^^^
# E           AssertionError
#
# C:\Users\runneradmin\micromamba\envs\argopy-tests\Lib\site-packages\fsspec\implementations\dirfs.py:74: AssertionError

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

    fs0 = fsspec.filesystem('file')
    # pp = fs0.unstrip_protocol(p)
    # print(pp)
    # # UBUNTU:
    # # file:///home/runner/.argopy_tutorial_data/ftp
    # # WINDOWS:
    # # file://C:/Users/runneradmin/.argopy_tutorial_data/ftp

    fs=fsspec.filesystem('dir', fs=fs0, path=p)
    print(fs)
    # UBUNTU:
    # DirFileSystem(path='/home/runner/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x7f443b470f10>)
    # WINDOWS:
    # DirFileSystem(path='C:/Users/runneradmin/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x000001BD2C02B790>)

    # Ok with a direct dir file system !
    print(fs.info('dac/aoml/13857/13857_meta.nc'))
    # UBUNTU:
    # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752650272.842394, 'islink': False, 'mode': 33188, 'uid': 1001, 'gid': 118, 'mtime': 1752650272.842394, 'ino': 320798, 'nlink': 1}
    # WINDOWS:
    # {'name': 'dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752651012.0670562, 'islink': False, 'mode': 33206, 'uid': 0, 'gid': 0, 'mtime': 1752651012.0670562, 'ino': 844424930409953, 'nlink': 1}

    # try:
    #     print(argopy.utils.dirfs_relpath(fs, '/dac/aoml/13857/13857_meta.nc'))
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


    # except Exception as e:
    #     print(e)
    #     pass
    print('='*30, 'Now with gdacfs')

    fs = argopy.gdacfs(p)
    try:
        print(fs)
        # UBUNTU:
        # <argopy.stores.implementations.local.filestore object at 0x7fe3b5cd4350>
        # WINDOWS:
        # <argopy.stores.implementations.local.filestore object at 0x00000128116A7350>

        print(fs.fs)
        # UBUNTU:
        # DirFileSystem(path='/home/runner/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x7fe3b707a250>)
        # WINDOWS:
        # DirFileSystem(path='C:/Users/runneradmin/.argopy_tutorial_data/ftp', fs=<fsspec.implementations.local.LocalFileSystem object at 0x00000128116ACE90>)

        print(fs.fs.fs)
        # UBUNTU:
        # <fsspec.implementations.local.LocalFileSystem object at 0x7f89c4f2ef90>
        # WINDOWS:
        # <fsspec.implementations.local.LocalFileSystem object at 0x000002232BA8AD90>

        print(fs.info('dac/aoml/13857/13857_meta.nc'))
        # UBUNTU:
        # {'name': '/home/runner/.argopy_tutorial_data/ftp/dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752665535.5351272, 'islink': False, 'mode': 33188, 'uid': 1001, 'gid': 118, 'mtime': 1752665535.5351272, 'ino': 320799, 'nlink': 1}
        # WINDOWS:
        # {'name': 'C:/Users/runneradmin/.argopy_tutorial_data/ftp/dac/aoml/13857/13857_meta.nc', 'size': 25352, 'type': 'file', 'created': 1752665593.1245666, 'islink': False, 'mode': 33206, 'uid': 0, 'gid': 0, 'mtime': 1752665593.1245666, 'ino': 844424930409965, 'nlink': 1}

        print(fs.fs._join('dac/aoml/13857/13857_meta.nc'))
        # UBUNTU:
        # /home/runner/.argopy_tutorial_data/ftp/dac/aoml/13857/13857_meta.nc
        # WINDOWS:
        # C:/Users/runneradmin/.argopy_tutorial_data/ftp\dac/aoml/13857/13857_meta.nc

        print(fs.fs._relpath(fs.fs._join('dac/aoml/13857/13857_meta.nc')))
        # UBUNTU:
        # dac/aoml/13857/13857_meta.nc
        # WINDOWS:
        # dac/aoml/13857/13857_meta.nc

        print(argopy.utils.dirfs_relpath(fs.fs, fs.fs._join('dac/aoml/13857/13857_meta.nc')))
        # UBUNTU:
        # dac/aoml/13857/13857_meta.nc
        # WINDOWS:
        # dac/aoml/13857/13857_meta.nc

        # print(fs.fs._relpath('/dac/aoml/13857/13857_meta.nc'))
        # UBUNTU:
        # WINDOWS:

    except Exception as e:
        print(e)
        pass

    # p=argopy.tutorial.open_dataset('gdac')[0]
    # print(p)
    # fs=argopy.gdacfs(p)
    # print(fs.fs)
    # try:
    #     print(argopy.utils.dirfs_relpath(fs.fs,'dac/aoml/13857/13857_meta.nc'))
    # except Exception as e:
    #     print(e)
    #     pass