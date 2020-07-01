#!/bin/env python
# -*coding: UTF-8 -*-
"""
Argo data fetcher for a local copy of GDAC ftp.

This is not intended to be used directly, only by the facade at fetchers.py

Since the GDAC ftp is organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.



About the index local ftp fetcher:

We have a large index csv file "ar_index_global_prof.txt", that is about ~200Mb
For a given request, we need to load/read it
and then to apply a filter to select lines matching the request
With the current version, a dataframe of the full index is cached
and then another cached file is created for the result of the filter.

df_full = pd.read_csv("index.txt")
df_small = filter(df_full)
write_on_file(df_small)

I think we can avoid this with a virtual file system
When a request is done, we



"""
import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
import warnings
import getpass

import multiprocessing as mp
import distributed

from .proto import ArgoDataFetcherProto
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import list_standard_variables, load_dict, mapp_dict, check_localftp
from argopy.options import OPTIONS
from argopy.stores import filestore, indexstore, indexfilter_wmo, indexfilter_box
from argopy.plotters import open_dashboard

access_points = ['wmo', 'box']
exit_formats = ['xarray']
dataset_ids = ['phy', 'bgc']  # First is default


class LocalFTPArgoDataFetcher(ArgoDataFetcherProto):
    """ Manage access to Argo data from a local copy of GDAC ftp """

    ###
    # Methods to be customised for a specific request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @abstractmethod
    def cname(self):
        """ Return a unique string defining the request """
        pass

    @abstractmethod
    def list_argo_files(self, errors: str = 'raise'):
        """ Set the internal list of absolute path of all files to load
        This function must defined the attribute: self._list_of_argo_files with a list of path(s)

        Parameters
        ----------
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If 'ignore', file not found is skipped when fetching data.
        """
        pass

    ###
    # Methods that must not change
    ###
    def __init__(self,
                 local_ftp: str = "",
                 ds: str = "",
                 cache: bool = False,
                 cachedir: str = "",
                 dimension: str = 'point',
                 **kwargs):
        """ Init fetcher

            Parameters
            ----------
            local_ftp : str
                Path to the local directory where the 'dac' folder is located.
            ds : str
                Name of the dataset to load. Use the global OPTIONS['dataset'] by default.
            cache : bool
                Determine if retrieved data should be cached locally or not, False by default.
            cachedir : str
                Absolute path to the cache directory
            dimension : str
                Main dimension of the output dataset. This can be "profile" to retrieve a collection of
                profiles, or "point" (default) to have data as a collection of measurements.
                This can be used to optimise performances

        """
        self.cache = cache
        self.cachedir = cachedir
        self.fs = filestore(cache=self.cache, cachedir=self.cachedir)
        self.definition = 'Local ftp Argo data fetcher'
        self.dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self.local_ftp = OPTIONS['local_ftp'] if local_ftp == '' else local_ftp
        check_localftp(self.local_ftp, errors='raise')  # Validate local_ftp
        self.init(**kwargs)

    def __repr__(self):
        summary = ["<datafetcher '%s'>" % self.definition]
        summary.append("FTP: %s" % self.local_ftp)
        summary.append("Domain: %s" % self.cname())
        return '\n'.join(summary)

    def _absfilepath(self, wmo: int, cyc: int = None, errors: str = 'raise') -> str:
        """ Return the absolute netcdf file path to load for a given wmo/cyc pair

        Based on the dataset, the wmo and the cycle requested, return the absolute path toward the file to load.

        The file is searched using its expected file name pattern (following GDAC conventions).

        If more than one file are found to match the pattern, the first 1 (alphabeticaly) is returned.

        If no files match the pattern, the function can raise an error or fail silently and return None.

        Parameters
        ----------
        wmo: int
            WMO float code
        cyc: int, optional
            Cycle number (None by default)
        errors: {'raise', 'ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if the requested
            file cannot be found. If set to 'ignore', return None silently.

        Returns
        -------
        netcdf_file_path : str
        """
        # This function will be used whatever the access point, since we are working with a GDAC like set of files
        def _filepathpattern(wmo, cyc=None):
            """ Return a file path pattern to scan for a given wmo/cyc pair

            Based on the dataset and the cycle number requested, construct the closest file path pattern to be loaded

            This path is absolute, the pattern can contain '*', and it is the file path, so it has '.nc' extension

            Returns
            -------
            file_path_pattern : str
            """
            if cyc is None:
                # Multi-profile file:
                # <FloatWmoID>_prof.nc
                if self.dataset_id == 'phy':
                    return os.path.sep.join([self.local_ftp, "dac", "*", str(wmo), "%i_prof.nc" % wmo])
                elif self.dataset_id == 'bgc':
                    return os.path.sep.join([self.local_ftp, "dac", "*", str(wmo), "%i_Sprof.nc" % wmo])
            else:
                # Single profile file:
                # <B/M/S><R/D><FloatWmoID>_<XXX><D>.nc
                if cyc < 1000:
                    return os.path.sep.join([self.local_ftp, "dac", "*", str(wmo), "profiles", "*%i_%0.3d*.nc" % (wmo, cyc)])
                else:
                    return os.path.sep.join([self.local_ftp, "dac", "*", str(wmo), "profiles", "*%i_%0.4d*.nc" % (wmo, cyc)])

        pattern = _filepathpattern(wmo, cyc)
        lst = sorted(glob(pattern))
        # lst = sorted(self.fs.glob(pattern))  # Much slower than the regular glob !
        if len(lst) == 1:
            return lst[0]
        elif len(lst) == 0:
            if errors == 'raise':
                raise NetCDF4FileNotFoundError(pattern)
            else:
                # Otherwise remain silent/ignore
                # todo: should raise a warning instead ?
                return None
        else:
            # warnings.warn("More than one file to load for a single float cycle ! Return the 1st one by default.")
            # The choice of the file to load depends on the user mode and dataset requested.
            # todo: define a robust choice
            if self.dataset_id == 'phy':
                if cyc is None:
                    # Use the synthetic profile:
                    lst = [file for file in lst if
                           [file for file in [os.path.split(w)[-1] for w in lst] if file[0] == 'S'][0] in file]
                else:
                    # Use the ascent profile:
                    lst = [file for file in lst if
                           [file for file in [os.path.split(w)[-1] for w in lst] if file[-1] != 'D'][0] in file]
                # print('phy', lst[0])
            elif self.dataset_id == 'bgc':
                lst = [file for file in lst if
                       [file for file in [os.path.split(w)[-1] for w in lst] if file[0] == 'M'][0] in file]
                # print('bgc:', lst)
            return lst[0]

    @property
    def files(self):
        """ Return the list of files to load """
        if not hasattr(self, '_list_of_argo_files'):
            self.list_argo_files()
        return self._list_of_argo_files

    @property
    def cachepath(self):
        """ Return path to cache file for this request """
        return [self.fs.cachepath(file) for file in self.files]

    def xload_multiprof(self, ncfile: str):
        """Load one Argo multi-profile file as a collection of points

        Parameters
        ----------
        ncfile: str
            Absolute path to a netcdf file to load

        Returns
        -------
        :class:`xarray.Dataset`

        """
        ds = self.fs.open_dataset(ncfile, decode_cf=1, use_cftime=0, mask_and_scale=1, engine='h5netcdf')

        # Replace JULD and JULD_QC by TIME and TIME_QC
        ds = ds.rename({'JULD': 'TIME', 'JULD_QC': 'TIME_QC'})
        ds['TIME'].attrs = {'long_name': 'Datetime (UTC) of the station',
                            'standard_name':  'time'}
        # Cast data types:
        ds = ds.argo.cast_types()

        # Enforce real pressure resolution: 0.1 db
        for vname in ds.data_vars:
            if 'PRES' in vname and 'QC' not in vname:
                ds[vname].values = np.round(ds[vname].values, 1)

        # Remove variables without dimensions:
        # We should be able to find a way to keep them somewhere in the data structure
        for v in ds.data_vars:
            if len(list(ds[v].dims)) == 0:
                ds = ds.drop_vars(v)

        ds = ds.argo.profile2point()  # Default output is a collection of points

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == 'phy':
            ds.attrs['DATA_ID'] = 'ARGO'
        if self.dataset_id == 'bgc':
            ds.attrs['DATA_ID'] = 'ARGO-BGC'
        ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
        ds.attrs['Fetched_from'] = self.local_ftp
        ds.attrs['Fetched_by'] = getpass.getuser()
        ds.attrs['Fetched_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
        ds.attrs['Fetched_constraints'] = self.cname()
        ds.attrs['Fetched_uri'] = ncfile
        ds = ds[np.sort(ds.data_vars)]

        return ds

    def open_mfdataset(self, **kwargs):
        """ Load data as efficiently as possible

        This allows to manage parallel retrieval of multiple files with a Dask client or a Multiprocessing pool.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        client = None if 'client' not in kwargs else kwargs['client']

        if len(self.files) == 1:
            return self.xload_multiprof(self.files[0])

        else:
            warnings.warn("Fetching more than one file in a single request is not yet fully reliable. "
                          "If you encounter an error, try to load each float separately.")

            if client is not None:
                if type(client) == distributed.client.Client:
                    # Use dask client:
                    futures = client.map(self.xload_multiprof, self.files)
                    results = client.gather(futures)
                else:
                    # Use multiprocessing Pool
                    with mp.Pool() as pool:
                        results = pool.map(self.xload_multiprof, self.files)
            else:
                results = []
                for f in self.files:
                    results.append(self.xload_multiprof(f))

        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            # ds = xr.concat(results, dim='N_POINTS', data_vars='all', coords='all', compat='equals')
            ds = xr.concat(results, dim='N_POINTS', data_vars='all', coords='all', compat='override')
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))  # Re-index to avoid duplicate values
            ds = ds.set_coords('N_POINTS')
            ds = ds.sortby('TIME')
            return ds
        else:
            raise ValueError("CAN'T FETCH ANY DATA !")

    def to_xarray(self, errors: str = 'raise', client=None):
        """ Load Argo data and return a xarray.Dataset

        Parameters
        ----------
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If 'ignore', file not found is skipped when fetching data.

        client: None, dask.client or 'mp'

        Returns
        -------
        :class:`xarray.Dataset`
        """
        # Set internal list of files to load:
        self.list_argo_files(errors=errors)

        # Load data (will raise an error if no data found):
        ds = self.open_mfdataset(client=client)

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == 'phy':
            ds.attrs['DATA_ID'] = 'ARGO'
        if self.dataset_id == 'bgc':
            ds.attrs['DATA_ID'] = 'ARGO-BGC'
        ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
        ds.attrs['Fetched_from'] = self.local_ftp
        ds.attrs['Fetched_by'] = getpass.getuser()
        ds.attrs['Fetched_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
        ds.attrs['Fetched_constraints'] = self.cname()
        if len(self.files) == 1:
            ds.attrs['Fetched_uri'] = self.files[0]
        return ds

    def filter_data_mode(self, ds, **kwargs):
        ds = ds.argo.filter_data_mode(errors='ignore', **kwargs)
        if ds.argo._type == 'point':
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))
        return ds

    def filter_qc(self, ds, **kwargs):
        ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == 'point':
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))
        return ds

    def filter_variables(self, ds, mode='standard'):
        if mode == 'standard':
            to_remove = sorted(list(set(list(ds.data_vars)) - set(list_standard_variables())))
            return ds.drop_vars(to_remove)
        else:
            return ds


class Fetch_wmo(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a list of WMOs  """
    def init(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo data loader for WMOs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
            CYC : int, np.array(int), list(int)
                The cycle numbers to load.
        """
        if isinstance(WMO, int):
            WMO = [WMO]  # Make sure we deal with a list
        if isinstance(CYC, int):
            CYC = np.array((CYC,), dtype='int')  # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(CYC, dtype='int')  # Make sure we deal with an array of integers
        self.WMO = WMO
        self.CYC = CYC
        # Build the internal list of files to load:
        # self.argo_files = [] # We do it in a lazy mode, ie only if data are requested with .to_xarray()
        return self

    def cname(self):
        """ Return a unique string defining the request """
        if len(self.WMO) > 1:
            listname = ["WMO%i" % i for i in sorted(self.WMO)]
            if isinstance(self.CYC, (np.ndarray)):
                [listname.append("CYC%0.4d" % i) for i in sorted(self.CYC)]
            listname = ";".join(listname)
            listname = self.dataset_id + ";" + listname
        else:
            listname = "WMO%i" % self.WMO[0]
            if isinstance(self.CYC, (np.ndarray)):
                listname = [listname]
                [listname.append("CYC%0.4d" % i) for i in sorted(self.CYC)]
                listname = "_".join(listname)
                listname = self.dataset_id + "_" + listname
        return listname

    def list_argo_files(self, errors: str = 'raise'):
        """ Set the internal list of files to load

        Parameters
        ----------
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If 'ignore', file not found is skipped when fetching data.
        """
        if not hasattr(self, '_list_of_argo_files'):
            self._list_of_argo_files = []
            for wmo in self.WMO:
                if self.CYC is None:
                    self._list_of_argo_files.append(self._absfilepath(wmo, errors=errors))
                else:
                    for cyc in self.CYC:
                        self._list_of_argo_files.append(self._absfilepath(wmo, cyc, errors=errors))
        return self

    def dashboard(self, **kw):
        if len(self.WMO) == 1:
            return open_dashboard(wmo=self.WMO[0], **kw)
        else:
            warnings.warn("Plot dashboard only available for one float frequest")


class Fetch_box(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a rectangular space/time domain  """

    def init(self, box: list ):
        """ Create Argo data loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        # We use a full domain definition (x, y, z, t) as argument for compatibility with the other fetchers
        # but we work only with x, y and t.
        if len(box) not in [6, 8]:
            raise ValueError('Box must be 6 or 8 length')

        if len(box) == 6:
            self.BOX = [box[ii] for ii in [0, 1, 2, 3]]
        elif len(box) == 8:
            self.BOX = [box[ii] for ii in [0, 1, 2, 3, 6, 7]]

        # if len(box) == 6:
        #     # Select the last months of data:
        #     end = pd.to_datetime('now')
        #     start = end - pd.DateOffset(months=1)
        #     self.BOX.append(start.strftime('%Y-%m-%d'))
        #     self.BOX.append(end.strftime('%Y-%m-%d'))

        self.fs_index = indexstore(self.cache, self.cachedir, os.path.sep.join([self.local_ftp, "ar_index_global_prof.txt"]))
        # todo we would need to check if the index file is there

    def cname(self):
        """ Return a unique string defining the constraints """
        BOX = self.BOX
        if len(self.BOX) == 4:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % \
                      (BOX[0], BOX[1], BOX[2], BOX[3])
        elif len(self.BOX) == 6:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % \
                      (BOX[0], BOX[1], BOX[2], BOX[3],
                       self._format(BOX[4], 'tim'), self._format(BOX[5], 'tim'))
        boxname = self.dataset_id + "_" + boxname
        return boxname

    def list_argo_files(self, errors: str = 'raise'):
        """ Set the internal list of files to load

        Parameters
        ----------
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If 'ignore', file not found is skipped when fetching data.
        """
        if not hasattr(self, '_list_of_argo_files'):
            self._list_of_argo_files = []
            # Fetch the index to retrieve the list of profiles to load:
            filt = indexfilter_box(self.BOX)
            df_index = self.fs_index.open_dataframe(filt)
            if isinstance(df_index, pd.core.frame.DataFrame):
                # Ok, we found profiles in the index file,
                # so now we can make sure these files exist:
                lst = list(df_index['file'])
                for file in lst:
                    abs_file = os.path.sep.join([self.local_ftp, "dac", file])
                    if self.fs.exists(abs_file):
                        self._list_of_argo_files.append(abs_file)
                    elif errors == 'raise':
                        raise NetCDF4FileNotFoundError(abs_file)
                    else:
                        # Otherwise remain silent/ignore
                        # todo should raise a warning instead ?
                        return None
        return self