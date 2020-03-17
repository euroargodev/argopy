#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for Ifremer ERDDAP.
#
# This is not intended to be used directly, only by the facade at fetchers.py
#
# Created by gmaze on 09/03/2020

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np

from argopy.utilities import urlopen

from erddapy import ERDDAP
import copy
from erddapy.utilities import parse_dates, quote_string_constraints

from abc import ABC, abstractmethod
from pathlib import Path
import getpass

access_points = ['box', 'wmo']
exit_formats = ['xarray']
dataset_ids = ['phy', 'ref', 'bgc']

class ErddapArgoDataFetcher(ABC):
    """ Manage access to Argo data through Ifremer ERDDAP

        ERDDAP transaction are managed with the erddapy library

        __author__: gmaze@ifremer.fr
    """

    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @abstractmethod
    def define_constraints(self):
        """ Define erddapy constraints """
        pass

    @abstractmethod
    def cname(self):
        """ Return a unique string defining the request """
        pass

    ###
    # Methods that must not changed
    ###
    def __init__(self, ds='phy', cache=False, cachedir=None, **kwargs):
        """ Instantiate an ERDDAP Argo data loader

            Parameters
            ----------
            db: 'phy' or 'ref'
            cache : False
            cachedir : None
        """
        self.cache = cache or not not cachedir # Yes, this is not not
        self.cachedir = cachedir
        if self.cache:
            #todo check if cachedir is a valid path
            Path(self.cachedir).mkdir(parents=True, exist_ok=True)

        self.definition = 'Ifremer erddap Argo data fetcher'
        self.dataset_id = ds
        self.init(**kwargs)
        self._init_erddapy()

    def __repr__(self):
        if hasattr(self, '_definition'):
            summary = [ "<datafetcher '%s'>" % self.definition ]
        else:
            summary = [ "<datafetcher '%s'>" % 'Ifremer erddap Argo data fetcher' ]
        summary.append( "Domain: %s" % self.cname(cache=0) )
        return '\n'.join(summary)

    def _format(self, x, typ):
        """ string formating helper """
        if typ=='lon':
            if x < 0:
                x = 360. + x
            return ("%05d") % (x * 100.)
        if typ=='lat':
            return ("%05d") % (x * 100.)
        if typ=='prs':
            return ("%05d") % (np.abs(x)*10.)
        if typ=='tim':
            return pd.to_datetime(x).strftime('%Y%m%d')
        return str(x)

    def _add_history(self, this, txt):
        if 'history' in this.attrs:
            this.attrs['history'] += "; %s" % txt
        else:
            this.attrs['history'] = txt
        return this

    def _cast_types_deprec(self, this):
        """ Make sure variables are of the appropriate types

            This is hard coded, but should be retrieved from an API somewhere
            #todo move this to the xarray argo accessor
        """
        def cast_this(da, type):
            try:
                da.values = da.values.astype(type)
            except ValueError:
                print("Fail to cast: ", da.dtype, "into:", type)
                print("Possible values:", np.unique(da))
            return da

        for v in this.data_vars:
            if "QC" in v:
                if this[v].dtype == 'O': # convert object to string
                    this[v] = cast_this(this[v], str)

                # Address weird string values:

                if this[v].dtype == '<U3': # string, len 3 because of a 'nan' somewhere
                    ii = this[v] == '   ' # This should not happen, but still ! That's real world data
                    this[v].loc[dict(index=ii)] = '0'

                    ii = this[v] == 'nan' # This should not happen, but still ! That's real world data
                    this[v].loc[dict(index=ii)] = '0'

                    this[v] = cast_this(this[v], np.dtype('U1')) # Get back to regular U1 string

                if this[v].dtype == '<U1': # string
                    ii = this[v] == ' ' # This should not happen, but still ! That's real world data
                    this[v].loc[dict(index=ii)] = '0'

                # finally convert strings to integers:
                this[v] = cast_this(this[v], int)

            if v == 'PLATFORM_NUMBER' and this['PLATFORM_NUMBER'].dtype == 'float64':  # Object
                this['PLATFORM_NUMBER'] = cast_this(this['PLATFORM_NUMBER'], int)

            if v == 'DATA_MODE' and this['DATA_MODE'].dtype == 'O':  # Object
                this['DATA_MODE'] = cast_this(this['DATA_MODE'], str)
            if v == 'DIRECTION' and this['DIRECTION'].dtype == 'O':  # Object
                this['DIRECTION'] = cast_this(this['DIRECTION'], str)
        return this

    def _add_attributes(self, this):
        """ Add variables attributes not return by erddap requests (csv)

            This is hard coded, but should be retrieved from an API somewhere
        """
        for v in this.data_vars:
            if 'TEMP' in v and '_QC' not in v:
                this[v].attrs = {'long_name': 'SEA TEMPERATURE IN SITU ITS-90 SCALE',
                              'standard_name': 'sea_water_temperature',
                              'units': 'degree_Celsius',
                              'valid_min': -2.,
                              'valid_max': 40.,
                              'resolution': 0.001}
                if 'ERROR' in v:
                    this[v].attrs['long_name'] = 'ERROR IN %s' % this[v].attrs['long_name']

        for v in this.data_vars:
            if 'PSAL' in v and '_QC' not in v:
                this[v].attrs = {'long_name': 'PRACTICAL SALINITY',
                              'standard_name': 'sea_water_salinity',
                              'units': 'psu',
                              'valid_min': 0.,
                              'valid_max': 43.,
                              'resolution': 0.001}
                if 'ERROR' in v:
                    this[v].attrs['long_name'] = 'ERROR IN %s' % this[v].attrs['long_name']

        for v in this.data_vars:
            if 'PRES' in v and '_QC' not in v:
                this[v].attrs = {'long_name': 'Sea Pressure',
                              'standard_name': 'sea_water_pressure',
                              'units': 'decibar',
                              'valid_min': 0.,
                              'valid_max': 12000.,
                              'resolution': 0.1,
                              'axis': 'Z'}
                if 'ERROR' in v:
                    this[v].attrs['long_name'] = 'ERROR IN %s' % this[v].attrs['long_name']

        for v in this.data_vars:
            if 'DOXY' in v and '_QC' not in v:
                this[v].attrs = {'long_name': 'Dissolved oxygen',
                              'standard_name': 'moles_of_oxygen_per_unit_mass_in_sea_water',
                              'units': 'micromole/kg',
                              'valid_min': -5.,
                              'valid_max': 600.,
                              'resolution': 0.001}
                if 'ERROR' in v:
                    this[v].attrs['long_name'] = 'ERROR IN %s' % this[v].attrs['long_name']

        for v in this.data_vars:
            if '_QC' in v:
                attrs = {'long_name': "Global quality flag of %s profile" % v,
                         'convention': "Argo reference table 2a"};
                this[v].attrs = attrs

        if 'CYCLE_NUMBER' in this.data_vars:
            this['CYCLE_NUMBER'].attrs = {'long_name': 'Float cycle number',
                             'convention': '0..N, 0 : launch cycle (if exists), 1 : first complete cycle'}

        if 'DATA_MODE' in this.data_vars:
            this['DATA_MODE'].attrs = {'long_name': 'Delayed mode or real time data',
                             'convention': 'R : real time; D : delayed mode; A : real time with adjustment'}

        if 'DIRECTION' in this.data_vars:
            this['DIRECTION'].attrs = {'long_name': 'Direction of the station profiles',
                             'convention': 'A: ascending profiles, D: descending profiles'}

        if 'PLATFORM_NUMBER' in this.data_vars:
            this['PLATFORM_NUMBER'].attrs = {'long_name': 'Float unique identifier',
                             'convention': 'WMO float identifier : A9IIIII'}

        return this

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(
            server='http://www.ifremer.fr/erddap',
            protocol='tabledap'
        )
        self.erddap.response = 'csv'

        if self.dataset_id=='phy':
            self.erddap.dataset_id = 'ArgoFloats'
        elif self.dataset_id=='ref':
            self.erddap.dataset_id = 'argo_reference'
        elif self.dataset_id=='bgc':
            self.erddap.dataset_id = 'ArgoFloats-bio'
        else:
            raise ValueError("Invalid database short name for Ifremer erddap (use: 'phy', 'bgc' or 'ref')")
        return self

    def _minimal_vlist(self):
        """ Return the minimal list of variables to retrieve measurements for """
        vlist = list()
        if self.dataset_id == 'phy' or self.dataset_id == 'bgc':
            plist = ['data_mode', 'latitude', 'longitude',
                     'position_qc', 'time', 'time_qc',
                     'direction', 'platform_number', 'cycle_number']
            [vlist.append(p) for p in plist]

            plist = ['pres', 'temp', 'psal']
            if self.dataset_id == 'bgc':
                plist = ['pres', 'temp', 'psal', 'doxy']
            [vlist.append(p) for p in plist]
            [vlist.append(p + '_qc') for p in plist]
            [vlist.append(p + '_adjusted') for p in plist]
            [vlist.append(p + '_adjusted_qc') for p in plist]
            [vlist.append(p + '_adjusted_error') for p in plist]

        elif self.dataset_id == 'ref':
            plist = ['latitude', 'longitude', 'time',
                     'platform_number', 'cycle_number']
            [vlist.append(p) for p in plist]
            plist = ['pres', 'temp', 'psal', 'ptmp']
            [vlist.append(p) for p in plist]

        return vlist


    @property
    def cachepath(self):
        """ Return path to cache file for this request """
        src = self.cachedir
        file = ("ERargo_%s.nc") % (self.cname(cache=True))
        fcache = os.path.join(src, file)
        return fcache

    @property
    def url(self, response=None):
        """ Return the URL used to download data

        """
        # Replace erddapy get_download_url
        # We need to replace it to better handle http responses with by-passing the _check_url_response
        # https://github.com/ioos/erddapy/blob/fa1f2c15304938cd0aa132946c22b0427fd61c81/erddapy/erddapy.py#L247

        # Define constraint to select this box of data:
        self.define_constraints()  # This will affect self.erddap.constraints

        # Define the list of variables to retrieve
        self.erddap.variables = self._minimal_vlist()

        #
        dataset_id = self.erddap.dataset_id
        protocol = self.erddap.protocol
        variables = self.erddap.variables
        if not response:
            response = self.erddap.response
        constraints = self.erddap.constraints
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"
        if variables:
            variables = ",".join(variables)
            url += f"{variables}"

        if constraints:
            _constraints = copy.copy(constraints)
            for k, v in _constraints.items():
                if k.startswith("time"):
                    _constraints.update({k: parse_dates(v)})
            _constraints = quote_string_constraints(_constraints)
            _constraints = "".join([f"&{k}{v}" for k, v in _constraints.items()])

            url += f"{_constraints}"

        url += '&distinct()&orderBy("time,pres")'
        # In erddapy:
        # url = _distinct(url, **kwargs)
        # return _check_url_response(url, **self.requests_kwargs)
        return url

    def to_xarray(self):
        """ Load Argo data and return a xarray.DataSet """

        # Try to load cached file if requested:
        if self.cache and os.path.exists(self.cachepath):
            ds = xr.open_dataset(self.cachepath)
            ds = ds.argo.cast_types() # Cast data types
            return ds
        # No cache found or requested, so we compute:

        # Download data: get a csv, open it as pandas dataframe, convert it to xarrat dataset
        df = pd.read_csv(urlopen(self.url), parse_dates=True, skiprows=[1])
        ds = xr.Dataset.from_dataframe(df)
        df['time'] = pd.to_datetime(df['time'])
        ds['time'].values = ds['time'].astype(np.datetime64)

        # Post-process the xarray.DataSet:

        # Set coordinates:
        coords = ('latitude', 'longitude', 'time')
        # Convert all coordinate variable names to upper case
        for v in ds.data_vars:
            if v not in coords:
                ds = ds.rename({v: v.upper()})
        ds = ds.set_coords(coords)

        # Cast data types and add variable attributes (not available in the csv download):
        ds = ds.argo.cast_types()
        ds = self._add_attributes(ds)

        # More convention:
        #         ds = ds.rename({'pres': 'pressure'})

        # Add useful attributes to the dataset:
        if self.dataset_id == 'phy':
            ds.attrs['DATA_ID'] = 'ARGO'
        elif self.dataset_id == 'ref':
            ds.attrs['DATA_ID'] = 'ARGO_Reference'
        if self.dataset_id == 'bgc':
            ds.attrs['DATA_ID'] = 'ARGO-BGC'
        ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
        ds.attrs['Downloaded_from'] = self.erddap.server
        ds.attrs['Downloaded_by'] = getpass.getuser()
        ds.attrs['Download_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
        ds.attrs['Download_url'] = self.url
        ds.attrs['Download_constraints'] = self.cname()
        ds = ds[np.sort(ds.data_vars)]

        # Possibly save in cache for later re-use
        if self.cache:
            ds.attrs['cache'] = self.cachepath
            ds.to_netcdf(self.cachepath)

        #
        return ds

    def filter_data_mode(self, ds, keep_error=True):
        """ Filter variables according to their data mode

            For data mode 'R' and 'A': keep <PARAM> (eg: 'PRES', 'TEMP' and 'PSAL')
            For data mode 'D': keep <PARAM_ADJUSTED> (eg: 'PRES_ADJUSTED', 'TEMP_ADJUSTED' and 'PSAL_ADJUSTED')

            This applies to <PARAM> and <PARAM_QC>
        """

        # Define variables to filter:
        if self.dataset_id == 'phy':
            plist = ['pres', 'temp', 'psal']
        elif self.dataset_id == 'bgc':
            plist = ['pres', 'temp', 'psal', 'doxy']
        else:
            raise ValueError('Data mode filtering not necessary for Reference dataset')

        def ds_split_datamode(xds):
            """ Create one dataset for each of the data_mode

                Split full dataset into 3 datasets
            """
            # Real-time:
            argo_r = ds.where(ds['DATA_MODE'] == 'R', drop=True)
            for v in plist:
                vname = v.upper() + '_ADJUSTED'
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
                vname = v.upper() + '_ADJUSTED_QC'
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
                vname = v.upper() + '_ADJUSTED_ERROR'
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
            # Real-time adjusted:
            argo_a = ds.where(ds['DATA_MODE'] == 'A', drop=True)
            for v in plist:
                vname = v.upper()
                if vname in argo_a:
                    argo_a = argo_a.drop_vars(vname)
                vname = v.upper() + '_QC'
                if vname in argo_a:
                    argo_a = argo_a.drop_vars(vname)
            # Delayed mode:
            argo_d = ds.where(ds['DATA_MODE'] == 'D', drop=True)
            return argo_r, argo_a, argo_d

        argo_r, argo_a, argo_d = ds_split_datamode(ds)

        def fill_adjusted_nan(ds, vname):
            """Fill in the adjusted field with the non-adjusted wherever it is NaN

               Ensure to have values even for bad QC data in delayed mode
            """
            ii = ds.where(np.isnan(ds[vname+'_ADJUSTED']), drop=1)['index']
            ds[vname+'_ADJUSTED'].loc[dict(index=ii)] = ds[vname].loc[dict(index=ii)]
            return ds

        argo_d = fill_adjusted_nan(argo_d, 'PRES')
        argo_d = fill_adjusted_nan(argo_d, 'TEMP')
        argo_d = fill_adjusted_nan(argo_d, 'PSAL')
        if 'doxy' in plist:
            argo_d = fill_adjusted_nan(argo_d, 'DOXY')

        # Drop QC fields in delayed mode dataset:
        for v in plist:
            vname = v.upper()
            if vname in argo_d:
                argo_d = argo_d.drop_vars(vname)
            vname = v.upper() + '_QC'
            if vname in argo_d:
                argo_d = argo_d.drop_vars(vname)

        # Then create new arrays with the appropriate variables:
        def new_arrays(argo_r, argo_a, argo_d, vname):
            """ Merge the 3 datasets into a single ine with the appropriate fields

                Homogeneise variable names.
                Based on xarray merge function with ’no_conflicts’: only values
                which are not null in both datasets must be equal. The returned
                dataset then contains the combination of all non-null values.

                Return a xarray.DataArray
            """
            DS = xr.merge(
                (argo_r[vname],
                 argo_a[vname+'_ADJUSTED'].rename(vname),
                 argo_d[vname+'_ADJUSTED'].rename(vname)))
            DS_QC = xr.merge((
                        argo_r[vname+'_QC'],
                        argo_a[vname+'_ADJUSTED_QC'].rename(vname+'_QC'),
                        argo_d[vname+'_ADJUSTED_QC'].rename(vname+'_QC')))
            if keep_error:
                DS_ERROR = xr.merge((
                        argo_a[vname+'_ADJUSTED_ERROR'].rename(vname+'_ERROR'),
                        argo_d[vname+'_ADJUSTED_ERROR'].rename(vname+'_ERROR')))
                DS = xr.merge((DS, DS_QC, DS_ERROR))
            else:
                DS = xr.merge((DS, DS_QC))
            return DS

        PRES = new_arrays(argo_r, argo_a, argo_d, 'PRES')
        TEMP = new_arrays(argo_r, argo_a, argo_d, 'TEMP')
        PSAL = new_arrays(argo_r, argo_a, argo_d, 'PSAL')
        if 'doxy' in plist:
            DOXY = new_arrays(argo_r, argo_a, argo_d, 'DOXY')

        # Create final dataset by merging all available variables
        if 'doxy' in plist:
            final = xr.merge((TEMP, PSAL, PRES, DOXY))
        else:
            final = xr.merge((TEMP, PSAL, PRES))

        # Merge with additional content:
        plist = ['position_qc', 'time', 'time_qc', 'data_mode',
                 'direction', 'platform_number', 'cycle_number']
        for p in plist:
            vname = p.upper()
            if vname in ds:
                final = xr.merge((final, ds[vname]))
        for v in final.data_vars:
            if "QC" in v:
                final[v] = final[v].astype(int)
        final.attrs = ds.attrs
        final = self._add_history(final, 'Variables selected according to DATA_MODE')
        final = final[np.sort(final.data_vars)]

        # Cast data types and add attributes:
        final = final.argo.cast_types()
        final = self._add_attributes(final)

        return final

    def filter_qc(self, this, QC_list=[1, 2], drop=True, mode='all', mask=False):
        """ Filter data set according to QC values

            Mask the dataset for points where 'all' or 'any' of the QC fields has a value in the list of
            integer QC flags.

            This method can return the filtered dataset or the filter mask.
        """
        if mode not in ['all', 'any']:
            raise ValueError("Mode must 'all' or 'any'")

        # Extract QC fields:
        QC_fields = []
        for v in this.data_vars:
            if "QC" in v:
                QC_fields.append(v)
        QC_fields = this[QC_fields]
        for v in QC_fields.data_vars:
            QC_fields[v] = QC_fields[v].astype(int)

        # Now apply filter
        this_mask = xr.DataArray(np.zeros_like(QC_fields['index']), dims=['index'],
                                 coords={'index': QC_fields['index']})
        for v in QC_fields.data_vars:
            for qc in QC_list:
                this_mask += QC_fields[v] == qc
        if mode == 'all':
            this_mask = this_mask == len(QC_fields)  # all
        else:
            this_mask = this_mask >= 1  # any

        if not mask:
            this = this.where(this_mask, drop=drop)
            for v in this.data_vars:
                if "QC" in v:
                    this[v] = this[v].astype(int)
            this = self._add_history(this, 'Variables selected according to QC')
            this = this.argo.cast_types()
            this = self._add_attributes(this)
            return this
        else:
            return this_mask

class ArgoDataFetcher_wmo(ErddapArgoDataFetcher):
    """ Manage access to Argo data through Ifremer ERDDAP for: a list of WMOs

        __author__: gmaze@ifremer.fr
    """

    def init(self, WMO=[6902746, 6902757, 6902766], CYC=None):
        """ Create Argo data loader for WMOs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
            CYC : int, np.array(int), list(int)
                The cycle numbers to load.
        """
        if isinstance(WMO, int):
            WMO = [WMO] # Make sure we deal with a list
        if isinstance(CYC, int):
            CYC = np.array((CYC,), dtype='int') # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(CYC, dtype='int') # Make sure we deal with an array of integers
        self.WMO = WMO
        self.CYC = CYC

        if self.dataset_id=='phy':
            self.definition = 'Ifremer erddap Argo data fetcher for floats'
        elif self.dataset_id=='ref':
            self.definition = 'Ifremer erddap Argo REFERENCE data fetcher for floats'
        return self

    def define_constraints(self):
        """ Define erddap constraints """
        self.erddap.constraints = {'platform_number=~': "|".join(["%i"%i for i in self.WMO])}
        if isinstance(self.CYC, (np.ndarray)):
            self.erddap.constraints.update({'cycle_number=~': "|".join(["%i"%i for i in self.CYC])})
        return self

    def cname(self, cache=False):
        """ Return a unique string defining the constraints """
        if len(self.WMO) > 1:
            if cache:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
            else:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = ";".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
            if isinstance(self.CYC, (np.ndarray)):
                listname = [listname]
                [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
        listname = self.dataset_id + "_" + listname
        return listname

class ArgoDataFetcher_box(ErddapArgoDataFetcher):
    """ Manage access to Argo data through Ifremer ERDDAP for: an ocean rectangle

        __author__: gmaze@ifremer.fr
    """

    def init(self, box=[-65,-55,37,38,0,300,'1900-01-01','2100-12-31']):
        """ Create Argo data loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        if len(box) == 6:
            # Use all time line:
            box.append('1900-01-01')
            box.append('2100-12-31')
        elif len(box) != 8:
            raise ValueError('Box must 6 or 8 length')
        self.BOX = box

        if self.dataset_id=='phy':
            self.definition = 'Ifremer erddap Argo data fetcher for a space/time region'
        elif self.dataset_id=='ref':
            self.definition = 'Ifremer erddap Argo REFERENCE data fetcher for a space/time region'

        return self

    def define_constraints(self):
        """ Define request constraints """
        self.erddap.constraints = {'longitude>=': self.BOX[0]}
        self.erddap.constraints.update({'longitude<=': self.BOX[1]})
        self.erddap.constraints.update({'latitude>=': self.BOX[2]})
        self.erddap.constraints.update({'latitude<=': self.BOX[3]})
        self.erddap.constraints.update({'pres>=': self.BOX[4]})
        self.erddap.constraints.update({'pres<=': self.BOX[5]})
        self.erddap.constraints.update({'time>=': self.BOX[6]})
        self.erddap.constraints.update({'time<=': self.BOX[7]})
        return None

    def cname(self, cache=False):
        """ Return a unique string defining the constraints """
        BOX = self.BOX
        if cache:
            boxname = ("%s_%s_%s_%s_%s_%s_%s_%s") % (self._format(BOX[0], 'lon'), self._format(BOX[1], 'lon'),
                                                     self._format(BOX[2], 'lat'), self._format(BOX[3], 'lat'),
                                                     self._format(BOX[4], 'prs'), self._format(BOX[5], 'prs'),
                                                     self._format(BOX[6], 'tim'), self._format(BOX[7], 'tim'))
        else:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; z=%0.1f/%0.1f; t=%s/%s]") % \
                      (BOX[0],BOX[1],BOX[2],BOX[3],BOX[4],BOX[5],
                       self._format(BOX[6], 'tim'), self._format(BOX[7], 'tim'))

        boxname = self.dataset_id + "_" + boxname
        return boxname

class ArgoDataFetcher_box_deployments(ErddapArgoDataFetcher):
    """ Manage access to Argo data through Ifremer ERDDAP for: an ocean rectangle and 1st cycles only

        __author__: gmaze@ifremer.fr
    """

    def init(self, box=[-180,180,-90,90,0,50,'2020-01-01','2020-01-31']):
        """ Create Argo data loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        if len(box) == 6:
            #todo Use last current month
            box.append('2020-01-01')
            box.append('2020-01-31')
        elif len(box) != 8:
            raise ValueError('Box must 6 or 8 length')
        self.BOX = box

        if self.dataset_id=='phy':
            self.definition = 'Ifremer erddap Argo data fetcher for deployments in a space/time region'
        elif self.dataset_id=='ref':
            self.definition = 'Ifremer erddap Argo REFERENCE data fetcher for deployments in a space/time region'

        return self

    def define_constraints(self):
        """ Define request constraints """
        self.erddap.constraints = {'longitude>=': self.BOX[0]}
        self.erddap.constraints.update({'longitude<=': self.BOX[1]})
        self.erddap.constraints.update({'latitude>=': self.BOX[2]})
        self.erddap.constraints.update({'latitude<=': self.BOX[3]})
        self.erddap.constraints.update({'pres>=': self.BOX[4]})
        self.erddap.constraints.update({'pres<=': self.BOX[5]})
        self.erddap.constraints.update({'time>=': self.BOX[6]})
        self.erddap.constraints.update({'time<=': self.BOX[7]})
        self.erddap.constraints.update({'cycle_number=~': "1"})
        return None

    def cname(self, cache=False):
        """ Return a unique string defining the constraints """
        BOX = self.BOX
        if cache:
            boxname = ("%s_%s_%s_%s_%s_%s_%s_%s_cyc001") % (self._format(BOX[0], 'lon'), self._format(BOX[1], 'lon'),
                                                     self._format(BOX[2], 'lat'), self._format(BOX[3], 'lat'),
                                                     self._format(BOX[4], 'prs'), self._format(BOX[5], 'prs'),
                                                     self._format(BOX[6], 'tim'), self._format(BOX[7], 'tim'))
        else:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; z=%0.1f/%0.1f; t=%s/%s]; CYC=1") % \
                      (BOX[0],BOX[1],BOX[2],BOX[3],BOX[4],BOX[5],
                       self._format(BOX[6], 'tim'), self._format(BOX[7], 'tim'))

        boxname = self.dataset_id + "_" + boxname
        return boxname
