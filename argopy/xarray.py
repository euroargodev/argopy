#!/bin/env python
# -*coding: UTF-8 -*-
#

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

from argopy.errors import NetCDF4FileNotFoundError, InvalidDatasetStructure
from sklearn import preprocessing

@xr.register_dataset_accessor('argo')
class ArgoAccessor:
    """

        Class registered under scope ``argo`` to access a :class:`xarray.Dataset` object.

        # Ensure all variables are of the Argo required dtype
        ds.argo.cast_types()

        # Convert a collection of points into a collection of profiles
        ds.argo.point2profile()
        # Convert a collection of profiles to a collection of points
        ds.argo.profile2point()

        #todo Implement new features in ArgoAccessor:

        # Make sure that the dataset complies with Argo vocabulary
        # Should be done at init with a private function ???
        # This could be usefull if a netcdf file is open directly
        ds.argo.check()


     """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._added = list() # Will record all new variables added by argo
        self._dims = list(xarray_obj.dims.keys()) # Store the initial list of dimensions
        self.encoding = xarray_obj.encoding
        self.attrs = xarray_obj.attrs

        if 'N_PROF' in self._dims:
            self._type = 'profile'
        elif 'index' in self._dims:
            self._type = 'point'
        else:
            raise InvalidDatasetStructure("Argo dataset structure not recognised")

    def _add_history(self, txt):
        if 'history' in self._obj.attrs:
            self._obj.attrs['history'] += "; %s" % txt
        else:
            self._obj.attrs['history'] = txt

    def cast_types(self):
        """ Make sure variables are of the appropriate types

            This is hard coded, but should be retrieved from an API somewhere
            Should be able to handle all possible variables encountered in the Argo dataset
        """
        ds = self._obj

        list_str = ['PLATFORM_NUMBER', 'DATA_MODE', 'DIRECTION', 'DATA_CENTRE', 'DATA_TYPE', 'FORMAT_VERSION',
                    'HANDBOOK_VERSION',
                    'PROJECT_NAME', 'PI_NAME', 'STATION_PARAMETERS', 'DATA_CENTER', 'DC_REFERENCE',
                    'DATA_STATE_INDICATOR',
                    'PLATFORM_TYPE', 'FIRMWARE_VERSION', 'POSITIONING_SYSTEM', 'PROFILE_PRES_QC', 'PROFILE_PSAL_QC',
                    'PROFILE_TEMP_QC',
                    'PARAMETER', 'SCIENTIFIC_CALIB_EQUATION', 'SCIENTIFIC_CALIB_COEFFICIENT',
                    'SCIENTIFIC_CALIB_COMMENT',
                    'HISTORY_INSTITUTION', 'HISTORY_STEP', 'HISTORY_SOFTWARE', 'HISTORY_SOFTWARE_RELEASE',
                    'HISTORY_REFERENCE', 'HISTORY_QCTEST',
                    'HISTORY_ACTION', 'HISTORY_PARAMETER', 'VERTICAL_SAMPLING_SCHEME', 'FLOAT_SERIAL_NO']
        list_int = ['PLATFORM_NUMBER', 'WMO_INST_TYPE', 'WMO_INST_TYPE', 'CYCLE_NUMBER', 'CONFIG_MISSION_NUMBER']
        list_datetime = ['REFERENCE_DATE_TIME', 'DATE_CREATION', 'DATE_UPDATE',
                         'JULD', 'JULD_LOCATION', 'SCIENTIFIC_CALIB_DATE', 'HISTORY_DATE']

        def cast_this(da, type):
            """ Low-level casting of DataArray values """
            try:
                da.values = da.values.astype(type)
                da.attrs['casted'] = 1
            except:
                print("Oops!", sys.exc_info()[0], "occured.")
                print("Fail to cast: ", da.dtype, "into:", type, "for: ", da.name)
                print("Encountered unique values:", np.unique(da))
            return da

        def cast_this_da(da):
            """ Cast any DataArray """
            da.attrs['casted'] = 0
            if v in list_str and da.dtype == 'O':  # Object
                da = cast_this(da, str)

            if v in list_int:  # and da.dtype == 'O':  # Object
                da = cast_this(da, int)

            if v in list_datetime and da.dtype == 'O':  # Object
                if 'conventions' in da.attrs and da.attrs['conventions'] == 'YYYYMMDDHHMISS':
                    if da.size != 0:
                        if len(da.dims) <= 1:
                            val = da.astype(str).values.astype('U14')
                            val[val == '              '] = 'nan'  # This should not happen, but still ! That's real world data
                            da.values = pd.to_datetime(val, format='%Y%m%d%H%M%S')
                        else:
                            s = da.stack(dummy_index=da.dims)
                            val = s.astype(str).values.astype('U14')
                            val[val == '              '] = 'nan'  # This should not happen, but still ! That's real world data
                            s.values = pd.to_datetime(val, format='%Y%m%d%H%M%S')
                            da.values = s.unstack('dummy_index')
                        da = cast_this(da, np.datetime64)
                    else:
                        da = cast_this(da, np.datetime64)

                elif v == 'SCIENTIFIC_CALIB_DATE':
                    da = cast_this(da, str)
                    s = da.stack(dummy_index=da.dims)
                    s.values = pd.to_datetime(s.values, format='%Y%m%d%H%M%S')
                    da.values = (s.unstack('dummy_index')).values
                    da = cast_this(da, np.datetime64)

            if "QC" in v and "PROFILE" not in v and "QCTEST" not in v:
                if da.dtype == 'O':  # convert object to string
                    da = cast_this(da, str)

                # Address weird string values:
                # (replace missing or nan values by a '0' that will be cast as a integer later

                if da.dtype == '<U3':  # string, len 3 because of a 'nan' somewhere
                    ii = da == '   '  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, '0', da)

                    ii = da == 'nan'  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, '0', da)

                    da = cast_this(da, np.dtype('U1'))  # Get back to regular U1 string

                if da.dtype == '<U1':  # string
                    ii = da == ' '  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, '0', da)

                    ii = da == 'n'  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, '0', da)

                # finally convert QC strings to integers:
                da = cast_this(da, int)

            if da.dtype != 'O':
                da.attrs['casted'] = 1

            return da

        for v in ds.data_vars:
            try:
                ds[v] = cast_this_da(ds[v])
            except:
                print("Oops!", sys.exc_info()[0], "occured.")
                print("Fail to cast: %s " % v)
                print("Encountered unique values:", np.unique(ds[v]))
                raise

        return ds

    def filter_data_mode(self, keep_error: bool = True, errors: str = 'raise'):
        """ Filter variables according to their data mode

            This applies to <PARAM> and <PARAM_QC>

            For data mode 'R' and 'A': keep <PARAM> (eg: 'PRES', 'TEMP' and 'PSAL')
            For data mode 'D': keep <PARAM_ADJUSTED> (eg: 'PRES_ADJUSTED', 'TEMP_ADJUSTED' and 'PSAL_ADJUSTED')

        Parameters
        ----------
        keep_error: bool, optional
            If true (default) keep the measurements error fields or not.

        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a InvalidDatasetStructure error if any of the expected dataset variables is
            not found. If 'ignore', fails silently and return unmodified dataset.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if self._type != 'point':
            raise InvalidDatasetStructure("Method only available to a collection of points")

        #########
        # Sub-functions
        #########
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

        def fill_adjusted_nan(ds, vname):
            """Fill in the adjusted field with the non-adjusted wherever it is NaN

               Ensure to have values even for bad QC data in delayed mode
            """
            ii = ds.where(np.isnan(ds[vname + '_ADJUSTED']), drop=1)['index']
            ds[vname + '_ADJUSTED'].loc[dict(index=ii)] = ds[vname].loc[dict(index=ii)]
            return ds

        def new_arrays(argo_r, argo_a, argo_d, vname):
            """ Merge the 3 datasets into a single one with the appropriate fields

                Homogeneise variable names.
                Based on xarray merge function with ’no_conflicts’: only values
                which are not null in both datasets must be equal. The returned
                dataset then contains the combination of all non-null values.

                Return a xarray.DataArray
            """
            DS = xr.merge(
                (argo_r[vname],
                 argo_a[vname + '_ADJUSTED'].rename(vname),
                 argo_d[vname + '_ADJUSTED'].rename(vname)))
            DS_QC = xr.merge((
                argo_r[vname + '_QC'],
                argo_a[vname + '_ADJUSTED_QC'].rename(vname + '_QC'),
                argo_d[vname + '_ADJUSTED_QC'].rename(vname + '_QC')))
            if keep_error:
                DS_ERROR = xr.merge((
                    argo_a[vname + '_ADJUSTED_ERROR'].rename(vname + '_ERROR'),
                    argo_d[vname + '_ADJUSTED_ERROR'].rename(vname + '_ERROR')))
                DS = xr.merge((DS, DS_QC, DS_ERROR))
            else:
                DS = xr.merge((DS, DS_QC))
            return DS

        #########
        # filter
        #########
        ds = self._obj
        if 'DATA_MODE' not in ds:
            if errors:
                raise InvalidDatasetStructure("Method only available for dataset with a 'DATA_MODE' variable ")
            else:
                #todo should raise a warning instead ?
                return ds

        # Define variables to filter:
        possible_list = ['PRES', 'TEMP', 'PSAL',
                         'DOXY',
                         'CHLA',
                         'BBP532',
                         'BBP700',
                         'DOWNWELLING_PAR',
                         'DOWN_IRRADIANCE380',
                         'DOWN_IRRADIANCE412',
                         'DOWN_IRRADIANCE490']
        plist = [p for p in possible_list if p in ds.data_vars]

        # Create one dataset for each of the data_mode:
        argo_r, argo_a, argo_d = ds_split_datamode(ds)

        # Fill in the adjusted field with the non-adjusted wherever it is NaN
        for v in plist:
            argo_d = fill_adjusted_nan(argo_d, v.upper())

        # Drop QC fields in delayed mode dataset:
        for v in plist:
            vname = v.upper()
            if vname in argo_d:
                argo_d = argo_d.drop_vars(vname)
            vname = v.upper() + '_QC'
            if vname in argo_d:
                argo_d = argo_d.drop_vars(vname)

        # Create new arrays with the appropriate variables:
        vlist = [new_arrays(argo_r, argo_a, argo_d, v) for v in plist]

        # Create final dataset by merging all available variables
        final = xr.merge(vlist)

        # Merge with all other variables:
        other_variables = list(set([v for v in list(ds.data_vars) if 'ADJUSTED' not in v]) - set(list(final.data_vars)))
        # other_variables.remove('DATA_MODE')  # Not necessary anymore
        for p in other_variables:
            final = xr.merge((final, ds[p]))

        final.attrs = ds.attrs
        final.argo._add_history('Variables filtered according to DATA_MODE')
        final = final[np.sort(final.data_vars)]

        # Cast data types and add attributes:
        final = final.argo.cast_types()

        return final

    def filter_qc(self, QC_list=[1, 2], drop=True, mode='all', mask=False):
        """ Filter data set according to QC values

            Mask the dataset for points where 'all' or 'any' of the QC fields has a value in the list of
            integer QC flags.

            This method can return the filtered dataset or the filter mask.
        """
        if self._type != 'point':
            raise InvalidDatasetStructure("Method only available to a collection of points")

        if mode not in ['all', 'any']:
            raise ValueError("Mode must 'all' or 'any'")

        this = self._obj

        # Extract QC fields:
        QC_fields = []
        for v in this.data_vars:
            if "QC" in v and "PROFILE" not in v:
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
                if "QC" in v and "PROFILE" not in v:
                    this[v] = this[v].astype(int)
            this.argo._add_history('Variables selected according to QC')
            this = this.argo.cast_types()
            return this
        else:
            return this_mask

    def uid(self, wmo_or_uid, cyc=None, direction=None):
        """ UID encoder/decoder

        Parameters
        ----------
        int
            WMO number (to encode) or UID (to decode)
        cyc: int, optional
            Cycle number (to encode), not used to decode
        direction: str, optional
            Direction of the profile, must be 'A' (Ascending) or 'D' (Descending)

        Return
        ------
        int or tuple of int

        Example
        -------
        unique_float_profile_id = uid(690024,13,'A') # Encode
        wmo, cyc, drc = uid(unique_float_profile_id) # Decode
        """
        le = preprocessing.LabelEncoder()
        le.fit(['A', 'D'])

        def encode_direction(x):
            y = 1 - le.transform(x)
            return np.where(y == 0, -1, y)

        def decode_direction(x):
            y = 1 - np.where(x == -1, 0, x)
            return le.inverse_transform(y)

        offset = 1e5

        if cyc is not None:
            # ENCODER
            if direction is not None:
                return encode_direction(direction) * np.vectorize(int)(offset * wmo_or_uid + cyc).ravel()
            else:
                return np.vectorize(int)(offset * wmo_or_uid + cyc).ravel()
        else:
            # DECODER
            drc = decode_direction(np.sign(wmo_or_uid))
            wmo = np.vectorize(int)(np.abs(wmo_or_uid) / offset)
            cyc = -np.vectorize(int)(offset * wmo - np.abs(wmo_or_uid))
            return wmo, cyc, drc

    def point2profile(self):
        """ Transform a collection of points into a collection of profiles

        """
        if self._type != 'point':
            raise InvalidDatasetStructure("Method only available to a collection of points")
        this = self._obj # Should not be modified

        def fillvalue(da):
            """ Return fillvalue for a dataarray """
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
            if da.dtype.kind in ['U']:
                fillvalue = ' '
            elif da.dtype.kind == 'i':
                fillvalue = 99999
            elif da.dtype.kind == 'M':
                fillvalue = np.datetime64("NaT")
            else:
                fillvalue = np.nan
            return fillvalue

        # Find the number of profiles (N_PROF) and vertical levels (N_LEVELS):
        dummy_argo_uid = xr.DataArray(self.uid(this['PLATFORM_NUMBER'].values,
                                               this['CYCLE_NUMBER'].values,
                                               this['DIRECTION'].values),
                                      dims='index',
                                      coords={'index': this['index']},
                                      name='dummy_argo_uid')
        N_PROF = len(np.unique(dummy_argo_uid))
        that = this.groupby(dummy_argo_uid)

        N_LEVELS = int(xr.DataArray(np.ones_like(this['index'].values),
                                    dims='index',
                                    coords={'index': this['index']})
                       .groupby(dummy_argo_uid).sum().max().values)
        assert N_PROF * N_LEVELS >= len(this['index'])

        # Store the initial set of coordinates:
        coords_list = list(this.coords)
        this = this.reset_coords()

        # For each variables, determine if it has unique value by profile,
        # if yes: the transformed variable should be [N_PROF]
        # if no: the transformed variable should be [N_PROF, N_LEVELS]
        count = np.zeros((N_PROF, len(this.data_vars)), 'int')
        for i_prof, grp in enumerate(this.groupby(dummy_argo_uid)):
            i_uid, prof = grp
            for iv, vname in enumerate(this.data_vars):
                count[i_prof, iv] = len(np.unique(prof[vname]))
        # Variables with a unique value for each profiles:
        list_1d = list(np.array(this.data_vars)[count.sum(axis=0) == count.shape[0]])
        # Variables with more than 1 value for each profiles:
        list_2d = list(np.array(this.data_vars)[count.sum(axis=0) != count.shape[0]])

        # Create new empty dataset:
        new_ds = []
        for vname in list_2d:
            new_ds.append(xr.DataArray(np.full((N_PROF, N_LEVELS), fillvalue(this[vname]), dtype=this[vname].dtype),
                                       dims=['N_PROF', 'N_LEVELS'],
                                       coords={'N_PROF': np.arange(N_PROF),
                                               'N_LEVELS': np.arange(N_LEVELS)},
                                       name=vname))
        for vname in list_1d:
            new_ds.append(xr.DataArray(np.full((N_PROF,), fillvalue(this[vname]), dtype=this[vname].dtype),
                                       dims=['N_PROF'],
                                       coords={'N_PROF': np.arange(N_PROF)},
                                       name=vname))
        new_ds = xr.merge(new_ds)

        # Now fill in each profile values:
        for i_prof, grp in enumerate(this.groupby(dummy_argo_uid)):
            i_uid, prof = grp
            for iv, vname in enumerate(this.data_vars):
                if len(new_ds[vname].dims) == 2:  # ['N_PROF', 'N_LEVELS'] array:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    try:
                        y[i_prof, 0:len(x)] = x
                    except:
                        print(vname, 'input', x.shape, 'output', y[i_prof, :].shape)
                        raise
                    new_ds[vname].values = y
                else:  # ['N_PROF', ] array:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    y[i_prof] = np.unique(x)[0]

        # Restore coordinate variables:
        new_ds = new_ds.set_coords([c for c in coords_list if c in new_ds])

        # Misc formating
        new_ds = new_ds.sortby('TIME')
        new_ds = new_ds.argo.cast_types()
        new_ds = new_ds[np.sort(new_ds.data_vars)]
        new_ds.encoding = self.encoding # Preserve low-level encoding information
        new_ds.attrs = self.attrs # Preserve original attributes
        new_ds.argo._add_history('Transformed with point2profile')
        new_ds.argo._type = 'profile'
        return new_ds

    def profile2point(self):
        """ Convert a collection of profiles to a collection of points """
        if self._type != 'profile':
            raise InvalidDatasetStructure("Method only available for a collection of profiles (N_PROF dimemsion)")
        ds = self._obj

        # Remove all variables for which a dimension is length=0 (eg: N_HISTORY)
        dim_list = []
        for v in ds.data_vars:
            dims = ds[v].dims
            for d in dims:
                if len(ds[d]) == 0:
                    dim_list.append(d)
                    break
        ds = ds.drop_dims(np.unique(dim_list))  # Drop dimensions and associated variables from this dataset

        # Remove any variable that is not with dimensions (N_PROF,) or (N_PROF, N_LEVELS)
        for v in ds:
            dims = list(ds[v].dims)
            dims = ".".join(dims)
            if dims not in ['N_PROF', 'N_PROF.N_LEVELS']:
                ds = ds.drop_vars(v)

        ds, = xr.broadcast(ds)
        ds = ds.stack({'index':list(ds.dims)})
        ds = ds.reset_index('index').drop_vars(['N_PROF', 'N_LEVELS'])
        possible_coords = ['LATITUDE', 'LONGITUDE', 'TIME', 'JULD', 'index']
        for c in [c for c in possible_coords if c in ds.data_vars]:
            ds = ds.set_coords(c)

        ds = ds.where(~np.isnan(ds['PRES']), drop=1) # Remove index without data (useless points)
        ds['index'] = np.arange(0, len(ds['index']))
        ds = ds.sortby('TIME')
        ds = ds.argo.cast_types()
        ds = ds[np.sort(ds.data_vars)]
        ds.encoding = self.encoding # Preserve low-level encoding information
        ds.attrs = self.attrs # Preserve original attributes
        ds.argo._add_history('Transformed with profile2point')
        ds.argo._type = 'point'
        return ds