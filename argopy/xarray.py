#!/bin/env python
# -*coding: UTF-8 -*-
#

import sys
import numpy as np
import pandas as pd
import xarray as xr

try:
    import gsw
    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False


from argopy.utilities import linear_interpolation_remap

from argopy.errors import InvalidDatasetStructure
from sklearn import preprocessing


@xr.register_dataset_accessor('argo')
class ArgoAccessor:
    """

        Class registered under scope ``argo`` to access a :class:`xarray.Dataset` object.

        - Ensure all variables are of the Argo required dtype with:

            ds.argo.cast_types()

        - Convert a collection of points into a collection of profiles:

            ds.argo.point2profile()

        - Convert a collection of profiles to a collection of points:

            ds.argo.profile2point()

     """

    def __init__(self, xarray_obj):
        """ Init """
        self._obj = xarray_obj
        self._added = list()  # Will record all new variables added by argo
        # Variables present in the initial dataset
        self._vars = list(xarray_obj.variables.keys())
        # Store the initial list of dimensions
        self._dims = list(xarray_obj.dims.keys())
        self.encoding = xarray_obj.encoding
        self.attrs = xarray_obj.attrs

        if 'N_PROF' in self._dims:
            self._type = 'profile'
        elif 'N_POINTS' in self._dims:
            self._type = 'point'
        else:
            raise InvalidDatasetStructure(
                "Argo dataset structure not recognised")

        if 'PRES_ADJUSTED' in self._vars:
            self._mode = 'expert'
        elif 'PRES' in self._vars:
            self._mode = 'standard'
        else:
            raise InvalidDatasetStructure(
                "Argo dataset structure not recognised")

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
        list_int = ['PLATFORM_NUMBER', 'WMO_INST_TYPE',
                    'WMO_INST_TYPE', 'CYCLE_NUMBER', 'CONFIG_MISSION_NUMBER']
        list_datetime = ['REFERENCE_DATE_TIME', 'DATE_CREATION', 'DATE_UPDATE',
                         'JULD', 'JULD_LOCATION', 'SCIENTIFIC_CALIB_DATE', 'HISTORY_DATE']

        def cast_this(da, type):
            """ Low-level casting of DataArray values """
            try:
                da.values = da.values.astype(type)
                da.attrs['casted'] = 1
            except Exception:
                print("Oops!", sys.exc_info()[0], "occured.")
                print("Fail to cast: ", da.dtype,
                      "into:", type, "for: ", da.name)
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
                            # This should not happen, but still ! That's real world data
                            val[val == '              '] = 'nan'
                            da.values = pd.to_datetime(
                                val, format='%Y%m%d%H%M%S')
                        else:
                            s = da.stack(dummy_index=da.dims)
                            val = s.astype(str).values.astype('U14')
                            # This should not happen, but still ! That's real world data
                            val[val == '              '] = 'nan'
                            s.values = pd.to_datetime(
                                val, format='%Y%m%d%H%M%S')
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

                    # Get back to regular U1 string
                    da = cast_this(da, np.dtype('U1'))

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
            except Exception:
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
            raise InvalidDatasetStructure(
                "Method only available to a collection of points")

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
            ii = ds.where(
                np.isnan(ds[vname + '_ADJUSTED']), drop=1)['N_POINTS']
            ds[vname + '_ADJUSTED'].loc[dict(N_POINTS=ii)
                                        ] = ds[vname].loc[dict(N_POINTS=ii)]
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
                raise InvalidDatasetStructure(
                    "Method only available for dataset with a 'DATA_MODE' variable ")
            else:
                # todo should raise a warning instead ?
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
        other_variables = list(set([v for v in list(
            ds.data_vars) if 'ADJUSTED' not in v]) - set(list(final.data_vars)))
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
            raise InvalidDatasetStructure(
                "Method only available to a collection of points")

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
        this_mask = xr.DataArray(np.zeros_like(QC_fields['N_POINTS']), dims=['N_POINTS'],
                                 coords={'N_POINTS': QC_fields['N_POINTS']})
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

        Returns
        -------
        int or tuple of int

        Examples
        --------
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
            raise InvalidDatasetStructure(
                "Method only available to a collection of points")
        this = self._obj  # Should not be modified

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
                                      dims='N_POINTS',
                                      coords={'N_POINTS': this['N_POINTS']},
                                      name='dummy_argo_uid')
        N_PROF = len(np.unique(dummy_argo_uid))
        # that = this.groupby(dummy_argo_uid)

        N_LEVELS = int(xr.DataArray(np.ones_like(this['N_POINTS'].values),
                                    dims='N_POINTS',
                                    coords={'N_POINTS': this['N_POINTS']})
                       .groupby(dummy_argo_uid).sum().max().values)
        assert N_PROF * N_LEVELS >= len(this['N_POINTS'])

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
        list_1d = list(np.array(this.data_vars)[
                       count.sum(axis=0) == count.shape[0]])
        # Variables with more than 1 value for each profiles:
        list_2d = list(np.array(this.data_vars)[
                       count.sum(axis=0) != count.shape[0]])

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
                # ['N_PROF', 'N_LEVELS'] array:
                if len(new_ds[vname].dims) == 2:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    try:
                        y[i_prof, 0:len(x)] = x
                    except Exception:
                        print(vname, 'input', x.shape,
                              'output', y[i_prof, :].shape)
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
        new_ds.encoding = self.encoding  # Preserve low-level encoding information
        new_ds.attrs = self.attrs  # Preserve original attributes
        new_ds.argo._add_history('Transformed with point2profile')
        new_ds.argo._type = 'profile'
        return new_ds

    def profile2point(self):
        """ Convert a collection of profiles to a collection of points """
        if self._type != 'profile':
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles (N_PROF dimemsion)")
        ds = self._obj

        # Remove all variables for which a dimension is length=0 (eg: N_HISTORY)
        dim_list = []
        for v in ds.data_vars:
            dims = ds[v].dims
            for d in dims:
                if len(ds[d]) == 0:
                    dim_list.append(d)
                    break
        # Drop dimensions and associated variables from this dataset
        ds = ds.drop_dims(np.unique(dim_list))

        # Remove any variable that is not with dimensions (N_PROF,) or (N_PROF, N_LEVELS)
        for v in ds:
            dims = list(ds[v].dims)
            dims = ".".join(dims)
            if dims not in ['N_PROF', 'N_PROF.N_LEVELS']:
                ds = ds.drop_vars(v)

        ds, = xr.broadcast(ds)
        ds = ds.stack({'N_POINTS': list(ds.dims)})
        ds = ds.reset_index('N_POINTS').drop_vars(['N_PROF', 'N_LEVELS'])
        possible_coords = ['LATITUDE', 'LONGITUDE', 'TIME', 'JULD', 'N_POINTS']
        for c in [c for c in possible_coords if c in ds.data_vars]:
            ds = ds.set_coords(c)

        # Remove index without data (useless points)
        ds = ds.where(~np.isnan(ds['PRES']), drop=1)
        ds = ds.sortby('TIME')
        ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))
        ds = ds.argo.cast_types()
        ds = ds[np.sort(ds.data_vars)]
        ds.encoding = self.encoding  # Preserve low-level encoding information
        ds.attrs = self.attrs  # Preserve original attributes
        ds.argo._add_history('Transformed with profile2point')
        ds.argo._type = 'point'
        return ds

    def interp_std_levels(self, std_lev):
        """ Returns a new dataset interpolated to new inputs levels                 
        
        Parameters
        ----------
        list or np.array 
            Standard levels used for interpolation

        Returns
        -------
        :class:`xarray.Dataset`           
        """
        if(self._mode != 'standard'):
           raise InvalidDatasetStructure(
               "Method only available for the standard mode yet")

        if (type(std_lev) is np.ndarray) | (type(std_lev) is list):
            std_lev = np.array(std_lev)
            if (np.any(sorted(std_lev) != std_lev)) | (np.any(std_lev < 0)):
                raise ValueError(
                    'Standard levels must be a list or a numpy array of positive and sorted values')
        else:
            raise ValueError('Standard levels must be a list or a numpy array of positive and sorted values')

        if self._type != 'profile':
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles")

        ds = self._obj

        # Selecting profiles that have a max(pressure) > max(std_lev) to avoid extrapolation in that direction
        # For levels < min(pressure), first level values of the profile are extended to surface.     
        i1 = (ds['PRES'].max('N_LEVELS') >= std_lev[-1])
        dsp = ds.where(i1, drop=True)

        # check if any profile is left, ie if any profile match the requested depth
        if (len(dsp['N_PROF']) == 0):
            raise Warning('None of the profiles can be interpolated (not reaching the requested depth range).')
            return None
            
        # add new vertical dimensions, this has to be in the datasets to apply ufunc later
        dsp['Z_LEVELS'] = xr.DataArray(std_lev, dims={'Z_LEVELS': std_lev})

        # init
        ds_out = xr.Dataset()

        # vars to interpolate
        datavars = [dv for dv in list(dsp.variables) if set(['N_LEVELS', 'N_PROF']) == set(
            dsp[dv].dims) and 'QC' not in dv]
        # coords
        coords = [dv for dv in list(dsp.coords)]
        # vars depending on N_PROF only
        solovars = [dv for dv in list(
            dsp.variables) if dv not in datavars and dv not in coords and 'QC' not in dv]

        for dv in datavars:
            ds_out[dv] = linear_interpolation_remap(
                dsp.PRES, dsp[dv], dsp['Z_LEVELS'], z_dim='N_LEVELS', z_regridded_dim='Z_LEVELS')
        ds_out = ds_out.rename({'remapped': 'PRES_INTERPOLATED'})

        for sv in solovars:
            ds_out[sv] = dsp[sv]

        for co in coords:
            ds_out.coords[co] = dsp[co]

        ds_out = ds_out.drop_vars(['N_LEVELS', 'Z_LEVELS'])
        ds_out = ds_out[np.sort(ds_out.data_vars)]
        ds_out.attrs = self.attrs # Preserve original attributes
        ds_out.argo._add_history('Interpolated on standard levels')
        
        return ds_out

    def teos10(self, vlist: list = ['SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP'], inplace: bool = True):
        """ Add TEOS10 variables to the dataset

        By default, add: 'SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP'
        Rely on the gsw library.

        Parameters
        ----------
        vlist: list(str)
            List with the name of variables to add.
        inplace: boolean, True by default
            If True, return the input :class:`xarray.Dataset` with new TEOS10 variables added as a new :class:`xarray.DataArray`
            If False, return a :class:`xarray.Dataset` with new TEOS10 variables

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if not with_gsw:
            raise ModuleNotFoundError("This functionality requires the gsw library")

        this = self._obj

        to_profile = False
        if self._type == 'profile':
            to_profile = True
            this = this.argo.profile2point()

        # Get base variables as numpy arrays:
        psal = this['PSAL'].values
        temp = this['TEMP'].values
        pres = this['PRES'].values
        lon = this['LONGITUDE'].values
        lat = this['LATITUDE'].values
        f = lat

        # Coriolis
        f = gsw.f(lat)

        # Depth:
        depth = gsw.z_from_p(pres, lat)

        # Absolute salinity
        sa = gsw.SA_from_SP(psal, pres, lon, lat)

        # Conservative temperature
        ct = gsw.CT_from_t(sa, temp, depth)

        # Potential Temperature
        if 'PTEMP' in vlist:
            pt = gsw.pt_from_CT(sa, ct)

        # Potential density referenced to surface
        if 'SIG0' in vlist:
            sig0 = gsw.sigma0(sa, ct)

        # N2
        if 'N2' in vlist or 'PV' in vlist:
            n2_mid, p_mid = gsw.Nsquared(sa, ct, pres, lat)
            # N2 on the CT grid:
            ishallow = (slice(0, -1), Ellipsis)
            ideep = (slice(1, None), Ellipsis)

            def mid(x):
                return 0.5 * (x[ideep] + x[ishallow])

            n2 = np.zeros(ct.shape) * np.nan
            n2[1:-1] = mid(n2_mid)

        # PV:
        if 'PV' in vlist:
            pv = f * n2 / gsw.grav(lat, pres)

        # Back to the dataset:    
        that = []
        if 'SA' in vlist:
            SA = xr.DataArray(sa, coords=this['PSAL'].coords, name='SA')
            SA.attrs['standard_name'] = 'Absolute Salinity'
            SA.attrs['unit'] = 'g/kg'
            that.append(SA)

        if 'CT' in vlist:
            CT = xr.DataArray(ct, coords=this['TEMP'].coords, name='CT')
            CT.attrs['standard_name'] = 'Conservative Temperature'
            CT.attrs['unit'] = 'degC'
            that.append(CT)

        if 'SIG0' in vlist:
            SIG0 = xr.DataArray(sig0, coords=this['TEMP'].coords, name='SIG0')
            SIG0.attrs['long_name'] = 'Potential density anomaly with reference pressure of 0 dbar'
            SIG0.attrs['standard_name'] = 'Potential Density'
            SIG0.attrs['unit'] = 'kg/m^3'
            that.append(SIG0)

        if 'N2' in vlist:
            N2 = xr.DataArray(n2, coords=this['TEMP'].coords, name='N2')
            N2.attrs['standard_name'] = 'Squared buoyancy frequency'
            N2.attrs['unit'] = '1/s^2'
            that.append(N2)

        if 'PV' in vlist:
            PV = xr.DataArray(pv, coords=this['TEMP'].coords, name='PV')
            PV.attrs['standard_name'] = 'Planetary Potential Vorticity'
            PV.attrs['unit'] = '1/m/s'
            that.append(PV)

        if 'PTEMP' in vlist:
            PTEMP = xr.DataArray(pt, coords=this['TEMP'].coords, name='PTEMP')
            PTEMP.attrs['standard_name'] = 'Potential Temperature'
            PTEMP.attrs['unit'] = 'degC'
            that.append(PTEMP)

        # Create a dataset with all new variables:
        that = xr.merge(that)
        # Add to the dataset essential Argo variables (allows to keep using the argo accessor):
        that = that.assign({k:this[k] for k in ['TIME', ' LATITUDE', 'LONGITUDE', 'PRES', 'PRES_ADJUSTED',
                                                'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION'] if k in this})
        # Manage output:
        if inplace:
            # Merge previous with new variables
            for v in that.variables:
                this[v] = that[v]
            if to_profile:
                this = this.argo.point2profile()
            for k in this:
                if k not in self._obj:
                    self._obj[k] = this[k]
            return self._obj
        else:
            if to_profile:
                return that.argo.point2profile()
            else:
                return that
