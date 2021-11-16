import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
import logging

try:
    import gsw

    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False

from argopy.utilities import linear_interpolation_remap, \
    is_list_equal, is_list_of_strings, toYearFraction, wrap_longitude
from argopy.errors import InvalidDatasetStructure, DataNotFound, OptionValueError


log = logging.getLogger("argopy.xarray")


@xr.register_dataset_accessor("argo")
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
        # self._register = collections.OrderedDict() # Will register mutable instances of sub-modules like 'plot'
        # Variables present in the initial dataset
        self._vars = list(xarray_obj.variables.keys())
        # Store the initial list of dimensions
        self._dims = list(xarray_obj.dims.keys())
        self.encoding = xarray_obj.encoding
        self.attrs = xarray_obj.attrs

        if "N_PROF" in self._dims:
            self._type = "profile"
        elif "N_POINTS" in self._dims:
            self._type = "point"
        else:
            raise InvalidDatasetStructure("Argo dataset structure not recognised")

        if "PRES_ADJUSTED" in self._vars:
            self._mode = "expert"
        elif "PRES" in self._vars:
            self._mode = "standard"
        else:
            raise InvalidDatasetStructure("Argo dataset structure not recognised")

    def __repr__(self):
        import xarray.core.formatting as xrf

        summary = ["<xarray.{}.argo>".format(type(self._obj).__name__)]
        if self._type == "profile":
            summary.append("This is a collection of Argo profiles")
        elif self._type == "point":
            summary.append("This is a collection of Argo points")

        col_width = xrf._calculate_col_width(xrf._get_col_items(self._obj.variables))
        # max_rows = xr.core.options.OPTIONS["display_max_rows"]

        dims_start = xrf.pretty_print("Dimensions:", col_width)
        summary.append("{}({})".format(dims_start, xrf.dim_summary(self._obj)))

        return "\n".join(summary)

    @property
    def N_PROF(self):
        """Number of profiles"""
        if self._type == "point":
            dummy_argo_uid = xr.DataArray(
                self.uid(
                    self._obj["PLATFORM_NUMBER"].values,
                    self._obj["CYCLE_NUMBER"].values,
                    self._obj["DIRECTION"].values,
                ),
                dims="N_POINTS",
                coords={"N_POINTS": self._obj["N_POINTS"]},
                name="dummy_argo_uid",
            )
            N_PROF = len(np.unique(dummy_argo_uid))
        else:
            N_PROF = len(np.unique(self._obj['N_PROF']))
        return N_PROF

    @property
    def N_LEVELS(self):
        """Number of vertical levels"""
        if self._type == "point":
            dummy_argo_uid = xr.DataArray(
                self.uid(
                    self._obj["PLATFORM_NUMBER"].values,
                    self._obj["CYCLE_NUMBER"].values,
                    self._obj["DIRECTION"].values,
                ),
                dims="N_POINTS",
                coords={"N_POINTS": self._obj["N_POINTS"]},
                name="dummy_argo_uid",
            )
            N_LEVELS = int(
                xr.DataArray(
                    np.ones_like(self._obj["N_POINTS"].values),
                    dims="N_POINTS",
                    coords={"N_POINTS": self._obj["N_POINTS"]},
                )
                    .groupby(dummy_argo_uid)
                    .sum()
                    .max()
                    .values
            )
        else:
            N_LEVELS = len(np.unique(self._obj['N_LEVELS']))
        return N_LEVELS

    @property
    def N_POINTS(self):
        """Number of measurement points"""
        if self._type == "profile":
            N_POINTS = self.N_PROF*self.N_LEVELS
        else:
            N_POINTS = len(np.unique(self._obj['N_POINTS']))
        return N_POINTS

    def _add_history(self, txt):
        if "history" in self._obj.attrs:
            self._obj.attrs["history"] += "; %s" % txt
        else:
            self._obj.attrs["history"] = txt

    def _where(self, cond, other=xr.core.dtypes.NA, drop: bool = False):
        """ where that preserve dtypes of Argo fields

        Parameters
        ----------
        cond : DataArray, Dataset, or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, it must expect this object as its only parameter.
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, these locations filled with NA.
        drop : bool, optional
            If True, coordinate labels that only correspond to False values of
            the condition are dropped from the result. Mutually exclusive with
            ``other``.
        """
        self._obj = self._obj.where(cond, other=other, drop=drop)
        return self.cast_types()

    def cast_types(self):  # noqa: C901
        """ Make sure variables are of the appropriate types

            This is hard coded, but should be retrieved from an API somewhere
            Should be able to handle all possible variables encountered in the Argo dataset
        """
        ds = self._obj

        list_str = [
            "PLATFORM_NUMBER",
            "DATA_MODE",
            "DIRECTION",
            "DATA_CENTRE",
            "DATA_TYPE",
            "FORMAT_VERSION",
            "HANDBOOK_VERSION",
            "PROJECT_NAME",
            "PI_NAME",
            "STATION_PARAMETERS",
            "DATA_CENTER",
            "DC_REFERENCE",
            "DATA_STATE_INDICATOR",
            "PLATFORM_TYPE",
            "FIRMWARE_VERSION",
            "POSITIONING_SYSTEM",
            "PROFILE_PRES_QC",
            "PROFILE_PSAL_QC",
            "PROFILE_TEMP_QC",
            "PARAMETER",
            "SCIENTIFIC_CALIB_EQUATION",
            "SCIENTIFIC_CALIB_COEFFICIENT",
            "SCIENTIFIC_CALIB_COMMENT",
            "HISTORY_INSTITUTION",
            "HISTORY_STEP",
            "HISTORY_SOFTWARE",
            "HISTORY_SOFTWARE_RELEASE",
            "HISTORY_REFERENCE",
            "HISTORY_QCTEST",
            "HISTORY_ACTION",
            "HISTORY_PARAMETER",
            "VERTICAL_SAMPLING_SCHEME",
            "FLOAT_SERIAL_NO",
        ]
        list_int = [
            "PLATFORM_NUMBER",
            "WMO_INST_TYPE",
            "WMO_INST_TYPE",
            "CYCLE_NUMBER",
            "CONFIG_MISSION_NUMBER",
        ]
        list_datetime = [
            "REFERENCE_DATE_TIME",
            "DATE_CREATION",
            "DATE_UPDATE",
            "JULD",
            "JULD_LOCATION",
            "SCIENTIFIC_CALIB_DATE",
            "HISTORY_DATE",
        ]

        def cast_this(da, type):
            """ Low-level casting of DataArray values """
            try:
                da.values = da.values.astype(type)
                da.attrs["casted"] = 1
            except Exception:
                print("Oops!", sys.exc_info()[0], "occurred.")
                print("Fail to cast: ", da.dtype,
                      "into:", type, "for: ", da.name)
                print("Encountered unique values:", np.unique(da))
            return da

        def cast_this_da(da):
            """ Cast any DataArray """
            da.attrs["casted"] = 0
            if v in list_str and da.dtype == "O":  # Object
                da = cast_this(da, str)

            if v in list_int:  # and da.dtype == 'O':  # Object
                da = cast_this(da, int)

            if v in list_datetime and da.dtype == "O":  # Object
                if (
                    "conventions" in da.attrs
                    and da.attrs["conventions"] == "YYYYMMDDHHMISS"
                ):
                    if da.size != 0:
                        if len(da.dims) <= 1:
                            val = da.astype(str).values.astype("U14")
                            # This should not happen, but still ! That's real world data
                            val[val == "              "] = "nan"
                            da.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                        else:
                            s = da.stack(dummy_index=da.dims)
                            val = s.astype(str).values.astype("U14")
                            # This should not happen, but still ! That's real world data
                            val[val == "              "] = "nan"
                            s.values = pd.to_datetime(val, format="%Y%m%d%H%M%S")
                            da.values = s.unstack("dummy_index")
                        da = cast_this(da, np.datetime64)
                    else:
                        da = cast_this(da, np.datetime64)

                elif v == "SCIENTIFIC_CALIB_DATE":
                    da = cast_this(da, str)
                    s = da.stack(dummy_index=da.dims)
                    s.values = pd.to_datetime(s.values, format="%Y%m%d%H%M%S")
                    da.values = (s.unstack("dummy_index")).values
                    da = cast_this(da, np.datetime64)

            if "QC" in v and "PROFILE" not in v and "QCTEST" not in v:
                if da.dtype == "O":  # convert object to string
                    da = cast_this(da, str)

                # Address weird string values:
                # (replace missing or nan values by a '0' that will be cast as an integer later

                if da.dtype == "<U3":  # string, len 3 because of a 'nan' somewhere
                    ii = (
                        da == "   "
                    )  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, "0", da)

                    ii = (
                        da == "nan"
                    )  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, "0", da)

                    # Get back to regular U1 string
                    da = cast_this(da, np.dtype("U1"))

                if da.dtype == "<U1":  # string
                    ii = (
                        da == ""
                    )  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, "0", da)

                    ii = (
                        da == " "
                    )  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, "0", da)

                    ii = (
                        da == "n"
                    )  # This should not happen, but still ! That's real world data
                    da = xr.where(ii, "0", da)

                # finally convert QC strings to integers:
                da = cast_this(da, int)

            if da.dtype != "O":
                da.attrs["casted"] = 1

            return da

        for v in ds.data_vars:
            try:
                ds[v] = cast_this_da(ds[v])
            except Exception:
                print("Oops!", sys.exc_info()[0], "occurred.")
                print("Fail to cast: %s " % v)
                print("Encountered unique values:", np.unique(ds[v]))
                raise

        return ds

    def filter_data_mode(self, keep_error: bool = True, errors: str = "raise"):  # noqa: C901
        """ Filter variables according to their data mode

        This filter applies to <PARAM> and <PARAM_QC>

        For data mode 'R' and 'A': keep <PARAM> (eg: 'PRES', 'TEMP' and 'PSAL')
        For data mode 'D': keep <PARAM_ADJUSTED> (eg: 'PRES_ADJUSTED', 'TEMP_ADJUSTED' and 'PSAL_ADJUSTED')

        Since ADJUSTED variables are not required anymore after the filter, all *ADJUSTED* variables are dropped in
        order to avoid confusion wrt variable content. DATA_MODE is preserved for the record.

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
        if self._type != "point":
            raise InvalidDatasetStructure(
                "Method only available to a collection of points"
            )

        #########
        # Sub-functions
        #########
        def safe_where_eq(xds, key, value):
            # xds.where(xds[key] == value, drop=True) is not safe to empty time variables, cf issue #64
            try:
                return xds.where(xds[key] == value, drop=True)
            except ValueError as v:
                if v.args[0] == ("zero-size array to reduction operation "
                                 "minimum which has no identity"):
                    # A bug in xarray will cause a ValueError if trying to
                    # decode the times in a NetCDF file with length 0.
                    # See:
                    # https://github.com/pydata/xarray/issues/1329
                    # https://github.com/euroargodev/argopy/issues/64
                    # Here, we just need to return an empty array
                    TIME = xds['TIME']
                    xds = xds.drop_vars('TIME')
                    xds = xds.where(xds[key] == value, drop=True)
                    xds['TIME'] = xr.DataArray(np.arange(len(xds['N_POINTS'])), dims='N_POINTS',
                                               attrs=TIME.attrs).astype(np.datetime64)
                    xds = xds.set_coords('TIME')
                    return xds

        def ds_split_datamode(xds):
            """ Create one dataset for each of the data_mode

                Split full dataset into 3 datasets
            """
            # Real-time:
            argo_r = safe_where_eq(xds, 'DATA_MODE', 'R')
            for v in plist:
                vname = v.upper() + "_ADJUSTED"
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
                vname = v.upper() + "_ADJUSTED_QC"
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
                vname = v.upper() + "_ADJUSTED_ERROR"
                if vname in argo_r:
                    argo_r = argo_r.drop_vars(vname)
            # Real-time adjusted:
            argo_a = safe_where_eq(xds, 'DATA_MODE', 'A')
            for v in plist:
                vname = v.upper()
                if vname in argo_a:
                    argo_a = argo_a.drop_vars(vname)
                vname = v.upper() + "_QC"
                if vname in argo_a:
                    argo_a = argo_a.drop_vars(vname)
            # Delayed mode:
            argo_d = safe_where_eq(xds, 'DATA_MODE', 'D')

            return argo_r, argo_a, argo_d

        def fill_adjusted_nan(this_ds, vname):
            """Fill in the adjusted field with the non-adjusted wherever it is NaN

               Ensure to have values even for bad QC data in delayed mode
            """
            ii = this_ds.where(np.isnan(this_ds[vname + "_ADJUSTED"]), drop=1)["N_POINTS"]
            this_ds[vname + "_ADJUSTED"].loc[dict(N_POINTS=ii)] = this_ds[vname].loc[
                dict(N_POINTS=ii)
            ]
            return this_ds

        def merge_arrays(this_argo_r, this_argo_a, this_argo_d, this_vname):
            """ Merge one variable from 3 DataArrays

                Based on xarray merge function with ’no_conflicts’: only values
                which are not null in all datasets must be equal. The returned
                dataset then contains the combination of all non-null values.

                Return a xarray.DataArray
            """

            def merge_this(a1, a2, a3):
                return xr.merge((xr.merge((a1, a2)), a3))

            DA = merge_this(
                this_argo_r[this_vname],
                this_argo_a[this_vname + "_ADJUSTED"].rename(this_vname),
                this_argo_d[this_vname + "_ADJUSTED"].rename(this_vname),
            )
            DA_QC = merge_this(
                this_argo_r[this_vname + "_QC"],
                this_argo_a[this_vname + "_ADJUSTED_QC"].rename(this_vname + "_QC"),
                this_argo_d[this_vname + "_ADJUSTED_QC"].rename(this_vname + "_QC"),
            )

            if keep_error:
                DA_ERROR = xr.merge((
                    this_argo_a[this_vname + "_ADJUSTED_ERROR"].rename(this_vname + "_ERROR"),
                    this_argo_d[this_vname + "_ADJUSTED_ERROR"].rename(this_vname + "_ERROR"),
                ))
                DA = merge_this(DA, DA_QC, DA_ERROR)
            else:
                DA = xr.merge((DA, DA_QC))
            return DA

        #########
        # filter
        #########
        ds = self._obj
        if "DATA_MODE" not in ds:
            if errors:
                raise InvalidDatasetStructure(
                    "Method only available for dataset with a 'DATA_MODE' variable "
                )
            else:
                # todo should raise a warning instead ?
                return ds

        # Define variables to filter:
        possible_list = [
            "PRES",
            "TEMP",
            "PSAL",
            "DOXY",
            "CHLA",
            "BBP532",
            "BBP700",
            "DOWNWELLING_PAR",
            "DOWN_IRRADIANCE380",
            "DOWN_IRRADIANCE412",
            "DOWN_IRRADIANCE490",
        ]
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
            vname = v.upper() + "_QC"
            if vname in argo_d:
                argo_d = argo_d.drop_vars(vname)

        # Create new arrays with the appropriate variables:
        vlist = [merge_arrays(argo_r, argo_a, argo_d, v) for v in plist]

        # Create final dataset by merging all available variables
        final = xr.merge(vlist)

        # Merge with all other variables:
        other_variables = list(
            set([v for v in list(ds.data_vars) if "ADJUSTED" not in v])
            - set(list(final.data_vars))
        )
        # other_variables.remove('DATA_MODE')  # Not necessary anymore
        for p in other_variables:
            final = xr.merge((final, ds[p]))

        final.attrs = ds.attrs
        final.argo._add_history("Variables filtered according to DATA_MODE")
        final = final[np.sort(final.data_vars)]

        # Cast data types and add attributes:
        final = final.argo.cast_types()

        return final

    def filter_qc(self, QC_list=[1, 2], QC_fields='all', drop=True, mode="all", mask=False):  # noqa: C901
        """ Filter data set according to QC values

            Filter the dataset to keep points where 'all' or 'any' of the QC fields has a value in the list of
            integer QC flags.

            This method can return the filtered dataset or the filter mask.

        Parameters
        ----------
        QC_list: list(int)
            List of QC flag values (integers) to keep
        QC_fields: 'all' or list(str)
            List of QC fields to consider to apply the filter. By default we use all available QC fields
        drop: bool
            Drop values not matching the QC filter, default is True
        mode: str
            Must be 'all' (default) or 'any'. Boolean operator on QC values: should we keep points
            matching 'all' QC fields or 'any' one of them.
        mask: bool
            'False' by default. Determine if we should return the QC mask or the filtered dataset.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if self._type != "point":
            raise InvalidDatasetStructure(
                "Method only available to a collection of points"
            )

        if mode not in ["all", "any"]:
            raise ValueError("Mode must be 'all' or 'any'")

        # Make sure we deal with a list of integers:
        if not isinstance(QC_list, list):
            if isinstance(QC_list, np.ndarray):
                QC_list = list(QC_list)
            else:
                QC_list = [QC_list]
        QC_list = [abs(int(qc)) for qc in QC_list]
        log.debug("filter_qc: Filtering dataset to keep points with QC in %s" % QC_list)

        this = self._obj

        # Extract QC fields:
        if isinstance(QC_fields, str) and QC_fields == 'all':
            QC_fields = []
            for v in this.data_vars:
                if "QC" in v and "PROFILE" not in v:
                    QC_fields.append(v)
        elif is_list_of_strings(QC_fields):
            for v in QC_fields:
                if v not in this.data_vars:
                    raise ValueError("%s not found in this dataset while trying to apply QC filter" % v)
        else:
            raise ValueError("Invalid content for parameter 'QC_fields'. Use 'all' or a list of strings")
        log.debug("filter_qc: Filter applied to '%s' of the fields: %s" % (mode, ",".join(QC_fields)))

        QC_fields = this[QC_fields]
        for v in QC_fields.data_vars:
            QC_fields[v] = QC_fields[v].astype(int)

        # Now apply filter
        this_mask = xr.DataArray(
            np.zeros_like(QC_fields["N_POINTS"]),
            dims=["N_POINTS"],
            coords={"N_POINTS": QC_fields["N_POINTS"]},
        )
        for v in QC_fields.data_vars:
            for qc_value in QC_list:
                this_mask += QC_fields[v] == qc_value
        if mode == "all":
            this_mask = this_mask == len(QC_fields)  # all
        else:
            this_mask = this_mask >= 1  # any

        if not mask:
            this = this.argo._where(this_mask, drop=drop)
            this.argo._add_history("Variables selected according to QC")
            # this = this.argo.cast_types()
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
        le.fit(["A", "D"])

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
                return (
                    encode_direction(direction)
                    * np.vectorize(int)(offset * wmo_or_uid + cyc).ravel()
                )
            else:
                return np.vectorize(int)(offset * wmo_or_uid + cyc).ravel()
        else:
            # DECODER
            drc = decode_direction(np.sign(wmo_or_uid))
            wmo = np.vectorize(int)(np.abs(wmo_or_uid) / offset)
            cyc = -np.vectorize(int)(offset * wmo - np.abs(wmo_or_uid))
            return wmo, cyc, drc

    def point2profile(self):  # noqa: C901
        """ Transform a collection of points into a collection of profiles

        """
        if self._type != "point":
            raise InvalidDatasetStructure(
                "Method only available to a collection of points"
            )
        this = self._obj  # Should not be modified

        def fillvalue(da):
            """ Return fillvalue for a dataarray """
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
            if da.dtype.kind in ["U"]:
                fillvalue = " "
            elif da.dtype.kind == "i":
                fillvalue = 99999
            elif da.dtype.kind == "M":
                fillvalue = np.datetime64("NaT")
            else:
                fillvalue = np.nan
            return fillvalue

        # Find the number of profiles (N_PROF) and vertical levels (N_LEVELS):
        dummy_argo_uid = xr.DataArray(
            self.uid(
                this["PLATFORM_NUMBER"].values,
                this["CYCLE_NUMBER"].values,
                this["DIRECTION"].values,
            ),
            dims="N_POINTS",
            coords={"N_POINTS": this["N_POINTS"]},
            name="dummy_argo_uid",
        )
        N_PROF = len(np.unique(dummy_argo_uid))

        N_LEVELS = int(
            xr.DataArray(
                np.ones_like(this["N_POINTS"].values),
                dims="N_POINTS",
                coords={"N_POINTS": this["N_POINTS"]},
            )
            .groupby(dummy_argo_uid)
            .sum()
            .max()
            .values
        )
        assert N_PROF * N_LEVELS >= len(this["N_POINTS"])

        # Store the initial set of coordinates:
        coords_list = list(this.coords)
        this = this.reset_coords()

        # For each variables, determine if it has unique value by profile,
        # if yes: the transformed variable should be [N_PROF]
        # if no: the transformed variable should be [N_PROF, N_LEVELS]
        count = np.zeros((N_PROF, len(this.data_vars)), "int")
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
            new_ds.append(
                xr.DataArray(
                    np.full(
                        (N_PROF, N_LEVELS),
                        fillvalue(this[vname]),
                        dtype=this[vname].dtype,
                    ),
                    dims=["N_PROF", "N_LEVELS"],
                    coords={
                        "N_PROF": np.arange(N_PROF),
                        "N_LEVELS": np.arange(N_LEVELS),
                    },
                    attrs=this[vname].attrs,
                    name=vname,
                )
            )
        for vname in list_1d:
            new_ds.append(
                xr.DataArray(
                    np.full((N_PROF,), fillvalue(this[vname]), dtype=this[vname].dtype),
                    dims=["N_PROF"],
                    coords={"N_PROF": np.arange(N_PROF)},
                    attrs=this[vname].attrs,
                    name=vname,
                )
            )
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
                        y[i_prof, 0: len(x)] = x
                    except Exception:
                        print(vname, "input", x.shape, "output", y[i_prof, :].shape)
                        raise
                    new_ds[vname].values = y
                else:  # ['N_PROF', ] array:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    y[i_prof] = np.unique(x)[0]

        # Restore coordinate variables:
        new_ds = new_ds.set_coords([c for c in coords_list if c in new_ds])

        # Misc formatting
        new_ds = new_ds.sortby("TIME")
        new_ds = new_ds.argo.cast_types()
        new_ds = new_ds[np.sort(new_ds.data_vars)]
        new_ds.encoding = self.encoding  # Preserve low-level encoding information
        new_ds.attrs = self.attrs  # Preserve original attributes
        new_ds.argo._add_history("Transformed with point2profile")
        new_ds.argo._type = "profile"
        return new_ds

    def profile2point(self):
        """ Convert a collection of profiles to a collection of points """
        if self._type != "profile":
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles (N_PROF dimemsion)"
            )
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
            if dims not in ["N_PROF", "N_PROF.N_LEVELS"]:
                ds = ds.drop_vars(v)

        (ds,) = xr.broadcast(ds)
        ds = ds.stack({"N_POINTS": list(ds.dims)})
        ds = ds.reset_index("N_POINTS").drop_vars(["N_PROF", "N_LEVELS"])
        possible_coords = ["LATITUDE", "LONGITUDE", "TIME", "JULD", "N_POINTS"]
        for c in [c for c in possible_coords if c in ds.data_vars]:
            ds = ds.set_coords(c)

        # Remove index without data (useless points)
        ds = ds.where(~np.isnan(ds["PRES"]), drop=1)
        ds = ds.sortby("TIME")
        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        ds = ds.argo.cast_types()
        ds = ds[np.sort(ds.data_vars)]
        ds.encoding = self.encoding  # Preserve low-level encoding information
        ds.attrs = self.attrs  # Preserve original attributes
        ds.argo._add_history("Transformed with profile2point")
        ds.argo._type = "point"
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

        if (type(std_lev) is np.ndarray) | (type(std_lev) is list):
            std_lev = np.array(std_lev)
            if (np.any(sorted(std_lev) != std_lev)) | (np.any(std_lev < 0)):
                raise ValueError(
                    "Standard levels must be a list or a numpy array of positive and sorted values"
                )
        else:
            raise ValueError(
                "Standard levels must be a list or a numpy array of positive and sorted values"
            )

        if self._type != "profile":
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles"
            )

        ds = self._obj

        # Selecting profiles that have a max(pressure) > max(std_lev) to avoid extrapolation in that direction
        # For levels < min(pressure), first level values of the profile are extended to surface.
        i1 = ds["PRES"].max("N_LEVELS") >= std_lev[-1]
        dsp = ds.where(i1, drop=True)

        # check if any profile is left, ie if any profile match the requested depth
        if len(dsp["N_PROF"]) == 0:
            raise Warning(
                "None of the profiles can be interpolated (not reaching the requested depth range)."
            )
            return None

        # add new vertical dimensions, this has to be in the datasets to apply ufunc later
        dsp["Z_LEVELS"] = xr.DataArray(std_lev, dims={"Z_LEVELS": std_lev})

        # init
        ds_out = xr.Dataset()

        # vars to interpolate
        datavars = [
            dv
            for dv in list(dsp.variables)
            if set(["N_LEVELS", "N_PROF"]) == set(dsp[dv].dims)
            and "QC" not in dv
            and "ERROR" not in dv
        ]
        # coords
        coords = [dv for dv in list(dsp.coords)]
        # vars depending on N_PROF only
        solovars = [
            dv
            for dv in list(dsp.variables)
            if dv not in datavars
            and dv not in coords
            and "QC" not in dv
            and "ERROR" not in dv
        ]

        for dv in datavars:
            ds_out[dv] = linear_interpolation_remap(
                dsp.PRES,
                dsp[dv],
                dsp["Z_LEVELS"],
                z_dim="N_LEVELS",
                z_regridded_dim="Z_LEVELS",
            )
        ds_out = ds_out.rename({"remapped": "PRES_INTERPOLATED"})

        for sv in solovars:
            ds_out[sv] = dsp[sv]

        for co in coords:
            ds_out.coords[co] = dsp[co]

        ds_out = ds_out.drop_vars(["N_LEVELS", "Z_LEVELS"])
        ds_out = ds_out[np.sort(ds_out.data_vars)]
        ds_out = ds_out.argo.cast_types()
        ds_out.attrs = self.attrs  # Preserve original attributes
        ds_out.argo._add_history("Interpolated on standard levels")

        return ds_out

    def teos10(  # noqa: C901
        self,
        vlist: list = ["SA", "CT", "SIG0", "N2", "PV", "PTEMP"],
        inplace: bool = True,
    ):
        """ Add TEOS10 variables to the dataset

        By default, adds: 'SA', 'CT'
        Other possible variables: 'SIG0', 'N2', 'PV', 'PTEMP', 'SOUND_SPEED'
        Relies on the gsw library.

        If one exists, the correct CF standard name will be added to the attrs.

        Parameters
        ----------
        vlist: list(str)
            List with the name of variables to add.
            Must be a list containing one or more of the following string values:

            * `"SA"`
                Adds an absolute salinity variable
            * `"CT"`
                Adds a conservative temperature variable
            * `"SIG0"`
                Adds a potential density anomaly variable referenced to 0 dbar
            * `"N2"`
                Adds a buoyancy (Brunt-Vaisala) frequency squared variable.
                This variable has been regridded to the original pressure levels in the Dataset using a linear interpolation.
            * `"PV"`
                Adds a planetary vorticity variable calculated from :math:`\\frac{f N^2}{\\text{gravity}}`.
                This is not a TEOS-10 variable from the gsw toolbox, but is provided for convenience.
                This variable has been regridded to the original pressure levels in the Dataset using a linear interpolation.
            * `"PTEMP"`
                Adds a potential temperature variable
            * `"SOUND_SPEED"`
                Adds a sound speed variable


        inplace: boolean, True by default
            If True, return the input :class:`xarray.Dataset` with new TEOS10 variables
                added as a new :class:`xarray.DataArray`.
            If False, return a :class:`xarray.Dataset` with new TEOS10 variables

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if not with_gsw:
            raise ModuleNotFoundError("This functionality requires the gsw library")

        allowed = ['SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP', 'SOUND_SPEED', 'CNDC']
        if any(var not in allowed for var in vlist):
            raise ValueError(f"vlist must be a subset of {allowed}, instead found {vlist}")

        if is_list_equal(vlist, ["SA", "CT", "SIG0", "N2", "PV", "PTEMP"]):
            warnings.warn("Default variables will be reduced to 'SA' and 'CT' in 0.1.9", category=FutureWarning)

        this = self._obj

        to_profile = False
        if self._type == "profile":
            to_profile = True
            this = this.argo.profile2point()

        # Get base variables as numpy arrays:
        psal = this['PSAL'].values
        temp = this['TEMP'].values
        pres = this['PRES'].values
        lon = this['LONGITUDE'].values
        lat = this['LATITUDE'].values

        # Coriolis
        f = gsw.f(lat)

        # Absolute salinity
        sa = gsw.SA_from_SP(psal, pres, lon, lat)

        # Conservative temperature
        ct = gsw.CT_from_t(sa, temp, pres)

        # Potential Temperature
        if "PTEMP" in vlist:
            pt = gsw.pt_from_CT(sa, ct)

        # Potential density referenced to surface
        if "SIG0" in vlist:
            sig0 = gsw.sigma0(sa, ct)

        # Electrical conductivity
        if "CNDC" in vlist:
            cndc = gsw.C_from_SP(psal, temp, pres)

        # N2
        if "N2" in vlist or "PV" in vlist:
            n2_mid, p_mid = gsw.Nsquared(sa, ct, pres, lat)
            # N2 on the CT grid:
            ishallow = (slice(0, -1), Ellipsis)
            ideep = (slice(1, None), Ellipsis)

            def mid(x):
                return 0.5 * (x[ideep] + x[ishallow])

            n2 = np.zeros(ct.shape) * np.nan
            n2[1:-1] = mid(n2_mid)

        # PV:
        if "PV" in vlist:
            pv = f * n2 / gsw.grav(lat, pres)

        # Sound Speed:
        if 'SOUND_SPEED' in vlist:
            cs = gsw.sound_speed(sa, ct, pres)

        # Back to the dataset:
        that = []
        if 'SA' in vlist:
            SA = xr.DataArray(sa, coords=this['PSAL'].coords, name='SA')
            SA.attrs['long_name'] = 'Absolute Salinity'
            SA.attrs['standard_name'] = 'sea_water_absolute_salinity'
            SA.attrs['unit'] = 'g/kg'
            that.append(SA)

        if 'CT' in vlist:
            CT = xr.DataArray(ct, coords=this['TEMP'].coords, name='CT')
            CT.attrs['long_name'] = 'Conservative Temperature'
            CT.attrs['standard_name'] = 'sea_water_conservative_temperature'
            CT.attrs['unit'] = 'degC'
            that.append(CT)

        if 'SIG0' in vlist:
            SIG0 = xr.DataArray(sig0, coords=this['TEMP'].coords, name='SIG0')
            SIG0.attrs['long_name'] = 'Potential density anomaly with reference pressure of 0 dbar'
            SIG0.attrs['standard_name'] = 'sea_water_sigma_theta'
            SIG0.attrs['unit'] = 'kg/m^3'
            that.append(SIG0)

        if 'CNDC' in vlist:
            CNDC = xr.DataArray(cndc, coords=this['TEMP'].coords, name='CNDC')
            CNDC.attrs['long_name'] = 'Electrical Conductivity'
            CNDC.attrs['standard_name'] = 'sea_water_electrical_conductivity'
            CNDC.attrs['unit'] = 'mS/cm'
            that.append(CNDC)

        if 'N2' in vlist:
            N2 = xr.DataArray(n2, coords=this['TEMP'].coords, name='N2')
            N2.attrs['long_name'] = 'Squared buoyancy frequency'
            N2.attrs['unit'] = '1/s^2'
            that.append(N2)

        if 'PV' in vlist:
            PV = xr.DataArray(pv, coords=this['TEMP'].coords, name='PV')
            PV.attrs['long_name'] = 'Planetary Potential Vorticity'
            PV.attrs['unit'] = '1/m/s'
            that.append(PV)

        if 'PTEMP' in vlist:
            PTEMP = xr.DataArray(pt, coords=this['TEMP'].coords, name='PTEMP')
            PTEMP.attrs['long_name'] = 'Potential Temperature'
            PTEMP.attrs['standard_name'] = 'sea_water_potential_temperature'
            PTEMP.attrs['unit'] = 'degC'
            that.append(PTEMP)

        if 'SOUND_SPEED' in vlist:
            CS = xr.DataArray(cs, coords=this['TEMP'].coords, name='SOUND_SPEED')
            CS.attrs['long_name'] = 'Speed of sound'
            CS.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
            CS.attrs['unit'] = 'm/s'
            that.append(CS)

        # Create a dataset with all new variables:
        that = xr.merge(that)
        # Add to the dataset essential Argo variables (allows to keep using the argo accessor):
        that = that.assign(
            {
                k: this[k]
                for k in [
                    "TIME",
                    " LATITUDE",
                    "LONGITUDE",
                    "PRES",
                    "PRES_ADJUSTED",
                    "PLATFORM_NUMBER",
                    "CYCLE_NUMBER",
                    "DIRECTION",
                ]
                if k in this
            }
        )
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

    def filter_scalib_pres(self, force: str = 'default', inplace: bool = True):
        """ Filter variables according to OWC salinity calibration software requirements

        By default: this filter will return a dataset with raw PRES, PSAL and TEMP; and if PRES is adjusted,
        PRES variable will be replaced by PRES_ADJUSTED.

        With option force='raw', you can force the filter to return a dataset with raw PRES, PSAL and TEMP wether
        PRES is adjusted or not.

        With option force='adjusted', you can force the filter to return a dataset where PRES/PSAL and TEMP replaced
        with adjusted variables: PRES_ADJUSTED, PSAL_ADJUSTED, TEMP_ADJUSTED.

        Since ADJUSTED variables are not required anymore after the filter, all *ADJUSTED* variables are dropped in
        order to avoid confusion wrt variable content.

        Parameters
        ----------
        force: str
            Use force='default' to load PRES/PSAL/TEMP or PRES_ADJUSTED/PSAL/TEMP according to PRES_ADJUSTED
            filled or not.
            Use force='raw' to force load of PRES/PSAL/TEMP
            Use force='adjusted' to force load of PRES_ADJUSTED/PSAL_ADJUSTED/TEMP_ADJUSTED
        inplace: boolean, True by default
            If True, return the filtered input :class:`xarray.Dataset`
            If False, return a new :class:`xarray.Dataset`

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if not with_gsw:
            raise ModuleNotFoundError("This functionality requires the gsw library")

        this = self._obj

        # Will work with a collection of points
        to_profile = False
        if this.argo._type == "profile":
            to_profile = True
            this = this.argo.profile2point()

        if force == 'raw':
            # PRES/PSAL/TEMP are not changed
            # All ADJUSTED variables are removed (not required anymore, avoid confusion with variable content):
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])
        elif force == 'adjusted':
            # PRES/PSAL/TEMP are replaced by PRES_ADJUSTED/PSAL_ADJUSTED/TEMP_ADJUSTED
            for v in ["PRES", "PSAL", "TEMP"]:
                if "%s_ADJUSTED" % v in this.data_vars:
                    this[v] = this["%s_ADJUSTED" % v]
                    this["%s_ERROR" % v] = this["%s_ADJUSTED_ERROR" % v]
                    this["%s_QC" % v] = this["%s_ADJUSTED_QC" % v]
                else:
                    raise InvalidDatasetStructure(
                        "%s_ADJUSTED not in this dataset. Tip: fetch data in 'expert' mode" % v)
            # All ADJUSTED variables are removed (not required anymore, avoid confusion with variable content):
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])
        else:
            # In default mode, we just need to do something if PRES_ADJUSTED is different from PRES, meaning
            # pressure was adjusted:
            if np.any(this['PRES_ADJUSTED'] == this['PRES']):  # Yes
                # We need to recompute salinity with adjusted pressur, so
                # Compute raw conductivity from raw salinity and raw pressure:
                cndc = gsw.C_from_SP(this['PSAL'].values,
                                     this['TEMP'].values,
                                     this['PRES'].values)
                # Then recompute salinity with adjusted pressure:
                sp = gsw.SP_from_C(cndc,
                                   this['TEMP'].values,
                                   this['PRES_ADJUSTED'].values)
                # Now fill in filtered variables (no need to change TEMP):
                this['PRES'] = this['PRES_ADJUSTED']
                this['PRES_QC'] = this['PRES_ADJUSTED_QC']
                this['PSAL'].values = sp

            # Finally drop everything not required anymore:
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])

        # Manage output:
        this.argo._add_history("Variables filtered according to OWC methodology")
        this = this[np.sort(this.data_vars)]
        if to_profile:
            this = this.argo.point2profile()

        # Manage output:
        if inplace:
            self._obj = this
            return self._obj
        else:
            return this

    def subsample_pressure(self, pressure_bin_start: float = 0., pressure_bin: float = 10., inplace: bool = True):
        """ Subsample dataset along pressure bins

        Select vertical levels to keep max 1 level every 10db, starting from the surface (0db)
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/
        # ow_source/create_float_source.m#L208

        You can check the outcome of this filter by comparing the following figures:
        plt.hist(ds['PRES'], bins=np.arange(0,100,1))
        plt.hist(subsample_pressure(ds)['PRES'], pressure_bin_start=0., pressure_bin=10.)

        Parameters
        ----------
        pressure_bin_start: float
            The shallowest pressure value to start bins
        pressure_bin: float
            Pressure bin size
        inplace: boolean, True by default
            If True, return the filtered input :class:`xarray.Dataset`
            If False, return a new :class:`xarray.Dataset`

        Returns
        -------
        :class:`xarray.Dataset`
        """
        this = self._obj

        # Will work with a collection of profiles:
        to_point = False
        if this.argo._type == "point":
            to_point = True
            this = this.argo.point2profile()

        def sub_this_one(pressure, db: int = 10, p0: float = 0):
            bins = np.arange(p0, np.max(pressure) + db, db)
            ip = np.digitize(pressure, bins, right=False)
            ii, ij = np.unique(ip, return_index=True)
            ij = ij[np.where(ij - 1 > 0)] - 1
            return pressure[ij], ij

        # Squeeze all profiles to 1 point every 10db (select deepest value per bins):
        this_dsp_lst = []
        for i_prof in this['N_PROF']:
            up, ip = sub_this_one(this['PRES'].sel(N_PROF=i_prof),
                                  p0=pressure_bin_start,
                                  db=pressure_bin)
            this_dsp_lst.append(this.sel(N_PROF=i_prof).isel(N_LEVELS=ip))

        # Reset N_LEVELS index
        N_LEVELS = 0
        for ids, this_dsp in enumerate(this_dsp_lst):
            maxn = this_dsp['N_LEVELS'].shape[0]
            N_LEVELS = np.max([N_LEVELS, maxn])
            this_dsp_lst[ids] = this_dsp_lst[ids].assign_coords(N_LEVELS=np.arange(0, maxn))

        # Reconstruct the dataset:
        final = xr.concat(this_dsp_lst, 'N_PROF')
        if N_LEVELS != final['N_LEVELS'].shape[0]:
            raise ValueError("Something went wrong with vertical levels")

        # Manage output:
        final.attrs = this.attrs
        if to_point:
            final = final.argo.profile2point()
        if inplace:
            self._obj = this
            return self._obj
        else:
            return final


    def create_float_source(self, file_name, force: str = 'default'):
        """ Create a Matlab file to start the OWC analysis workflow

            From an Argo xarray dataset (as returned by argopy), create a Matlab file to start the OWC analysis workflow.

            Matlab file will have the following variables (n is the number of profiles, m is the number of vertical levels):
                DATES (1xn, in decimal year, e.g. 10 Dec 2000 = 2000.939726)
                LAT   (1xn, in decimal degrees, -ve means south of the equator, e.g. 20.5S = -20.5)
                LONG  (1xn, in decimal degrees, from 0 to 360, e.g. 98.5W in the eastern Pacific = 261.5E)
                PRES  (mxn, dbar, from shallow to deep, e.g. 10, 20, 30 ... These have to line up along a fixed
                        nominal depth axis.)
                TEMP  (mxn, in-situ IPTS-90)
                SAL   (mxn, PSS-78)
                PTMP  (mxn, potential temperature referenced to zero pressure, use SAL in PSS-78 and in-situ TEMP
                        in IPTS-90 for calculation, e.g. sw_ptmp.m)
                PROFILE_NO (1xn, this goes from 1 to n. PROFILE_NO is the same as CYCLE_NO in the Argo files.)

        Parameters
        ----------
        file_name, not used yet
        force: str
            Use force='default' to load PRES/PSAL/TEMP or PRES_ADJUSTED/PSAL/TEMP according to PRES_ADJUSTED filled or not.
            Use force='raw' to force load of PRES/PSAL/TEMP
            Use force='adjusted' to force load of PRES_ADJUSTED/PSAL_ADJUSTED/TEMP_ADJUSTED

        """
        this = self._obj
        log.debug("===================== START create_float_source")
        # log.debug(np.unique(this['PSAL_QC'].values))
        # log.debug("; ".join(["".join(v) for v in this.data_vars]))

        if 'history' in this.attrs and 'DATA_MODE' in this.attrs['history'] and 'QC' in this.attrs['history']:
            # This is surely a dataset fetch with 'standard' mode, we can't deal with this, we need 'expert' file
            raise InvalidDatasetStructure("Need a full Argo dataset to create OWC float source. "
                                          "This dataset was probably loaded with a 'standard' user mode. "
                                          "Try to fetch float data in 'expert' mode")

        if force not in ['default', 'raw', 'adjusted']:
            raise OptionValueError("force option must be 'default', 'raw' or 'adjusted'.")

        def pretty_print_count(dd, txt):
            # if dd.argo._type == "point":
            #     np = len(dd['N_POINTS'].values)
            #     nc = len(dd.argo.point2profile()['N_PROF'].values)
            # else:
            #     np = len(dd.argo.profile2point()['N_POINTS'].values)
            #     nc = len(dd['N_PROF'].values)
            out = []
            np, nc = dd.argo.N_POINTS, dd.argo.N_PROF
            out.append("%i points / %i profiles in dataset %s" % (np, nc, txt))
            # np.unique(this['PSAL_QC'].values))
            out.append(pd.to_datetime(dd['TIME'][0].values).strftime('%Y/%m/%d %H:%M:%S'))
            return "\n".join(out)

        def ds_align_pressure(this, pressure_bins_start, pressure_bin: float = 10.):
            """ Create a new dataset where binned pressure values align on pressure index for all profiles

            This method is intended to be used after subsample_pressure

            Parameters
            ----------
            pressure_bins_start: list
            pressure_bin: float

            Returns
            ------
            :class:`xarray.Dataset`
            """
            def align_pressure(pres_raw, pres_bins_start, pressure_bin: float = 10., fill_value: float = np.nan):
                """ Align pressure values along a given pressure axis
                    For numpy arrays
                """
                pres_align = np.ones_like(pres_bins_start) * fill_value
                index_array = []
                ip_inserted = []
                for ip_insert, p_low in enumerate(pres_bins_start):
                    p_hgh = p_low + pressure_bin
                    ifound = np.digitize(pres_raw, [p_low, p_hgh], right=False)
                    ip = np.argwhere(ifound == 1)
                    if len(ip) > 0:
                        # ip_selected = ip[0][0]  # Select the lowest pressure value in bins
                        ip_selected = ip[-1][-1]  # Select the highest pressure value in bins
                        pres_align[ip_insert] = pres_raw[ip_selected]
                        index_array.append(ip_selected)
                        ip_inserted.append(ip_insert)
                index_array = np.array(index_array)
                ip_inserted = np.array(ip_inserted)
                return pres_align, index_array, ip_inserted

            def replace_i_prof_values(this_da, i_prof, new_values):
                if this_da.dims == ('m', 'n') or this_da.dims == ('m_aligned', 'n'):
                    values = this_da.values
                    values[:, i_prof] = new_values
                    this_da.values = values
                else:
                    raise ValueError("Array not with expected (m, n) shape")
                return this_da

            # Create an empty dataset with the correct nb of vertical levels for each (m,n) variables
            m_aligned = len(pressure_bins_start)
            n = len(this['n'])
            PRES_BINS = np.broadcast_to(pressure_bins_start[:, np.newaxis], (m_aligned, n))
            dsp_aligned = xr.DataArray(PRES_BINS,
                                       dims=['m_aligned', 'n'],
                                       coords={'m_aligned': np.arange(0, PRES_BINS.shape[0]), 'n': this['n']},
                                       name='PRES_BINS').to_dataset(promote_attrs=False)

            for v in this.data_vars:
                if this[v].dims == ('n',):
                    # print('1D:', v)
                    dsp_aligned[v] = this[v]
                if this[v].dims == ('m', 'n'):
                    # print("2D:", v)
                    dsp_aligned[v] = xr.DataArray(np.full_like(PRES_BINS, np.nan),
                                                  dims=['m_aligned', 'n'],
                                                  coords={'m_aligned': np.arange(0, PRES_BINS.shape[0]),
                                                          'n': np.arange(0, PRES_BINS.shape[1])},
                                                  name=v)

            # Align pressure/field values for each profiles:
            for i_prof in dsp_aligned['n']:
                assert this.isel(n=i_prof) == dsp_aligned.isel(n=i_prof)

                p0 = this.isel(n=i_prof)['PRES'].values
                pres_align, index_array, ip_inserted = align_pressure(p0, pressure_bins_start, pressure_bin)
                pres_align = np.round(pres_align, 2)
                dsp_aligned['PRES'] = replace_i_prof_values(dsp_aligned['PRES'], i_prof, pres_align)

                for var in this.data_vars:
                    if this[var].dims == ('m', 'n'):
                        v0 = this.isel(n=i_prof)[var].values
                        v_align = dsp_aligned.isel(n=i_prof)[var].values
                        v_align[ip_inserted] = v0[index_array]
                        dsp_aligned[var] = replace_i_prof_values(dsp_aligned[var], i_prof, v_align)
                        dsp_aligned[var].attrs = this[var].attrs

            dsp_aligned = dsp_aligned.rename({'m_aligned': 'm'})

            # Manage output:
            dsp_aligned.attrs = this.attrs
            return dsp_aligned

        # Add potential temperature:
        if 'PTEMP' not in this:
            this = this.argo.teos10(vlist=['PTEMP'], inplace=True)
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Only use Ascending profiles:
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L143
        this = this.argo._where(this['DIRECTION'] == 'A', drop=True)
        # this = this.argo.cast_types()
        log.debug(pretty_print_count(this, "after direction selection"))
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Todo: ensure we load only the primary profile of cycles with multiple sampling schemes:
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L194

        # Subsample vertical levels (max 1 level every 10db):
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L208
        this = this.argo.subsample_pressure(inplace=False)
        log.debug(pretty_print_count(this, "after vertical levels subsampling"))
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Filter variables according to OWC workflow
        # (I don't understand why this come at the end of the Matlab routine ...)
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L258
        this = this.argo.filter_scalib_pres(force=force, inplace=False)
        log.debug(pretty_print_count(this, "after pressure fields selection"))
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Filter along some QC:
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L372
        this = this.argo.filter_qc(QC_list=[0, 1, 2],
                                   QC_fields=['TIME_QC'],
                                   drop=True)  # Matlab says to reject > 3
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L420
        this = this.argo.filter_qc(QC_list=[v for v in range(10) if v != 3],
                                   QC_fields=['PRES_QC'],
                                   drop=True)  # Matlab says to keep != 3
        this = this.argo.filter_qc(QC_list=[v for v in range(10) if v != 4],
                                   QC_fields=['PRES_QC', 'TEMP_QC', 'PSAL_QC'],
                                   drop=True, mode='any')  # Matlab says to keep != 4
        if len(this['N_POINTS']) == 0:
            raise DataNotFound(
                'All data have been discarded because either PSAL_QC or TEMP_QC is filled with 4 or'
                ' PRES_QC is filled with 3 or 4\n'
                'NO SOURCE FILE WILL BE GENERATED !!!')
        log.debug(pretty_print_count(this, "after QC filter"))

        # Exclude dummies
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L427
        this = this.argo._where(this['PSAL'] <= 50, drop=True)\
                    .argo._where(this['PSAL'] >= 0, drop=True) \
                    .argo._where(this['PTEMP'] <= 50, drop=True)\
                    .argo._where(this['PTEMP'] >= -10, drop=True) \
                    .argo._where(this['PRES'] <= 6000, drop=True)\
                    .argo._where(this['PRES'] >= 0, drop=True)
        if len(this['N_POINTS']) == 0:
            raise DataNotFound(
                'All data have been discarded because they are filled with values out of range\n'
                'NO SOURCE FILE WILL BE GENERATED !!!')
        log.debug(pretty_print_count(this, "after dummy values exclusion"))
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Transform measurements to a collection of profiles for Matlab-like formation:
        this = this.argo.point2profile()
        # log.debug(np.unique(this['PSAL_QC'].values))

        # Compute fractional year:
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L334
        DATES = np.array([toYearFraction(d) for d in pd.to_datetime(this['TIME'].values)])[np.newaxis, :]

        # Read measurements:
        PRES = this['PRES'].values.T  # (mxn)
        TEMP = this['TEMP'].values.T  # (mxn)
        PTMP = this['PTEMP'].values.T  # (mxn)
        SAL = this['PSAL'].values.T  # (mxn)
        LAT = this['LATITUDE'].values[np.newaxis, :]
        # LONG = wrap_longitude(this['LONGITUDE'].values)[np.newaxis, :]
        LONG = this['LONGITUDE'].values[np.newaxis, :]
        PROFILE_NO = this['CYCLE_NUMBER'].values[np.newaxis, :]

        # Create dataset with preprocessed data:
        this_dsp_processed = xr.DataArray(PRES, dims=['m', 'n'], coords={'m': np.arange(0, PRES.shape[0]),
                                                                         'n': np.arange(0, PRES.shape[1])},
                                          name='PRES').to_dataset(promote_attrs=False)
        this_dsp_processed['TEMP'] = xr.DataArray(TEMP, dims=['m', 'n'], coords={'m': np.arange(0, TEMP.shape[0]),
                                                                                 'n': np.arange(0, TEMP.shape[1])},
                                                  name='TEMP')
        this_dsp_processed['PTMP'] = xr.DataArray(PTMP, dims=['m', 'n'], coords={'m': np.arange(0, PTMP.shape[0]),
                                                                                  'n': np.arange(0, PTMP.shape[1])},
                                                   name='PTMP')
        this_dsp_processed['SAL'] = xr.DataArray(SAL, dims=['m', 'n'], coords={'m': np.arange(0, SAL.shape[0]),
                                                                               'n': np.arange(0, SAL.shape[1])},
                                                 name='SAL')
        this_dsp_processed['PROFILE_NO'] = xr.DataArray(PROFILE_NO[0, :], dims=['n'],
                                                        coords={'n': np.arange(0, PROFILE_NO.shape[1])},
                                                        name='PROFILE_NO')
        this_dsp_processed['DATES'] = xr.DataArray(DATES[0, :], dims=['n'], coords={'n': np.arange(0, DATES.shape[1])},
                                                   name='DATES')
        this_dsp_processed['LAT'] = xr.DataArray(LAT[0, :], dims=['n'], coords={'n': np.arange(0, LAT.shape[1])},
                                                 name='LAT')
        this_dsp_processed['LONG'] = xr.DataArray(LONG[0, :], dims=['n'], coords={'n': np.arange(0, LONG.shape[1])},
                                                  name='LONG')
        this_dsp_processed['m'].attrs = {'long_name': 'vertical levels'}
        this_dsp_processed['n'].attrs = {'long_name': 'profiles'}

        # Put all pressure measurements at the same index levels
        # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L451
        bins = np.arange(0., np.max(this_dsp_processed['PRES']) + 10., 10.)
        this_dsp_processed = ds_align_pressure(this_dsp_processed, pressure_bins_start=bins, pressure_bin=10.)

        # Create Matlab dictionary with preprocessed data (to be used by savemat):
        mdata = {}
        mdata['PROFILE_NO'] = PROFILE_NO.astype('uint8')
        mdata['DATES'] = DATES
        mdata['LAT'] = LAT
        mdata['LONG'] = LONG
        mdata['PRES'] = PRES
        mdata['TEMP'] = TEMP
        mdata['PTMP'] = PTMP
        mdata['SAL'] = SAL

        # Save it
        # savemat(file_name, mdata)

        # Temporary output
        return mdata, this_dsp_processed, this
