import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Union
from xarray.backends import BackendEntrypoint  # For xarray > 0.18
from xarray.backends import ZarrStore

try:
    import gsw

    with_gsw = True
except ModuleNotFoundError:
    with_gsw = False

try:
    from dask.delayed import Delayed

    with_dask = True
except ModuleNotFoundError:
    with_dask = False
    Delayed = lambda x: x  # noqa: E731


from .utils import is_list_of_strings
from .utils import (
    cast_Argo_variable_type,
    DATA_TYPES,
    to_list,
)
from .utils import (
    linear_interpolation_remap,
    groupby_remap,
)
from .utils import list_core_parameters
from .utils import toYearFraction
from .errors import InvalidDatasetStructure, DataNotFound, OptionValueError

log = logging.getLogger("argopy.xarray")


@xr.register_dataset_accessor("argo")
class ArgoAccessor:
    """Class registered under scope ``argo`` to access a :class:`xarray.Dataset` object.



    Examples
    --------
    .. code-block:: python
        :caption: Conformity

        ds.argo.cast_types()

    .. code-block:: python
        :caption: Transformation

        ds.argo.point2profile()
        ds.argo.profile2point()
        ds.argo.inter_std_levels(std_lev=[10., 500., 1000.])
        ds.argo.groupby_pressure_bins(bins=[0, 200., 500., 1000.])

    .. code-block:: python
        :caption: QC flags and methods

        ds.argo.filter_qc(QC_list=[1, 2], QC_fields='all')
        ds.argo.filter_scalib_pres(force='default')
        ds.argo.create_float_source("output_folder")

    .. code-block:: python
        :caption: TEOS10

        ds.argo.teos10(vlist='PV')

    .. code-block:: python
        :caption: Extensions: Data Mode

        ds.argo.datamode.compute()
        ds.argo.datamode.merge()
        ds.argo.datamode.filter()
        ds.argo.datamode.filter(dm=['D'], params='all')
        ds.argo.datamode.split()

    .. code-block:: python
        :caption: Extensions: CANYON-MED

        ds.argo.canyon_med.fit()
        ds.argo.canyon_med.predict()
        ds.argo.canyon_med.predict('PO4')

    .. code-block:: python
        :caption: Extensions: Optical modeling

        ds.argo.optic.Zeu()
        ds.argo.optic.Zeu(inplace=True)
        ds.argo.optic.Zeu(method='percentage')
        ds.argo.optic.Zeu(method='KdPAR')

        ds.argo.optic.Zpd()
        ds.argo.optic.Zpd(inplace=True)

    """

    def __init__(self, xarray_obj):
        """Init"""
        self._obj = xarray_obj
        self._added = list()  # Will record all new variables added by argo
        # self._register = collections.OrderedDict() # Will register mutable instances of sub-modules like 'plot'

        # Variables present in the initial dataset
        self._vars = list(xarray_obj.variables.keys())

        # Store the initial list of dimensions
        self._dims = list(xarray_obj.sizes.keys())
        self.encoding = xarray_obj.encoding
        self.attrs = xarray_obj.attrs

        if "N_PROF" in self._dims:
            self._type = "profile"
        elif "N_POINTS" in self._dims:
            self._type = "point"
        else:
            raise InvalidDatasetStructure(
                "Argo dataset structure not recognised (dimensions N_PROF or N_POINTS not found)"
            )

        if "PRES_ADJUSTED" in self._vars:
            self._mode = "expert"
        elif "PRES" in self._vars:
            self._mode = "standard"
        else:
            raise InvalidDatasetStructure(
                "Argo dataset structure not recognised (no PRES nor PRES_ADJUSTED)"
            )

    def __repr__(self):
        # import xarray.core.formatting as xrf
        # col_width = xrf._calculate_col_width(xrf._get_col_items(self._obj.variables))
        # max_rows = xr.core.options.OPTIONS["display_max_rows"]

        summary = ["<xarray.{}.argo>".format(type(self._obj).__name__)]
        if self._type == "profile":
            summary.append("This is a collection of Argo profiles")
            summary.append(
                "N_PROF(%i) x N_LEVELS(%i) ~ N_POINTS(%i)"
                % (self.N_PROF, self.N_LEVELS, self.N_POINTS)
            )

        elif self._type == "point":
            summary.append("This is a collection of Argo points")
            summary.append(
                "N_POINTS(%i) ~ N_PROF(%i) x N_LEVELS(%i)"
                % (self.N_POINTS, self.N_PROF, self.N_LEVELS)
            )

        # dims_start = xrf.pretty_print("Dimensions:", col_width)
        # summary.append("{}({})".format(dims_start, xrf.dim_summary(self._obj)))

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
            N_PROF = len(np.unique(self._obj["N_PROF"]))
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
            N_LEVELS = len(np.unique(self._obj["N_LEVELS"]))
        return N_LEVELS

    @property
    def N_POINTS(self):
        """Number of measurement points"""
        if self._type == "profile":
            N_POINTS = self.N_PROF * self.N_LEVELS
        else:
            N_POINTS = len(np.unique(self._obj["N_POINTS"]))
        return N_POINTS

    def add_history(self, txt):
        if "Processing_history" in self._obj.attrs:
            self._obj.attrs["Processing_history"] += "; %s" % txt
        else:
            self._obj.attrs["Processing_history"] = txt

    def _where(self, cond, other=xr.core.dtypes.NA, drop: bool = False):
        """where that preserve dtypes of Argo fields

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
        this = self._obj.copy(deep=True)
        this = this.where(cond, other=other, drop=drop)
        this = this.argo.cast_types()
        # this.argo.add_history("Modified with 'where' statement")
        return this

    def cast_types(self, **kwargs) -> xr.Dataset:  # noqa: C901
        """Make sure variables are of the appropriate types according to Argo"""
        ds = self._obj
        return cast_Argo_variable_type(ds, **kwargs)

    @property
    def _TNAME(self):
        """Internal handling of the time variable names

        This allows the accessor to work with a dataset:
        - post-processed internally by a datafetcher (where JULD has been renamed TIME)
        - loaded from a raw netcdf

        """
        if "TIME" in self._obj:
            return "TIME"
        elif "JULD" in self._obj:
            return "JULD"

    @property
    def _dummy_argo_uid(self):
        if self._type == "point":
            return xr.DataArray(
                self.uid(
                    self._obj["PLATFORM_NUMBER"].values,
                    self._obj["CYCLE_NUMBER"].values,
                    self._obj["DIRECTION"].values,
                ),
                dims="N_POINTS",
                coords={"N_POINTS": self._obj["N_POINTS"]},
                name="dummy_argo_uid",
            )
        else:
            raise InvalidDatasetStructure(
                "Property only available for a collection of points"
            )

    def uid(self, wmo_or_uid, cyc=None, direction=None):
        """UID encoder/decoder

        Parameters
        ----------
        int
            WMO number (to encode) or UID (to decode)
        cyc: int, optional
            Cycle number (to encode), not used to decode
        direction: str, optional
            Direction of the profile, must be ``A`` (Ascending) or ``D`` (Descending)

        Returns
        -------
        int or tuple of int

        Examples
        --------
        unique_float_profile_id = uid(690024,13,'A') # Encode
        wmo, cyc, drc = uid(unique_float_profile_id) # Decode

        """

        def encode_direction(x):
            y = np.where(x == "A", 1, x.astype(object))
            y = np.where(y == "D", -1, y.astype(object))
            try:
                return y.astype(int)
            except ValueError:
                raise ValueError("x has un-expected values")

        def decode_direction(x):
            x = np.array(x)
            if np.any(np.unique(np.abs(x)) != 1):
                raise ValueError("x has un-expected values")
            y = np.where(x == 1, "A", x)
            y = np.where(y == "-1", "D", y)
            return y.astype("<U1")

        offset = 1e5

        if cyc is not None:
            # ENCODER
            if direction is not None:
                return (
                    encode_direction(direction)
                    * np.vectorize(np.int64)(offset * wmo_or_uid + cyc).ravel()
                )
            else:
                return np.vectorize(np.int64)(offset * wmo_or_uid + cyc).ravel()
        else:
            # DECODER
            drc = decode_direction(np.sign(wmo_or_uid))
            wmo = np.vectorize(int)(np.abs(wmo_or_uid) / offset)
            cyc = -np.vectorize(int)(offset * wmo - np.abs(wmo_or_uid))
            return wmo, cyc, drc

    @property
    def index(self):
        """Basic profile index"""
        if self._type != "point":
            raise InvalidDatasetStructure(
                "Method only available for a collection of points"
            )
        this = self._obj
        dummy_argo_uid = self._dummy_argo_uid

        idx = (
            xr.DataArray(
                this[self._TNAME],
                dims="N_POINTS",
                coords={"N_POINTS": this["N_POINTS"]},
            )
            .groupby(dummy_argo_uid)
            .max()
            .to_dataset()
        )

        for v in ["PLATFORM_NUMBER", "CYCLE_NUMBER", "LONGITUDE", "LATITUDE"]:
            idx[v] = (
                xr.DataArray(
                    this[v],
                    dims="N_POINTS",
                    coords={"N_POINTS": this["N_POINTS"]},
                )
                .groupby(dummy_argo_uid)
                .max()
            )

        df = idx.to_dataframe()
        df = (
            df.reset_index()
            .rename(
                columns={
                    "PLATFORM_NUMBER": "wmo",
                    "CYCLE_NUMBER": "cyc",
                    "LONGITUDE": "longitude",
                    "LATITUDE": "latitude",
                    self._TNAME: "date",
                }
            )
            .drop(columns="dummy_argo_uid")
        )
        df = df[["date", "latitude", "longitude", "wmo", "cyc"]]
        return df

    @property
    def domain(self):
        """Space/time domain of the dataset

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        this_ds = self._obj
        if "PRES_ADJUSTED" in this_ds.data_vars:
            Pmin = np.nanmin(
                (
                    np.min(this_ds["PRES"].values),
                    np.min(this_ds["PRES_ADJUSTED"].values),
                )
            )
            Pmax = np.nanmax(
                (
                    np.max(this_ds["PRES"].values),
                    np.max(this_ds["PRES_ADJUSTED"].values),
                )
            )
        else:
            Pmin = np.min(this_ds["PRES"].values)
            Pmax = np.max(this_ds["PRES"].values)

        return [
            np.min(this_ds["LONGITUDE"].values),
            np.max(this_ds["LONGITUDE"].values),
            np.min(this_ds["LATITUDE"].values),
            np.max(this_ds["LATITUDE"].values),
            Pmin,
            Pmax,
            np.min(this_ds[self._TNAME].values),
            np.max(this_ds[self._TNAME].values),
        ]

    def point2profile(self, drop: bool = False) -> xr.Dataset:  # noqa: C901
        """Transform a collection of points into a collection of profiles

        - A "point" is a location with unique (N_PROF, N_LEVELS) indexes
        - A "profile" is a collection of points with an unique UID based on WMO, CYCLE_NUMBER and DIRECTION

        Parameters
        ----------
        drop: bool, default=False
            By default will return all variables. But if set to True, then all [N_PROF, N_LEVELS] 2d variables will be
            dropped, and only 1d variables of dimension [N_PROF] will be returned.

        Returns
        -------
        :class:`xr.Dataset`

        See Also
        --------
        :meth:`profile2point`

        """
        if self._type != "point":
            raise InvalidDatasetStructure(
                "Method only available for a collection of points"
            )
        if self.N_POINTS == 0:
            raise DataNotFound("Empty dataset, no data to transform !")

        this = self._obj  # Should not be modified

        def fillvalue(da):
            """Return fillvalue for a dataarray"""
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
        dummy_argo_uid = self._dummy_argo_uid
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
        log.debug(
            "point2profile: New dataset should be [N_PROF=%i, N_LEVELS=%i]"
            % (N_PROF, N_LEVELS)
        )
        assert N_PROF * N_LEVELS >= len(this["N_POINTS"])
        if N_LEVELS == 1:
            log.debug(
                "point2profile: This dataset has a single vertical level, thus final variables will only have a N_PROF "
                "dimension and no N_LEVELS"
            )

        # Store the initial set of coordinates:
        coords_list = list(this.coords)
        this = this.reset_coords()

        # For each variable, determine if it has a single unique value by profile,
        # if yes: the transformed variable should be [N_PROF]
        # if no: the transformed variable should be [N_PROF, N_LEVELS]
        # Note: this may lead to differences with the Argo User Manual convention for some variables
        count = np.zeros((N_PROF, len(this.data_vars)), "int")
        for i_prof, grp in enumerate(this.groupby(dummy_argo_uid)):
            i_uid, prof = grp
            for iv, vname in enumerate(this.data_vars):
                try:
                    count[i_prof, iv] = len(np.unique(prof[vname]))  # This is very long because it must read all the data !
                except Exception:
                    log.error(
                        "point2profile: An error happened when dealing with the '%s' data variable"
                        % vname
                    )
                    raise

        # Variables with a unique value for each profiles:
        list_1d = list(np.array(this.data_vars)[count.sum(axis=0) == count.shape[0]])
        # Variables with more than 1 value for each profiles:
        list_2d = list(np.array(this.data_vars)[count.sum(axis=0) != count.shape[0]])

        # Create new empty dataset:
        new_ds = []
        if not drop:
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
                if vname in new_ds and not drop and len(new_ds[vname].dims) == 2:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    try:
                        y[i_prof, 0 : len(x)] = x
                    except Exception:
                        print(vname, "input", x.shape, "output", y[i_prof, :].shape)
                        raise
                    new_ds[vname].values = y
                # ['N_PROF', ] array:
                elif vname in new_ds:
                    y = new_ds[vname].values
                    x = prof[vname].values
                    y[i_prof] = np.unique(x)[0]

        # Restore coordinate variables:
        new_ds = new_ds.set_coords([c for c in coords_list if c in new_ds])
        new_ds["N_PROF"] = np.arange(N_PROF)
        if "N_LEVELS" in new_ds["LATITUDE"].dims:
            new_ds["LATITUDE"] = new_ds["LATITUDE"].isel(
                N_LEVELS=0
            )  # Make sure LAT is (N_PROF) and not (N_PROF, N_LEVELS)
            new_ds["LONGITUDE"] = new_ds["LONGITUDE"].isel(N_LEVELS=0)

        # Misc formatting
        new_ds = new_ds.sortby(self._TNAME)
        new_ds = (
            new_ds.argo.cast_types() if not drop else cast_Argo_variable_type(new_ds)
        )
        new_ds = new_ds[np.sort(new_ds.data_vars)]
        new_ds.encoding = self.encoding  # Preserve low-level encoding information
        new_ds.attrs = self.attrs  # Preserve original attributes
        if not drop:
            new_ds.argo.add_history("Transformed with 'point2profile'")
            new_ds.argo._type = "profile"
        return new_ds

    def profile2point(self) -> xr.Dataset:
        """Transform a collection of profiles to a collection of points

        - A "point" is a location with unique (N_PROF, N_LEVELS) indexes
        - A "profile" is a collection of points with an unique UID based on WMO, CYCLE_NUMBER and DIRECTION

        Note that this method will systematically apply the :meth:`datamode.split` method.

        Returns
        -------
        :class:`xr.Dataset`

        Warnings
        --------
        This method will remove any variable that is not with dimensions (N_PROF,) or (N_PROF, N_LEVELS)

        See Also
        --------
        :meth:`point2profile`
        """
        if self._type != "profile":
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles (N_PROF dimension)"
            )
        ds = self._obj
        ds = ds.argo.datamode.split()  # Otherwise this method will fail with BGC netcdf files

        # Remove all variables for which a dimension is length=0 (eg: N_HISTORY)
        # todo: We should be able to find a way to keep them somewhere in the data structure
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
        # todo: We should be able to find a way to keep them somewhere in the data structure
        for v in ds:
            dims = list(ds[v].dims)
            dims = ".".join(dims)
            if dims not in ["N_PROF", "N_PROF.N_LEVELS"]:
                ds = ds.drop_vars(v)

        (ds,) = xr.broadcast(ds)
        ds = ds.stack({"N_POINTS": list(ds.dims)})
        ds = ds.reset_index("N_POINTS").drop_vars(["N_PROF", "N_LEVELS"])
        possible_coords = ["LATITUDE", "LONGITUDE", self._TNAME, "N_POINTS"]
        for c in [c for c in possible_coords if c in ds.data_vars]:
            ds = ds.set_coords(c)

        # Remove index without data (useless points)
        ds["PRES"].load()
        ds = ds.where(~np.isnan(ds["PRES"]), drop=1)
        ds = ds.sortby(self._TNAME)
        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        ds = cast_Argo_variable_type(ds, overwrite=False)
        ds = ds[np.sort(ds.data_vars)]
        ds.encoding = self.encoding  # Preserve low-level encoding information
        ds.argo.add_history("Transformed with 'profile2point'")
        ds.argo._type = "point"
        return ds

    def filter_qc(  # noqa: C901
        self, QC_list=[1, 2], QC_fields="all", drop=True, mode="all", mask=False
    ):
        """Filter measurements according to QC values

        Filter the dataset to keep points where ``all`` or ``any`` of the QC fields has a value in the list
        of integer QC flags.

        This method can return the filtered dataset or the filter mask.

        Warnings
        --------
        This method does not consider PROFILE QC variable(s).

        Parameters
        ----------
        QC_list: list of int
            List of QC flag values (integers) to keep
        QC_fields: 'all' or list(str)
            List of QC fields to consider to apply the filter. By default, we use all available QC fields
        drop: bool
            Drop values not matching the QC filter, default is True
        mode: str
            Must be ``all`` (default) or ``any``. Boolean operator on QC values: should we keep points
            matching ``all`` QC fields or 'any' one of them.
        mask: bool
            ``False`` by default. Determine if we should return the QC mask or the filtered dataset.

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

        this = self._obj

        # Extract QC fields:
        if isinstance(QC_fields, str) and QC_fields == "all":
            QC_fields = []
            for v in this.data_vars:
                if "QC" in v and "PROFILE" not in v:
                    QC_fields.append(v)
        elif is_list_of_strings(QC_fields):
            for v in QC_fields:
                if v not in this.data_vars:
                    raise ValueError(
                        "%s not found in this dataset while trying to apply QC filter"
                        % v
                    )
        else:
            raise ValueError(
                "Invalid content for parameter 'QC_fields'. Use 'all' or a list of strings"
            )

        if len(QC_fields) == 0:
            this.argo.add_history(
                "Variables selected according to QC (but found no QC variables)"
            )
            return this

        log.debug(
            "filter_qc: Filtering dataset to keep points with QC in %s for '%s' fields in %s"
            % (QC_list, mode, ",".join(QC_fields))
        )
        # log.debug("filter_qc: Filter applied to '%s' of the fields: %s" % (mode, ",".join(QC_fields)))

        QC_fields = this[QC_fields]  # QC_fields is now a :class:`xr.Dataset`
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
            this.argo.add_history(
                "[%s] filtered to retain points with QC in [%s]"
                % (
                    ",".join(list(QC_fields.data_vars)),
                    ",".join([str(qc) for qc in QC_list]),
                )
            )
            if this.argo.N_POINTS == 0:
                log.warning("No data left after QC filtering !")
            return this
        else:
            return this_mask

    def filter_scalib_pres(self, force: str = "default", inplace: bool = True):
        """Filter variables according to OWC salinity calibration software requirements

        By default, this filter will return a dataset with raw PRES, PSAL and TEMP; and if PRES is adjusted,
        PRES variable will be replaced by PRES_ADJUSTED.

        With option force='raw', you can force the filter to return a dataset with raw PRES, PSAL and TEMP whether
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

        if force == "raw":
            # PRES/PSAL/TEMP are not changed
            # All ADJUSTED variables are removed (not required anymore, avoid confusion with variable content):
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])
        elif force == "adjusted":
            # PRES/PSAL/TEMP are replaced by PRES_ADJUSTED/PSAL_ADJUSTED/TEMP_ADJUSTED
            for v in ["PRES", "PSAL", "TEMP"]:
                if "%s_ADJUSTED" % v in this.data_vars:
                    this[v] = this["%s_ADJUSTED" % v]
                    this["%s_ERROR" % v] = this["%s_ADJUSTED_ERROR" % v]
                    this["%s_QC" % v] = this["%s_ADJUSTED_QC" % v]
                else:
                    raise InvalidDatasetStructure(
                        "%s_ADJUSTED not in this dataset. Tip: fetch data in 'expert' mode"
                        % v
                    )
            # All ADJUSTED variables are removed (not required anymore, avoid confusion with variable content):
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])
        else:
            if "PRES_ADJUSTED" not in this:
                raise InvalidDatasetStructure(
                    "%s_ADJUSTED not in this dataset. Tip: fetch data in 'expert' mode"
                    % "PRES"
                )

            # In default mode, we just need to do something if PRES_ADJUSTED is different from PRES, meaning
            # pressure was adjusted:
            if np.any(this["PRES_ADJUSTED"] == this["PRES"]):  # Yes
                # We need to recompute salinity with adjusted pressure, so
                # Compute raw conductivity from raw salinity and raw pressure:
                cndc = gsw.C_from_SP(
                    this["PSAL"].values, this["TEMP"].values, this["PRES"].values
                )
                # Then recompute salinity with adjusted pressure:
                sp = gsw.SP_from_C(
                    cndc, this["TEMP"].values, this["PRES_ADJUSTED"].values
                )
                # Now fill in filtered variables (no need to change TEMP):
                this["PRES"] = this["PRES_ADJUSTED"]
                this["PRES_QC"] = this["PRES_ADJUSTED_QC"]
                this["PSAL"].values = sp

            # Finally drop everything not required anymore:
            this = this.drop_vars([v for v in this.data_vars if "ADJUSTED" in v])

        # Manage output:
        this.argo.add_history("Variables filtered according to OWC methodology")
        this = this[np.sort(this.data_vars)]
        if to_profile:
            this = this.argo.point2profile()

        # Manage output:
        if inplace:
            self._obj = this
            return self._obj
        else:
            return this

    def filter_researchmode(self) -> xr.Dataset:
        """Filter dataset for research user mode

        This filter depends on the dataset:

        - For the ``phy`` dataset (core/deep missions): select delayed mode data with QC=1 and with pressure errors smaller than 20db
        - For the ``bgc`` dataset: do nothing, filtering for the ``research`` user mode is implemented in the fetcher facade


        Returns
        -------
        :class:`xarray.Dataset`
        """
        this = self._obj

        # Ensure to work with a collection of points
        to_profile = False
        if this.argo._type == "profile":
            to_profile = True
            this = this.argo.profile2point()

        core_params = list_core_parameters()
        if "PSAL" not in this.data_vars and "PSAL_ADJUSTED" not in this.data_vars:
            core_params.remove("PSAL")

        # Apply transforms and filters:
        this = this.argo.filter_qc(QC_list=1, QC_fields=["POSITION_QC", "TIME_QC"])
        this = this.argo.datamode.merge(params=core_params)
        this = this.argo.datamode.filter(params=core_params, dm="D")

        this = this.argo.filter_qc(
            QC_list=1, QC_fields=["%s_QC" % p for p in core_params]
        )

        if (
            "PRES_ERROR" in this.data_vars
        ):  # PRES_ADJUSTED_ERROR was renamed PRES_ERROR by transform_data_mode
            this = this.where(this["PRES_ERROR"] < 20, drop=True)
        this.argo.add_history(
            "[%s] parameters selected for pressure error < 20db"
            % (",".join(core_params))
        )

        # Manage output:
        if to_profile:
            this = this.argo.point2profile()
        if this.argo.N_POINTS == 0:
            log.warning("No data left after Research-mode filtering !")
        else:
            this = this.argo.cast_types()
        return this

    def interp_std_levels(
        self, std_lev: list or np.array, axis: str = "PRES"
    ) -> xr.Dataset:
        """Interpolate measurements to standard pressure levels

        Parameters
        ----------
        std_lev: list or np.array
            Standard pressure levels used for interpolation. It has to be 1-dimensional and monotonic.
        axis: str, default: ``PRES``
            The dataset variable to use as pressure axis. This could be ``PRES`` or ``PRES_ADJUSTED``.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        this_dsp = self._obj

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

        if axis not in ["PRES", "PRES_ADJUSTED"]:
            raise ValueError("'axis' option must be 'PRES' or 'PRES_ADJUSTED'")

        if self._type != "profile":
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles"
            )

        # Will work with a collection of profiles:
        # to_point = False
        # if this_ds.argo._type == "point":
        #     to_point = True
        #     this_dsp = this_ds.argo.point2profile()
        # else:
        #     this_dsp = this_ds.copy(deep=True)

        # Selecting profiles that have a max(pressure) > max(std_lev) to avoid extrapolation in that direction
        # For levels < min(pressure), first level values of the profile are extended to surface.
        i1 = this_dsp[axis].max("N_LEVELS") >= std_lev[-1]
        this_dsp = this_dsp.where(i1, drop=True)

        # check if any profile is left, ie if any profile match the requested depth
        if len(this_dsp["N_PROF"]) == 0:
            warnings.warn(
                "None of the profiles can be interpolated (not reaching the requested depth range)."
            )
            return None

        # add new vertical dimensions, this has to be in the datasets to apply ufunc later
        this_dsp["Z_LEVELS"] = xr.DataArray(std_lev, dims={"Z_LEVELS": std_lev})

        # init
        ds_out = xr.Dataset()

        # vars to interpolate
        datavars = [
            dv
            for dv in list(this_dsp.variables)
            if set(["N_LEVELS", "N_PROF"]) == set(this_dsp[dv].dims)
            and "QC" not in dv
            and "ERROR" not in dv
            and "DATA_MODE" not in dv
        ]
        # coords
        coords = [dv for dv in list(this_dsp.coords)]
        # vars depending on N_PROF only
        solovars = [
            dv
            for dv in list(this_dsp.variables)
            if dv not in datavars
            and dv not in coords
            and "QC" not in dv
            and "ERROR" not in dv
        ]

        for dv in datavars:
            ds_out[dv] = linear_interpolation_remap(
                this_dsp[axis],
                this_dsp[dv],
                this_dsp["Z_LEVELS"],
                z_dim="N_LEVELS",
                z_regridded_dim="Z_LEVELS",
            )
            ds_out[dv].attrs = this_dsp[dv].attrs  # Preserve attributes
            if "long_name" in ds_out[dv].attrs:
                ds_out[dv].attrs["long_name"] = (
                    "Interpolated %s" % ds_out[dv].attrs["long_name"]
                )

        ds_out = ds_out.rename({"remapped": "%s_INTERPOLATED" % axis})
        ds_out["%s_INTERPOLATED" % axis].attrs = this_dsp[axis].attrs
        if "long_name" in ds_out["%s_INTERPOLATED" % axis].attrs:
            ds_out["%s_INTERPOLATED" % axis].attrs["long_name"] = (
                "Standard %s levels" % axis
            )

        for sv in solovars:
            ds_out[sv] = this_dsp[sv]

        for co in coords:
            ds_out.coords[co] = this_dsp[co]

        ds_out = ds_out.drop_vars(["N_LEVELS", "Z_LEVELS"])
        ds_out = ds_out[np.sort(ds_out.data_vars)]
        ds_out = ds_out.argo.cast_types()
        ds_out.attrs = self.attrs  # Preserve original attributes
        ds_out.argo.add_history("Interpolated on standard %s levels" % axis)

        # if to_point:
        #     ds_out = ds_out.argo.profile2point()

        return ds_out

    def groupby_pressure_bins(
        self,  # noqa: C901
        bins: list or np.array,
        axis: str = "PRES",
        right: bool = False,
        select: str = "deep",
        squeeze: bool = True,
        merge: bool = True,
    ) -> xr.Dataset:
        """Group measurements by pressure bins

        This method can be used to subsample and align an irregular dataset (pressure not being similar in all profiles)
        on a set of pressure bins. The output dataset could then be used to perform statistics along the ``N_PROF`` dimension
        because ``N_LEVELS`` will correspond to similar pressure bins, while avoiding to interpolate data.

        Parameters
        ----------
        bins: list or np.array,
            Array of bins. It has to be 1-dimensional and monotonic. Bins of data are localised using values from
            options `axis` (default: ``PRES``) and `right` (default: ``False``), see below.
        axis: str, default: ``PRES``
            The dataset variable to use as pressure axis. This could be ``PRES`` or ``PRES_ADJUSTED``
        right: bool, default: False
            Indicating whether the bin intervals include the right or the left bin edge. Default behavior is
            (right==False) indicating that the interval does not include the right edge. The left bin end is open
            in this case, i.e., bins[i-1] <= x < bins[i] is the default behavior for monotonically increasing bins.
            Note the ``merge`` option is intended to work only for the default ``right=False``.
        select: {'deep','shallow','middle','random','min','max','mean','median'}, default: 'deep'
            The value selection method for bins.

            This selection can be based on values at the pressure axis level with: ``deep`` (default), ``shallow``,
            ``middle``, ``random``. For instance, ``select='deep'`` will lead to the value
            returned for a bin to be taken at the deepest pressure level in the bin.

            Or this selection can be based on statistics of measurements in a bin. Stats available are: ``min``, ``max``,
            ``mean``, ``median``. For instance ``select='mean'`` will lead to the value returned for a bin to be the mean of
            all measurements in the bin.
        squeeze: bool, default: True
            Squeeze from the output bin levels without measurements.
        merge: bool, default: True
            Optimize the output bins axis size by merging levels with/without data. The pressure bins axis is modified
            accordingly. This means that the return ``STD_PRES_BINS`` axis has not necessarily the same size as
            the input ``bins``.

        Returns
        -------
        :class:`xarray.Dataset`

        See Also
        --------
        :class:`numpy.digitize`, :class:`argopy.utils.groupby_remap`
        """
        this_ds = self._obj

        if (type(bins) is np.ndarray) | (type(bins) is list):
            bins = np.array(bins)
            if (np.any(sorted(bins) != bins)) | (np.any(bins < 0)):
                raise ValueError(
                    "Standard bins must be a list or a numpy array of positive and sorted values"
                )
        else:
            raise ValueError(
                "Standard bins must be a list or a numpy array of positive and sorted values"
            )

        if axis not in ["PRES", "PRES_ADJUSTED"]:
            raise ValueError("'axis' option must be 'PRES' or 'PRES_ADJUSTED'")

        # Will work with a collection of profiles:
        to_point = False
        if this_ds.argo._type == "point":
            to_point = True
            this_dsp = this_ds.argo.point2profile()
        else:
            this_dsp = this_ds.copy(deep=True)

        # Adjust bins axis if we possibly have to squeeze empty bins:
        h, bin_edges = np.histogram(np.unique(np.round(this_dsp[axis], 1)), bins)
        N_bins_empty = len(np.where(h == 0)[0])
        # check if any profile is left, ie if any profile match the requested bins
        if N_bins_empty == len(h):
            warnings.warn(
                "None of the profiles can be aligned (pressure values out of bins range)."
            )
            return None
        if N_bins_empty > 0 and squeeze:
            log.debug(
                "bins axis was squeezed to full bins only (%i bins found empty out of %i)"
                % (N_bins_empty, len(bins))
            )
            bins = bins[np.where(h > 0)]

        def replace_i_level_values(this_da, this_i_level, new_values_along_profiles):
            """Convenience fct to update only one level of a ["N_PROF", "N_LEVELS"] xr.DataArray"""
            if this_da.dims == ("N_PROF", "N_LEVELS"):
                values = this_da.values
                values[:, this_i_level] = new_values_along_profiles
                this_da.values = values
            # else:
            #     raise ValueError("Array not with expected ['N_PROF', 'N_LEVELS'] shape")
            return this_da

        def nanmerge(x, y):
            """Merge two 1D array

            Given 2 arrays x, y of 1 dimension, return a new array with:
            - x values where x is not NaN
            - y values where x is NaN
            """
            z = x.copy()
            for i, v in enumerate(x):
                if np.isnan(v):
                    z[i] = y[i]
            return z

        merged_is_nan = lambda l1, l2: len(  # noqa: E731
            np.unique(np.where(np.isnan(l1.values + l2.values)))
        ) == len(l1)

        def merge_bin_matching_levels(this_ds: xr.Dataset) -> xr.Dataset:
            """Levels merger of type 'bins' value

            Merge pair of lines with the following pattern:
               nan,    VAL, VAL, nan,    VAL, VAL
               BINVAL, nan, nan, BINVAL, nan, nan

            This pattern is due to the bins definition: bins[i] <= x < bins[i+1]

            Parameters
            ----------
            :class:`xarray.Dataset`

            Returns
            -------
            :class:`xarray.Dataset`
            """
            new_ds = this_ds.copy(deep=True)
            N_LEVELS = new_ds.argo.N_LEVELS
            idel = []
            for i_level in range(0, N_LEVELS - 1 - 1):
                this_ds_level = this_ds[axis].isel(N_LEVELS=i_level)
                this_ds_dw = this_ds[axis].isel(N_LEVELS=i_level + 1)
                pres_dw = np.unique(this_ds_dw[~np.isnan(this_ds_dw)])
                if (
                    len(pres_dw) == 1
                    and pres_dw[0] in this_ds["STD_%s_BINS" % axis]
                    and merged_is_nan(this_ds_level, this_ds_dw)
                ):
                    new_values = nanmerge(this_ds_dw.values, this_ds_level.values)
                    replace_i_level_values(new_ds[axis], i_level, new_values)
                    idel.append(i_level + 1)

            ikeep = [i for i in np.arange(0, new_ds.argo.N_LEVELS - 1) if i not in idel]
            new_ds = new_ds.isel(N_LEVELS=ikeep)
            new_ds = new_ds.assign_coords(
                {"N_LEVELS": np.arange(0, len(new_ds["N_LEVELS"]))}
            )
            val = new_ds[axis].values
            new_ds[axis].values = np.where(val == 0, np.nan, val)
            return new_ds

        def merge_all_matching_levels(this_ds: xr.Dataset) -> xr.Dataset:
            """Levels merger

            Merge any pair of levels with a "matching" pattern like this:
               VAL, VAL, VAL, nan, nan, VAL, nan, nan,
               nan, nan, nan, VAL, VAL, nan, VAL, nan

            This pattern is due to a strict application of the bins definition.
            But when bins are small (eg: 10db), many bins can have no data.
            This has the consequence to change the size and number of the bins.

            Parameters
            ----------
            :class:`xarray.Dataset`

            Returns
            -------
            :class:`xarray.Dataset`
            """
            new_ds = this_ds.copy(deep=True)
            N_LEVELS = new_ds.argo.N_LEVELS
            idel = []
            for i_level in range(0, N_LEVELS):
                if i_level + 1 < N_LEVELS:
                    this_ds_level = this_ds[axis].isel(N_LEVELS=i_level)
                    this_ds_dw = this_ds[axis].isel(N_LEVELS=i_level + 1)
                    if merged_is_nan(this_ds_level, this_ds_dw):
                        new_values = nanmerge(this_ds_level.values, this_ds_dw.values)
                        replace_i_level_values(new_ds[axis], i_level, new_values)
                        idel.append(i_level + 1)

            ikeep = [i for i in np.arange(0, new_ds.argo.N_LEVELS - 1) if i not in idel]
            new_ds = new_ds.isel(N_LEVELS=ikeep)
            new_ds = new_ds.assign_coords(
                {"N_LEVELS": np.arange(0, len(new_ds["N_LEVELS"]))}
            )
            val = new_ds[axis].values
            new_ds[axis].values = np.where(val == 0, np.nan, val)
            return new_ds

        # init
        new_ds = []

        # add new vertical dimensions, this has to be in the datasets to apply ufunc later
        this_dsp["Z_LEVELS"] = xr.DataArray(bins, dims={"Z_LEVELS": bins})

        # vars to align
        if select in ["shallow", "deep", "middle", "random"]:
            datavars = [
                dv
                for dv in list(this_dsp.data_vars)
                if set(["N_LEVELS", "N_PROF"]) == set(this_dsp[dv].dims)
                and dv not in DATA_TYPES["data"]["str"]
            ]
        else:
            datavars = [
                dv
                for dv in list(this_dsp.data_vars)
                if set(["N_LEVELS", "N_PROF"]) == set(this_dsp[dv].dims)
                and "QC" not in dv
                and "ERROR" not in dv
                and dv not in DATA_TYPES["data"]["str"]
            ]

        # All other variables:
        othervars = [
            dv
            for dv in list(this_dsp.variables)
            if dv not in datavars and dv not in this_dsp.coords
        ]

        # Sub-sample and align:
        for dv in datavars:
            v = groupby_remap(
                this_dsp[axis],
                this_dsp[dv],
                this_dsp["Z_LEVELS"],
                z_dim="N_LEVELS",
                z_regridded_dim="Z_LEVELS",
                select=select,
                right=right,
            )
            v.name = this_dsp[dv].name
            v.attrs = this_dsp[dv].attrs
            new_ds.append(v)

        # Finish
        new_ds = xr.merge(new_ds)
        new_ds = new_ds.rename({"remapped": "N_LEVELS"})
        new_ds = new_ds.assign_coords({"N_LEVELS": range(0, len(new_ds["N_LEVELS"]))})
        # new_ds["STD_%s_BINS" % axis] = new_ds['N_LEVELS']
        new_ds["STD_%s_BINS" % axis] = xr.DataArray(
            bins,
            dims=["N_LEVELS"],
            attrs={
                "Comment": "Range of bins is: bins[i] <= x < bins[i+1] for i=[0,N_LEVELS-2]\n"
                "Last bins is bins[N_LEVELS-1] <= x"
            },
        )

        new_ds = new_ds.set_coords("STD_%s_BINS" % axis)
        new_ds.attrs = this_ds.attrs

        for dv in othervars:
            new_ds[dv] = this_dsp[dv]

        new_ds = new_ds.argo.cast_types()
        new_ds = new_ds[np.sort(new_ds.data_vars)]
        new_ds.attrs = this_dsp.attrs  # Preserve original attributes
        new_ds.argo.add_history("Sub-sampled and re-aligned on standard bins")

        if merge:
            new_ds = merge_bin_matching_levels(new_ds)
            new_ds = merge_all_matching_levels(new_ds)

        if to_point:
            new_ds = new_ds.argo.profile2point()

        return new_ds

    def teos10(  # noqa: C901
        self,
        vlist: list = ["SA", "CT", "SIG0", "N2", "PV", "PTEMP"],
        inplace: bool = True,
    ):
        """Add TEOS10 variables to the dataset

        By default, adds: 'SA', 'CT'
        Other possible variables: 'SIG0', 'N2', 'PV', 'PTEMP', 'SOUND_SPEED'
        Relies on the gsw library.

        If one exists, the correct CF standard name will be added to the attrs.

        Parameters
        ----------
        vlist: list(str)
            List with the name of variables to add.
            Must be a list containing one or more of the following string values:

            * ``SA``
                Adds an absolute salinity variable
            * ``CT``
                Adds a conservative temperature variable
            * ``SIG0``
                Adds a potential density anomaly variable referenced to 0 dbar
            * ``N2``
                Adds a buoyancy (Brunt-Vaisala) frequency squared variable.
                This variable has been regridded to the original pressure levels in the Dataset using a linear interpolation.
            * ``PV``
                Adds a planetary vorticity variable calculated from :math:`\\frac{f N^2}{\\text{gravity}}`.
                This is not a TEOS-10 variable from the gsw toolbox, but is provided for convenience.
                This variable has been regridded to the original pressure levels in the Dataset using a linear interpolation.
            * ``PTEMP``
                Add potential temperature
            * ``SOUND_SPEED``
                Add sound speed
            * ``CNDC``
                Add Electrical Conductivity

        inplace: boolean, True by default
            * If True, return the input :class:`xarray.Dataset` with new TEOS10 variables
                added as a new :class:`xarray.DataArray`.
            * If False, return a :class:`xarray.Dataset` with new TEOS10 variables

        Returns
        -------
        :class:`xarray.Dataset`
        """
        if not with_gsw:
            raise ModuleNotFoundError("This functionality requires the gsw library")

        allowed = ["SA", "CT", "SIG0", "N2", "PV", "PTEMP", "SOUND_SPEED", "CNDC"]
        if any(var not in allowed for var in vlist):
            raise ValueError(
                f"vlist must be a subset of {allowed}, instead found {vlist}"
            )

        # if is_list_equal(vlist, ["SA", "CT"]):
        #     warnings.warn(
        #         "Default variables will be reduced to 'SA' and 'CT' in 0.1.9",
        #         category=FutureWarning,
        #     )

        this = self._obj

        to_profile = False
        if self._type == "profile":
            to_profile = True
            this = this.argo.profile2point()

        # Get base variables as numpy arrays:
        psal = this["PSAL"].values
        temp = this["TEMP"].values
        pres = this["PRES"].values
        lon = this["LONGITUDE"].values
        lat = this["LATITUDE"].values

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
        if "SOUND_SPEED" in vlist:
            cs = gsw.sound_speed(sa, ct, pres)

        # Back to the dataset:
        that = []
        if "SA" in vlist:
            SA = xr.DataArray(sa, coords=this["PSAL"].coords, name="SA")
            SA.attrs["long_name"] = "Absolute Salinity"
            SA.attrs["standard_name"] = "sea_water_absolute_salinity"
            SA.attrs["unit"] = "g/kg"
            that.append(SA)

        if "CT" in vlist:
            CT = xr.DataArray(ct, coords=this["TEMP"].coords, name="CT")
            CT.attrs["long_name"] = "Conservative Temperature"
            CT.attrs["standard_name"] = "sea_water_conservative_temperature"
            CT.attrs["unit"] = "degC"
            that.append(CT)

        if "SIG0" in vlist:
            SIG0 = xr.DataArray(sig0, coords=this["TEMP"].coords, name="SIG0")
            SIG0.attrs[
                "long_name"
            ] = "Potential density anomaly with reference pressure of 0 dbar"
            SIG0.attrs["standard_name"] = "sea_water_sigma_theta"
            SIG0.attrs["unit"] = "kg/m^3"
            that.append(SIG0)

        if "CNDC" in vlist:
            CNDC = xr.DataArray(cndc, coords=this["TEMP"].coords, name="CNDC")
            CNDC.attrs["long_name"] = "Electrical Conductivity"
            CNDC.attrs["standard_name"] = "sea_water_electrical_conductivity"
            CNDC.attrs["unit"] = "mS/cm"
            that.append(CNDC)

        if "N2" in vlist:
            N2 = xr.DataArray(n2, coords=this["TEMP"].coords, name="N2")
            N2.attrs["long_name"] = "Squared buoyancy frequency"
            N2.attrs["unit"] = "1/s^2"
            that.append(N2)

        if "PV" in vlist:
            PV = xr.DataArray(pv, coords=this["TEMP"].coords, name="PV")
            PV.attrs["long_name"] = "Planetary Potential Vorticity"
            PV.attrs["unit"] = "1/m/s"
            that.append(PV)

        if "PTEMP" in vlist:
            PTEMP = xr.DataArray(pt, coords=this["TEMP"].coords, name="PTEMP")
            PTEMP.attrs["long_name"] = "Potential Temperature"
            PTEMP.attrs["standard_name"] = "sea_water_potential_temperature"
            PTEMP.attrs["unit"] = "degC"
            that.append(PTEMP)

        if "SOUND_SPEED" in vlist:
            CS = xr.DataArray(cs, coords=this["TEMP"].coords, name="SOUND_SPEED")
            CS.attrs["long_name"] = "Speed of sound"
            CS.attrs["standard_name"] = "speed_of_sound_in_sea_water"
            CS.attrs["unit"] = "m/s"
            that.append(CS)

        # Create a dataset with all new variables:
        that = xr.merge(that)
        # Add to the dataset essential Argo variables (allows to keep using the argo accessor):
        that = that.assign(
            {
                k: this[k]
                for k in [
                    self._TNAME,
                    "LATITUDE",
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

    def create_float_source(
        self,  # noqa: C901
        path: str or os.PathLike = None,
        force: str = "default",
        select: str = "deep",
        file_pref: str = "",
        file_suff: str = "",
        format: str = "5",
        do_compression: bool = True,
        debug_output: bool = False,
    ):
        """Preprocess data for OWC software calibration

        This method can create a FLOAT SOURCE file (i.e. the .mat file that usually goes into /float_source/) for OWC software.
        The FLOAT SOURCE file is saved as:

        ``<path>/<file_pref><float_WMO><file_suff>.mat``

        where ``<float_WMO>`` is automatically extracted from the dataset variable PLATFORM_NUMBER (in order to avoid mismatch
        between user input and data content). So if this dataset has measurements from more than one float, more than one
        Matlab file will be created.

        By default, variables loaded are raw PRES, PSAL and TEMP.
        If PRES is adjusted, variables loaded are PRES_ADJUSTED, raw PSAL calibrated in pressure and raw TEMP.

        You can force the program to load raw PRES, PSAL and TEMP whatever PRES is adjusted or not:

        ds.argo.create_float_source(force='raw')

        or you can force the program to load adjusted variables: PRES_ADJUSTED, PSAL_ADJUSTED, TEMP_ADJUSTED

        ds.argo.create_float_source(force='adjusted')

        **Pre-processing details**:

        #.  select only ascending profiles

        #.  subsample vertical levels to keep the deepest pressure levels on each 10db bins from the surface down
            to the deepest level.

        #.  align pressure values, i.e. make sure that a pressure index corresponds to measurements from the same
            binned pressure values. This can lead to modify the number of levels in the dataset.

        #.  filter variables according to the ``force`` option (see below)

        #.  filter variables according to QC flags:

            *  Remove measurements where timestamp QC is >= 3
            *  Keep measurements where pressure QC is anything but 3
            *  Keep measurements where pressure, temperature or salinity QC are anything but 4


        #.  remove dummy values: salinity not in [0/50], potential temperature not in [-10/50] and pressure not
            in [0/60000]. Bounds inclusive.

        #.  convert timestamp to fractional year

        #.  convert longitudes to 0-360



        Parameters
        ----------
        path: str or path-like, optional
            Path or folder name to which to save this Matlab file. If no path is provided, this function returns the
            resulting Matlab file as :class:`xarray.Dataset`.
        force: {"default", "raw", "adjusted"}, default: "default"
            If force='default' will load PRES/PSAL/TEMP or PRES_ADJUSTED/PSAL/TEMP according to PRES_ADJUSTED filled or not.

            If force='raw' will load PRES/PSAL/TEMP

            If force='adjusted' will load PRES_ADJUSTED/PSAL_ADJUSTED/TEMP_ADJUSTED
        select: {'deep','shallow','middle','random','min','max','mean','median'}, default: 'deep'
        file_pref: str, optional
            Prefix to add at the beginning of output file(s).
        file_suff: str, optional
            Suffix to add at the end of output file(s).
        do_compression: bool, optional
            Whether to compress matrices on write. Default is True.
        format: {'5', '4'}, string, optional
            Matlab file format version. '5' (the default) for MATLAB 5 and up (to 7.2). Use '4' for MATLAB 4 .mat files.

        Returns
        -------
        :class:`xarray.Dataset`
            The output dataset, or Matlab file, will have the following variables (``n`` is the number of profiles, ``m``
            is the number of vertical levels):

            - ``DATES`` (1xn): decimal year, e.g. 10 Dec 2000 = 2000.939726
            - ``LAT``   (1xn): decimal degrees, -ve means south of the equator, e.g. 20.5S = -20.5
            - ``LONG``  (1xn): decimal degrees, from 0 to 360, e.g. 98.5W in the eastern Pacific = 261.5E
            - ``PROFILE_NO`` (1xn): this goes from 1 to n. PROFILE_NO is the same as CYCLE_NO in the Argo files
            - ``PRES``  (mxn): dbar, from shallow to deep, e.g. 10, 20, 30 ... These have to line up along a fixed nominal depth axis.
            - ``TEMP``  (mxn): in-situ IPTS-90
            - ``SAL``   (mxn): PSS-78
            - ``PTMP``  (mxn): potential temperature referenced to zero pressure, use SAL in PSS-78 and in-situ TEMP in IPTS-90 for calculation.

        """
        this = self._obj

        if (
            "history" in this.attrs
            and "DATA_MODE" in this.attrs["history"]
            and "QC" in this.attrs["history"]
        ):
            # This is surely a dataset fetch with 'standard' mode, we can't deal with this, we need 'expert' file
            raise InvalidDatasetStructure(
                "Need a full Argo dataset to create OWC float source. "
                "This dataset was probably loaded with a 'standard' user mode. "
                "Try to fetch float data in 'expert' mode"
            )

        if force not in ["default", "raw", "adjusted"]:
            raise OptionValueError(
                "force option must be 'default', 'raw' or 'adjusted'."
            )

        log.debug(
            "===================== START create_float_source in '%s' mode" % force
        )

        if len(np.unique(this["PLATFORM_NUMBER"])) > 1:
            log.debug(
                "Found more than one 1 float in this dataset, will split processing"
            )

        def ds2mat(this_dsp):
            # Return a Matlab dictionary with dataset data to be used by savemat:
            mdata = {}
            mdata["PROFILE_NO"] = (
                this_dsp["PROFILE_NO"].astype("uint16").values.T[np.newaxis, :]
            )  # 1-based index in Matlab
            mdata["DATES"] = this_dsp["DATES"].values.T[np.newaxis, :]
            mdata["LAT"] = this_dsp["LAT"].values.T[np.newaxis, :]
            mdata["LONG"] = this_dsp["LONG"].values.T[np.newaxis, :]
            mdata["PRES"] = this_dsp["PRES"].values
            mdata["TEMP"] = this_dsp["TEMP"].values
            mdata["PTMP"] = this_dsp["PTMP"].values
            mdata["SAL"] = this_dsp["SAL"].values
            return mdata

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
            # out.append(pd.to_datetime(dd['TIME'][0].values).strftime('%Y/%m/%d %H:%M:%S'))
            return "\n".join(out)

        def getfilled_bins(pressure, bins):
            ip = np.digitize(np.unique(pressure), bins, right=False)
            ii, ij = np.unique(ip, return_index=True)
            ii = ii[np.where(ii - 1 > 0)] - 1
            return bins[ii]

        def preprocess_one_float(
            this_one: xr.Dataset,
            this_path: str or os.PathLike = None,
            select: str = "deep",
            debug_output: bool = False,
        ):
            """Run the entire preprocessing on a given dataset with one float data"""

            # Add potential temperature:
            if "PTEMP" not in this_one:
                this_one = this_one.argo.teos10(vlist=["PTEMP"], inplace=True)

            # Only use Ascending profiles:
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L143
            this_one = this_one.argo._where(this_one["DIRECTION"] == "A", drop=True)
            log.debug(pretty_print_count(this_one, "after direction selection"))

            # Todo: ensure we load only the primary profile of cycles with multiple sampling schemes:
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L194

            # # Subsample and align vertical levels (max 1 level every 10db):
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L208
            # this_one = this_one.argo.align_std_bins(inplace=False)
            # log.debug(pretty_print_count(this_one, "after vertical levels subsampling"))

            # Filter variables according to OWC workflow
            # (I don't understand why this_one come at the end of the Matlab routine ...)
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L258
            this_one = this_one.argo.filter_scalib_pres(force=force, inplace=False)
            log.debug(pretty_print_count(this_one, "after pressure fields selection"))

            # Filter along some QC:
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L372
            this_one = this_one.argo.filter_qc(
                QC_list=[0, 1, 2], QC_fields=["TIME_QC"], drop=True
            )  # Matlab says to reject > 3
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L420
            this_one = this_one.argo.filter_qc(
                QC_list=[v for v in range(10) if v != 3],
                QC_fields=["PRES_QC"],
                drop=True,
            )  # Matlab says to keep != 3
            this_one = this_one.argo.filter_qc(
                QC_list=[v for v in range(10) if v != 4],
                QC_fields=["PRES_QC", "TEMP_QC", "PSAL_QC"],
                drop=True,
                mode="any",
            )  # Matlab says to keep != 4
            if len(this_one["N_POINTS"]) == 0:
                raise DataNotFound(
                    "All data have been discarded because either PSAL_QC or TEMP_QC is filled with 4 or"
                    " PRES_QC is filled with 3 or 4\n"
                    "NO SOURCE FILE WILL BE GENERATED !!!"
                )
            log.debug(pretty_print_count(this_one, "after QC filter"))

            # Exclude dummies
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L427
            this_one = (
                this_one.argo._where(this_one["PSAL"] <= 50, drop=True)
                .argo._where(this_one["PSAL"] >= 0, drop=True)
                .argo._where(this_one["PTEMP"] <= 50, drop=True)
                .argo._where(this_one["PTEMP"] >= -10, drop=True)
                .argo._where(this_one["PRES"] <= 6000, drop=True)
                .argo._where(this_one["PRES"] >= 0, drop=True)
            )
            if len(this_one["N_POINTS"]) == 0:
                raise DataNotFound(
                    "All data have been discarded because they are filled with values out of range\n"
                    "NO SOURCE FILE WILL BE GENERATED !!!"
                )
            log.debug(pretty_print_count(this_one, "after dummy values exclusion"))

            # Transform measurements to a collection of profiles for Matlab-like formation:
            this_one = this_one.argo.point2profile()

            # Subsample and align vertical levels (max 1 level every 10db):
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L208
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L451
            bins = np.arange(0.0, np.max(this_one["PRES"]) + 10.0, 10.0)
            this_one = this_one.argo.groupby_pressure_bins(
                bins=bins, select=select, axis="PRES"
            )
            log.debug(
                pretty_print_count(
                    this_one, "after vertical levels subsampling and re-alignment"
                )
            )

            # Compute fractional year:
            # https://github.com/euroargodev/dm_floats/blob/c580b15202facaa0848ebe109103abe508d0dd5b/src/ow_source/create_float_source.m#L334
            DATES = np.array(
                [toYearFraction(d) for d in pd.to_datetime(this_one[self._TNAME].values)]
            )[np.newaxis, :]

            # Read measurements:
            PRES = this_one["PRES"].values.T  # (mxn)
            TEMP = this_one["TEMP"].values.T  # (mxn)
            PTMP = this_one["PTEMP"].values.T  # (mxn)
            SAL = this_one["PSAL"].values.T  # (mxn)
            LAT = this_one["LATITUDE"].values[np.newaxis, :]
            LONG = this_one["LONGITUDE"].values[np.newaxis, :]
            LONG[0][np.argwhere(LONG[0] < 0)] = LONG[0][np.argwhere(LONG[0] < 0)] + 360
            PROFILE_NO = this_one["CYCLE_NUMBER"].values[np.newaxis, :]

            # Create dataset with preprocessed data:
            this_one_dsp_processed = xr.DataArray(
                PRES,
                dims=["m", "n"],
                coords={
                    "m": np.arange(0, PRES.shape[0]),
                    "n": np.arange(0, PRES.shape[1]),
                },
                name="PRES",
            ).to_dataset(promote_attrs=False)
            this_one_dsp_processed["TEMP"] = xr.DataArray(
                TEMP,
                dims=["m", "n"],
                coords={
                    "m": np.arange(0, TEMP.shape[0]),
                    "n": np.arange(0, TEMP.shape[1]),
                },
                name="TEMP",
            )
            this_one_dsp_processed["PTMP"] = xr.DataArray(
                PTMP,
                dims=["m", "n"],
                coords={
                    "m": np.arange(0, PTMP.shape[0]),
                    "n": np.arange(0, PTMP.shape[1]),
                },
                name="PTMP",
            )
            this_one_dsp_processed["SAL"] = xr.DataArray(
                SAL,
                dims=["m", "n"],
                coords={
                    "m": np.arange(0, SAL.shape[0]),
                    "n": np.arange(0, SAL.shape[1]),
                },
                name="SAL",
            )
            this_one_dsp_processed["PROFILE_NO"] = xr.DataArray(
                PROFILE_NO[0, :],
                dims=["n"],
                coords={"n": np.arange(0, PROFILE_NO.shape[1])},
                name="PROFILE_NO",
            )
            this_one_dsp_processed["DATES"] = xr.DataArray(
                DATES[0, :],
                dims=["n"],
                coords={"n": np.arange(0, DATES.shape[1])},
                name="DATES",
            )
            this_one_dsp_processed["LAT"] = xr.DataArray(
                LAT[0, :],
                dims=["n"],
                coords={"n": np.arange(0, LAT.shape[1])},
                name="LAT",
            )
            this_one_dsp_processed["LONG"] = xr.DataArray(
                LONG[0, :],
                dims=["n"],
                coords={"n": np.arange(0, LONG.shape[1])},
                name="LONG",
            )
            this_one_dsp_processed["m"].attrs = {"long_name": "vertical levels"}
            this_one_dsp_processed["n"].attrs = {"long_name": "profiles"}

            # Create Matlab dictionary with preprocessed data (to be used by savemat):
            mdata = ds2mat(this_one_dsp_processed)

            # Output
            log.debug("float source data saved in: %s" % this_path)
            if this_path is None:
                if debug_output:
                    return mdata, this_one_dsp_processed, this_one  # For debug/devel
                else:
                    return this_one_dsp_processed
            else:
                from scipy.io import savemat

                # Validity check of the path type is delegated to savemat
                return savemat(
                    this_path,
                    mdata,
                    appendmat=False,
                    format=format,
                    do_compression=do_compression,
                )

        # Run pre-processing for each float data
        output = {}
        for WMO in np.unique(this["PLATFORM_NUMBER"]):
            log.debug("> Preprocessing data for float WMO %i" % WMO)
            this_float = this.argo._where(this["PLATFORM_NUMBER"] == WMO, drop=True)
            if path is None:
                output[WMO] = preprocess_one_float(
                    this_float, this_path=path, select=select, debug_output=debug_output
                )
            else:
                os.makedirs(path, exist_ok=True)  # Make path exists
                float_path = os.path.join(
                    path, "%s%i%s.mat" % (file_pref, WMO, file_suff)
                )
                preprocess_one_float(
                    this_float,
                    this_path=float_path,
                    select=select,
                    debug_output=debug_output,
                )
                output[WMO] = float_path
        if path is None:
            log.debug("===================== END create_float_source")
            return output

    def list_N_PROF_variables(self, uid=False):
        """Return the list of variables with unique values along the N_PROF dimension"""
        this = self._obj  # Should not be modified

        # Find the number of profiles (N_PROF):
        dummy_argo_uid = self._dummy_argo_uid
        N_PROF = len(np.unique(dummy_argo_uid))

        # For each variable, determine if it has unique value by profile,
        # if yes: the transformed variable should be [N_PROF]
        # if no: the transformed variable should be [N_PROF, N_LEVELS]
        count = np.zeros((N_PROF, len(this.variables)), "int")
        for i_prof, grp in enumerate(this.groupby(dummy_argo_uid)):
            i_uid, prof = grp
            for iv, vname in enumerate(this.variables):
                try:
                    count[i_prof, iv] = len(np.unique(prof[vname]))
                except Exception as e:
                    print(
                        "An error happened when dealing with the '%s' data variable"
                        % vname
                    )
                    raise (e)

        # Variables with a single unique value for each profile:
        list_1d = list(np.array(this.variables)[count.sum(axis=0) == count.shape[0]])

        if not uid:
            return list_1d
        else:
            return list_1d, dummy_argo_uid

    @property
    def list_WMO_CYC(self):
        """Return a tuple with all (PLATFORM_NUMBER, CYCLE_NUMBER) in the dataset"""
        profiles = []
        for wmo, grp in self._obj.groupby("PLATFORM_NUMBER"):
            [profiles.append((wmo, cyc)) for cyc in np.unique(grp["CYCLE_NUMBER"])]
        return profiles

    @property
    def list_WMO(self):
        """Return all possible WMO as a list"""
        return to_list(np.unique(self._obj["PLATFORM_NUMBER"].values))

    def to_zarr(self, *args, **kwargs) -> Union[ZarrStore, Delayed]:
        """Write Argo dataset content to a zarr group

        Before write operation is delegated to :class:`xarray.Dataset.to_zarr`, we perform the following:

        - Ensure all variables are appropriately cast.
        - If the ``encoding`` argument is not specified, we automatically add a ``Blosc(cname="zstd", clevel=3, shuffle=2)`` compression to all variables. Set `encoding=None` for no compression.

        Parameters
        ----------
        *args, **kwargs:
            Passed to :class:`xarray.Dataset.to_zarr`.

        Returns
        -------
        The output from :class:`xarray.Dataset.to_zarr` call

        See Also
        --------
        :class:`xarray.Dataset.to_zarr`, :class:`numcodecs.blosc.Blosc`
        """

        # Ensure that all variables are cast appropriately
        # (those already cast are not changed)
        self._obj = self.cast_types()

        # Add zarr compression to encoding:
        if "encoding" not in kwargs:
            from numcodecs import Blosc
            compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
            encoding = {}
            for v in self._obj:
                encoding.update({v: {"compressor": compressor}})
            kwargs.update({'encoding': encoding})

        # Convert to a zarr file using compression:
        return self._obj.to_zarr(*args, **kwargs)

    def reduce_profile(self, func, params=[], **kwargs) -> xr.DataArray:
        """Apply a vectorized function for unlabeled arrays for each Argo profiles

        This method allows to execute a per profile diagnostic function very efficiently. Such a diagnostic function
        takes vertical profiles as input and return a single value as output (see examples below).

        Typical usage example would include computation of mixed layer depth or euphotic layer depth.

        Parameters
        ----------
        func: callable
            A function that takes one or more profile parameters as input, and return a single value as output.

        params: [List, str]
            Name, or list of names, of the dataset parameters expected by ``func``. All of these parameters
            must have ``N_LEVELS`` as a dimension.

        **kwargs: dict, optional
            Keyword arguments to be passed to ``func``.

        Returns
        -------
        :class:`xarray.DataArray`

        Examples
        --------
        .. code-block:: python
            :caption: Example 1

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            def max_salinity_depth(pres, psal):
                # A dummy function returning depth of the maximum salinity
                idx = ~np.logical_or(np.isnan(pres), np.isnan(psal))
                return pres[idx][np.argmax(psal[idx])]

            # Apply reduce function on all profiles:
            dsp.argo.reduce_profile(max_salinity_depth, params=['PRES', 'PSAL'])

        .. code-block:: python
            :caption: Example 2: with keyword arguments

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            def max_salinity_depth(pres, psal, max_layer=1000.):
                # A dummy function returning depth of the maximum salinity above max_layer:
                idx = ~np.logical_or(np.isnan(pres), np.isnan(psal))
                idx = np.logical_and(idx, pres<=max_layer)
                if np.any(idx):
                    return pres[idx][np.argmax(psal[idx])]
                else:
                    return np.NaN

            # Apply reduce function on all profiles:
            dsp.argo.reduce_profile(max_salinity_depth, params=['PRES', 'PSAL'], max_layer=700)

        .. code-block:: python
            :caption: Example 3: automatically parallelize func with Dask

            from argopy import ArgoFloat
            dsp = ArgoFloat(6901864).open_dataset('Sprof')

            # Make sure we're working with dask arrays
            dsp = dsp.chunk({'N_PROF': 10})

            def max_salinity_depth(pres, psal):
                # A dummy function returning depth of the maximum salinity
                idx = ~np.logical_or(np.isnan(pres), np.isnan(psal))
                return pres[idx][np.argmax(psal[idx])]

            # Apply reduce function on all profiles:
            da = dsp.argo.reduce_profile(max_salinity_depth, params=['PRES', 'PSAL'])  # Return a dask array
            da.compute()
        """
        if self._type != "profile":
            raise InvalidDatasetStructure(
                "Method only available for a collection of profiles (with N_PROF dimension)"
            )

        # plist holds the list of dataset variable(s) required by the reducer with a N_PROF dimension:
        # eg: ['PRES', 'TEMP']
        plist = to_list(params)
        for param in plist:
            if param not in self._obj:
                raise ValueError(f"Parameter {param} not in dataset")
            if 'N_LEVELS' not in self._obj[param].dims:
                raise ValueError(f"Parameter {param} must have the 'N_LEVELS' dimension")

        # There should be one input core dimension 'N_LEVELS' for each argument of the reduce function
        input_core_dims = [["N_LEVELS"] for _ in plist]

        # Create the reduce function list of arguments:
        ufunc_args = []
        [ufunc_args.append(self._obj[param]) for param in plist]

        # Create the xr.apply_ufunc list of keywords arguments:
        ufunc_kwargs = dict(
            kwargs=kwargs,  # Keywords arguments to be passed to the reduce function
            input_core_dims=input_core_dims,

            # dimensions allowed to change size. Must be set!
            # must also appear in ``input_core_dims`` for at least one argument
            exclude_dims=set(("N_LEVELS",)),

            vectorize=True,  # loop over non-core dims
            dask="parallelized",
        )
        reduced = xr.apply_ufunc(
            func,
            *ufunc_args,
            **ufunc_kwargs,
        )
        return reduced


def open_Argo_dataset(filename_or_obj):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)
    ds = xr.open_dataset(filename_or_obj, decode_cf=1, decode_times=time_coder, mask_and_scale=1, decode_timedelta=True)
    ds = cast_Argo_variable_type(ds)
    return ds


class ArgoEngine(BackendEntrypoint):
    """Backend for Argo netCDF files based on the xarray netCDF4 engine

    It can open any Argo ".nc" files with 'Argo' in their global attribute 'Conventions'.

    But it will not be detected as valid backend for netcdf files, so make
    sure to specify ``engine="argo"`` in :func:`xarray.open_dataset`.

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        ds = xr.open_dataset("dac/aoml/1901393/1901393_prof.nc", engine='argo')

    """

    description = "Open Argo netCDF files (.nc)"
    url = "https://argopy.readthedocs.io/en/latest/generated/argopy.xarray.ArgoEngine.html#argopy.xarray.ArgoEngine"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        return open_Argo_dataset(filename_or_obj)

    open_dataset_parameters = ["filename_or_obj"]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        if ext in {".nc"}:
            attrs = xr.open_dataset(filename_or_obj, engine="netcdf4").attrs
            return "Conventions" in attrs and "Argo" in attrs["Conventions"]
        else:
            return False
