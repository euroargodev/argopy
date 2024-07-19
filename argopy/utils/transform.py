"""
Manipulate/transform xarray objects or list of objects
"""
import numpy as np
import xarray as xr
import logging
from typing import List, Union

from ..errors import InvalidDatasetStructure


log = logging.getLogger("argopy.utils.manip")


def drop_variables_not_in_all_datasets(ds_collection: List[xr.Dataset]) -> List[xr.Dataset]:
    """Drop variables that are not in all datasets (the lowest common denominator)

    Parameters
    ----------
    ds_collection: List[xarray.Dataset]
        A list of :class:`xarray.Dataset`

    Returns
    -------
    List[xarray.Dataset]
    """

    # List all possible data variables:
    vlist = []
    for res in ds_collection:
        [vlist.append(v) for v in list(res.data_vars)]
    vlist = np.unique(vlist)

    # Check if each variable are in each dataset:
    ishere = np.zeros((len(vlist), len(ds_collection)))
    for ir, res in enumerate(ds_collection):
        for iv, v in enumerate(res.data_vars):
            for iu, u in enumerate(vlist):
                if v == u:
                    ishere[iu, ir] = 1

    # List of dataset with missing variables:
    # ir_missing = np.sum(ishere, axis=0) < len(vlist)
    # List of variables missing in some dataset:
    iv_missing = np.sum(ishere, axis=1) < len(ds_collection)
    if len(iv_missing) > 0:
        log.debug(
            "Dropping these variables that are missing from some dataset in this list: %s"
            % vlist[iv_missing]
        )

    # List of variables to keep
    iv_tokeep = np.sum(ishere, axis=1) == len(ds_collection)
    for ir, res in enumerate(ds_collection):
        #         print("\n", res.attrs['Fetched_uri'])
        v_to_drop = []
        for iv, v in enumerate(res.data_vars):
            if v not in vlist[iv_tokeep]:
                v_to_drop.append(v)
        ds_collection[ir] = ds_collection[ir].drop_vars(v_to_drop)
    return ds_collection


def fill_variables_not_in_all_datasets(
    ds_collection: List[xr.Dataset], concat_dim: str = "rows"
) -> List[xr.Dataset]:
    """Add empty variables to dataset so that all the collection have the same :attr:`xarray.Dataset.data_vars` and :props:`xarray.Dataset.coords`

    This is to make sure that the collection of dataset can be concatenated

    Parameters
    ----------
    ds_collection: List[xarray.Dataset]
        A list of :class:`xarray.Dataset`
    concat_dim: str, default='rows'
        Name of the dimension to use to create new variables. Typically, this is the name of the dimension the collection will be concatenated along afterward.

    Returns
    -------
    List[xarray.Dataset]
    """

    def first_variable_with_concat_dim(this_ds, concat_dim="rows"):
        """Return the 1st variable in the collection that have the concat_dim in dims"""
        first = None
        for v in this_ds.data_vars:
            if concat_dim in this_ds[v].dims:
                first = v
                pass
        return first

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

    # List all possible data variables:
    vlist = []
    for res in ds_collection:
        [vlist.append(v) for v in list(res.variables) if concat_dim in res[v].dims]
    vlist = np.unique(vlist)
    # log.debug('variables', vlist)

    # List all possible coordinates:
    clist = []
    for res in ds_collection:
        [clist.append(c) for c in list(res.coords) if concat_dim in res[c].dims]
    clist = np.unique(clist)
    # log.debu('coordinates', clist)

    # Get the first occurrence of each variable, to be used as a template for attributes and dtype
    meta = {}
    for ir, ds in enumerate(ds_collection):
        for v in vlist:
            if v in ds.variables:
                meta[v] = {
                    "attrs": ds[v].attrs,
                    "dtype": ds[v].dtype,
                    "fill_value": fillvalue(ds[v]),
                }
    # [log.debug(meta[m]) for m in meta.keys()]

    # Add missing variables to dataset
    datasets = [ds.copy() for ds in ds_collection]
    for ir, ds in enumerate(datasets):
        for v in vlist:
            if v not in ds.variables:
                like = ds[first_variable_with_concat_dim(ds, concat_dim=concat_dim)]
                datasets[ir][v] = xr.full_like(
                    like, fill_value=meta[v]["fill_value"], dtype=meta[v]["dtype"]
                )
                datasets[ir][v].attrs = meta[v]["attrs"]

    # Make sure that all datasets have the same set of coordinates
    results = []
    for ir, ds in enumerate(datasets):
        results.append(datasets[ir].set_coords(clist))

    #
    return results


def merge_param_with_param_adjusted(ds: xr.Dataset, param: str, errors: str = 'raise') -> xr.Dataset:
    """Copy <PARAM>_ADJUSTED values onto <PARAM> for points where param data mode is 'A' or 'D'

    After values have been copied, all <PARAM>_ADJUSTED* variables are dropped to avoid confusion.

    For core and deep datasets (ds='phy'), we use the ``<DATA>_MODE`` variable.

    For the bgc dataset (ds='bgc'), we use the ``<PARAM>_DATA_MODE`` variables.

    The type of dataset is inferred automatically.

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        Dataset to transform
    param: str
        Name of the parameter to merge
    errors: str, optional, default='raise'
        If 'raise': raises a InvalidDatasetStructure error if any of the expected dataset variables is
        not found.
        If 'ignore', fails silently and return unmodified dataset.

    Returns
    -------
    :class:`xarray.Dataset`

    """
    if "%s_ADJUSTED" % param not in ds:
        if errors == 'raise':
            raise InvalidDatasetStructure("Parameter '%s_ADJUSTED' adjusted values not found in this dataset" % param)
        else:
            return ds
    if ds.argo._type != "point":
        raise InvalidDatasetStructure(
            "Method only available to a collection of points"
        )

    core_ds = False
    if "%s_DATA_MODE" % param not in ds and param in ['PRES', 'TEMP', 'PSAL']:
        if "DATA_MODE" not in ds:
            if errors == 'raise':
                    raise InvalidDatasetStructure(
                    "Parameter '%s' data mode not found in this dataset (no 'DATA_MODE')" % param
                )
            else:
                return ds
        else:
            core_ds = True
            # Create a bgc-like parameter data mode variable:
            ds["%s_DATA_MODE" % param] = ds["DATA_MODE"].copy()
            # that will be dropped at the end of the process

    if param not in ds:
        ds[param] = ds["%s_ADJUSTED" % param].copy()
    if "%s_QC" % param not in ds and "%s_ADJUSTED_QC" % param in ds:
        ds["%s_QC" % param] = ds["%s_ADJUSTED_QC" % param].copy()
    if "%s_ERROR" % param not in ds and "%s_ADJUSTED_ERROR" % param in ds:
        ds["%s_ERROR" % param] = ds["%s_ADJUSTED_ERROR" % param].copy()

    ii_measured = np.logical_or.reduce((ds["%s_DATA_MODE" % param] == 'R',
                                        ds["%s_DATA_MODE" % param] == 'A',
                                        ds["%s_DATA_MODE" % param] == 'D'))
    ii_missing = np.logical_and.reduce((ds["%s_DATA_MODE" % param] != 'R',
                                        ds["%s_DATA_MODE" % param] != 'A',
                                        ds["%s_DATA_MODE" % param] != 'D'))
    assert ii_measured.sum() + ii_missing.sum() == len(ds['N_POINTS']), "Unexpected data mode values !"

    ii_measured_adj = np.logical_and.reduce((ii_measured,
                                             np.logical_or.reduce((ds["%s_DATA_MODE" % param] == 'A',
                                                                   ds["%s_DATA_MODE" % param] == 'D'))
                                             ))

    # Copy param_adjusted values onto param indexes where data_mode is in 'a' or 'd':
    ds["%s" % param].loc[dict(N_POINTS=ii_measured_adj)] = ds["%s_ADJUSTED" % param].loc[
        dict(N_POINTS=ii_measured_adj)]
    ds = ds.drop_vars(["%s_ADJUSTED" % param])

    if "%s_ADJUSTED_QC" % param in ds and "%s_ADJUSTED_QC" % param in ds:
        ds["%s_QC" % param].loc[dict(N_POINTS=ii_measured_adj)] = ds["%s_ADJUSTED_QC" % param].loc[
            dict(N_POINTS=ii_measured_adj)]
        ds = ds.drop_vars(["%s_ADJUSTED_QC" % param])

    if "%s_ERROR" % param in ds and "%s_ADJUSTED_ERROR" % param in ds:
        ds["%s_ERROR" % param].loc[dict(N_POINTS=ii_measured_adj)] = ds["%s_ADJUSTED_ERROR" % param].loc[
            dict(N_POINTS=ii_measured_adj)]
        ds = ds.drop_vars(["%s_ADJUSTED_ERROR" % param])

    if core_ds:
        ds = ds.drop_vars(["%s_DATA_MODE" % param])

    return ds


def filter_param_by_data_mode(ds: xr.Dataset,
                              param: str,
                              dm: Union[str, List[str]] = ['R', 'A', 'D'],
                              mask: bool = False,
                              errors: str = 'raise') -> xr.Dataset:
    """Filter measurements according to a parameter data mode

    Filter the dataset to keep points where a parameter is in any of the data mode specified.

    This method can return the filtered dataset or the filter mask.

    Notes
    -----
    - Method compatible with core, deep and BGC datasets
    - Can be applied after the :class:`xarray.Dataset.transform_data_mode`

    Parameters
    ----------
    ds: :class:`xarray.Dataset`
        The dataset to work filter
    param: str
        Name of the parameter to apply the filter to
    dm: str, list(str), optional, default=['R', 'A', 'D']
        List of DATA_MODE values (string) to keep
    mask: bool, optional, default=False
        Determine if we should return the filter mask or the filtered dataset
    errors: str, optional, default='raise'
        If ``raise``, raises a InvalidDatasetStructure error if any of the expected variables is
        not found.
        If ``ignore``, fails silently and return unmodified dataset.

    Returns
    -------
    :class:`xarray.Dataset`
    """

    core_ds = False
    if "%s_DATA_MODE" % param not in ds and param in ['PRES', 'TEMP', 'PSAL']:
        if "DATA_MODE" not in ds:
            if errors == 'raise':
                raise InvalidDatasetStructure(
                    "Parameter '%s' data mode not found in this dataset (no 'DATA_MODE')" % param
                )
            else:
                return ds
        else:
            core_ds = True
            # Create a bgc-like parameter data mode variable:
            ds["%s_DATA_MODE" % param] = ds["DATA_MODE"].copy()
            # that will be dropped at the end of the process

    filter = []
    for this_dm in dm:
        vname = "%s_DATA_MODE" % param
        if vname not in ds:
            log.warning("The parameter '%s' has no associated data mode" % vname)
        else:
            filter.append(ds[vname] == "%s" % this_dm.upper())

    if len(filter) > 0:
        filter = np.logical_or.reduce(filter)

    if core_ds:
        ds = ds.drop_vars(["%s_DATA_MODE" % param])

    if mask:
        return filter
    else:
        return ds.loc[dict(N_POINTS=filter)] if len(filter) > 0 else ds


def split_data_mode(ds: xr.Dataset) -> xr.Dataset:
    """Convert PARAMETER_DATA_MODE(N_PROF, N_PARAM) into several <PARAM>_DATA_MODE(N_PROF) variables

    Using the list of *PARAM* found in ``STATION_PARAMETERS``, this method will create ``N_PARAM``
    new variables in the dataset ``*PARAM*_DATA_MODE(N_PROF)``.

    The variable ``PARAMETER_DATA_MODE`` is drop from the dataset at the end of the process.

    Returns
    -------
    :class:`xr.Dataset`
    """
    if "STATION_PARAMETERS" in ds and "PARAMETER_DATA_MODE" in ds:

        u64 = lambda s: "%s%s" % (s, " " * (64 - len(s)))
        params = [p.strip() for p in np.unique(ds['STATION_PARAMETERS'])]

        for param in params:
            name = "%s_DATA_MODE" % param.replace("_PARAMETER", "").replace("PARAMETER_", "")
            mask = ds['STATION_PARAMETERS'] == xr.full_like(ds['STATION_PARAMETERS'], u64(param),
                                                            dtype=ds['STATION_PARAMETERS'].dtype)
            da = ds['PARAMETER_DATA_MODE'].where(mask, drop=True).isel(N_PARAM=0)
            da = da.rename(name)
            da = da.astype(ds['PARAMETER_DATA_MODE'].dtype)
            ds[name] = da

        ds = ds.drop_vars('PARAMETER_DATA_MODE')
        ds.argo.add_history("Transformed with 'split_data_mode'")

    return ds
