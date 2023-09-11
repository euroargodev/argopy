"""
Manipulate/transform xarray objects or list of objects
"""
import numpy as np
import xarray as xr
import logging


log = logging.getLogger("argopy.utils.manip")


def drop_variables_not_in_all_datasets(ds_collection):
    """Drop variables that are not in all datasets (the lowest common denominator)

    Parameters
    ----------
    list of :class:`xr.DataSet`

    Returns
    -------
    list of :class:`xr.DataSet`
    """

    # List all possible data variables:
    vlist = []
    for res in ds_collection:
        [vlist.append(v) for v in list(res.data_vars)]
    vlist = np.unique(vlist)

    # Check if each variables are in each datasets:
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
        log.debug("Dropping these variables that are missing from some dataset in this list: %s" % vlist[iv_missing])

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


def fill_variables_not_in_all_datasets(ds_collection, concat_dim='rows'):
    """Add empty variables to dataset so that all the collection have the same data_vars and coords

    This is to make sure that the collection of dataset can be concatenated

    Parameters
    ----------
    list of :class:`xr.DataSet`

    Returns
    -------
    list of :class:`xr.DataSet`
    """
    def first_variable_with_concat_dim(this_ds, concat_dim='rows'):
        """Return the 1st variable in the collection that have the concat_dim in dims"""
        first = None
        for v in this_ds.data_vars:
            if concat_dim in this_ds[v].dims:
                first = v
                pass
        return first

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
                meta[v] = {'attrs': ds[v].attrs, 'dtype': ds[v].dtype, 'fill_value': fillvalue(ds[v])}
    # [log.debug(meta[m]) for m in meta.keys()]

    # Add missing variables to dataset
    datasets = [ds.copy() for ds in ds_collection]
    for ir, ds in enumerate(datasets):
        for v in vlist:
            if v not in ds.variables:
                like = ds[first_variable_with_concat_dim(ds, concat_dim=concat_dim)]
                datasets[ir][v] = xr.full_like(like, fill_value=meta[v]['fill_value'], dtype=meta[v]['dtype'])
                datasets[ir][v].attrs = meta[v]['attrs']

    # Make sure that all datasets have the same set of coordinates
    results = []
    for ir, ds in enumerate(datasets):
        results.append(datasets[ir].set_coords(clist))

    #
    return results
