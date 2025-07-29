"""
If client is online (connected to the web) we work with the 'online' implementation
otherwise we fall back on an offline implementation.

The choice is really meaningful when the client is using a local host. In this case
we don't know if client intends to be online or offline, so we check and implement.

"""

import logging
import xarray as xr

from ...utils import isconnected
from .implementations.plot import ArgoFloatPlot


log = logging.getLogger("argopy.stores.ArgoFloat")


if isconnected():
    from .implementations.online.float import FloatStore

    log.info("Using ONLINE Argo Float implementation")
else:
    from .implementations.offline.float import FloatStore

    log.info("Using OFFLINE Argo Float implementation")


class ArgoFloat(FloatStore):
    """Argo GDAC float store

    This store makes it easy to load/read/visualize data for a given float from any GDAC location and netcdf files

    Examples
    --------
    .. code-block:: python
        :caption: A float store is instantiated with float WMO number and a host (any access path: local, http, ftp or s3) where float files are to be found.

        from argopy import ArgoFloat
        af = ArgoFloat(WMO)  # Use argopy 'gdac' option by default
        af = ArgoFloat(WMO, host='/home/ref-argo/gdac')  # Use your local GDAC copy
        af = ArgoFloat(WMO, host='http')   # Shortcut for https://data-argo.ifremer.fr
        af = ArgoFloat(WMO, host='ftp')    # shortcut for ftp://ftp.ifremer.fr/ifremer/argo
        af = ArgoFloat(WMO, host='s3')     # Shortcut for s3://argo-gdac-sandbox/pub

    .. code-block:: python
        :caption: Load/read GDAC netcdf files as a :class:`xarray.Dataset` or :class:`netCDF4.Dataset`

        af.ls_dataset() # Return a dictionary with all available datasets for this float
        ds = af.open_dataset('prof') # Use keys from the available datasets dictionary
        ds = af.open_dataset('meta')
        ds = af.open_dataset('tech')
        ds = af.open_dataset('Rtraj')
        ds = af.open_dataset('Sprof')

        ds = af.open_dataset('Sprof', netCDF4=True)  # Return a netCDF4 Dataset instead of an xarray

    .. code-block:: python
        :caption: Other attributes and methods

        af.N_CYCLES  # Number of cycles (estimated)
        af.path  # root path for all float datasets
        af.dac   # name of the DAC this float belongs to
        af.metadata  # a dictionary with all available metadata for this file (from netcdf or fleetmonitoring API)
        af.ls()  # list af.path folder content

    .. code-block:: python
        :caption: Working with float profiles

        af.lsprofiles() # list float "profiles" folder content
        af.describe_profiles()  # Pandas DataFrame describing all available float profile files

    .. code-block:: python
        :caption: Quick plotting methods

        af.plot.trajectory()
        af.plot.trajectory(figsize=(18,18), padding=[1, 5])
        af.plot.map('TEMP', pres=450, cmap='Spectral_r')
        af.plot.map('DATA_MODE', cbar=False, legend=True)
        af.plot.scatter('PSAL')
        af.plot.scatter('DOXY', ds='Sprof')
        af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Possibly register extensions:
    plot = xr.core.utils.UncachedAccessor(ArgoFloatPlot)
