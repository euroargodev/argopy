"""
If client is online (connected to the web) we work with the 'online' implementation
otherwise we fall back on an offline implementation.

The choice is really meaningful when the client is using a local host. In this case
we don't know if client intends to be online or offline, so we check and implement.

"""

import logging
import xarray as xr

from argopy.utils.checkers import isconnected
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
        :caption: Load/read GDAC netcdf files

        af.ls_dataset() # Return a dictionary with all available datasets for this float
        ds = af.open_dataset('prof') # Use keys from .ls_dataset()
        ds = af.open_dataset('meta')
        ds = af.open_dataset('tech')
        ds = af.open_dataset('Rtraj')
        ds = af.open_dataset('Sprof')

        ds = af.open_dataset('Sprof', netCDF4=True)  # Return a netCDF4 Dataset instead of an xarray

    .. code-block:: python
        :caption: Load/read GDAC netcdf mono-cycle profile files

        af.describe_profiles()  # Pandas DataFrame describing all available profile files

        af.lsp() # Return the full list of profile files

        af.ls_profiles() # Return a dictionary with all available mono-cycle profile files (everything under the 'profiles' sub-folder)

        # To load one single file, use keys from af.ls_profiles():
        ds = af.open_profile(12) # cycle number 12, core file
        ds = af.open_profile('1D') # cycle number 1, descending core file
        ds = af.open_profile('B15') # cycle number 15, BGC file
        ds = af.open_profile('B1D') # cycle number 1, descending BGC file
        ds = af.open_profile('S28') # cycle number 28, BGC synthetic file

        # To load one or more files, provide cycle number(s) and other attributes:
        ds_list = af.open_profiles([1,2,3])
        ds_list = af.open_profiles([1,2,3], direction='D')
        ds_list = af.open_profiles([1,2,3], dataset='B') # Return 'BGC' B files
        ds_list = af.open_profiles([1,2,3], dataset='B', direction='D') # Return 'BGC' B files, descending
        ds_list = af.open_profiles([1,2,3], dataset='S') # Return 'BGC' Synthetic files

        # If you don't specify cycle numbers, all cycles are loaded:
        ds_list = af.open_profiles(direction='D') # Return *all* core descending files


    .. code-block:: python
        :caption: Other attributes and methods

        af.N_CYCLES  # Number of unique cycle number (estimated)
        af.path  # root path for all float datasets
        af.dac   # name of the DAC this float belongs to
        af.metadata  # a dictionary with all available metadata for this file (from netcdf or fleetmonitoring API)
        af.ls()  # list af.path folder content


    .. code-block:: python
        :caption: Quick plotting methods

        af.plot.trajectory()
        af.plot.trajectory(figsize=(18,18), padding=[1, 5])
        af.plot.map('TEMP', pres=450, cmap='Spectral_r')
        af.plot.map('DATA_MODE')
        af.plot.scatter('TEMP')
        af.plot.scatter('PSAL_QC')
        af.plot.scatter('DOXY', ds='Sprof')
        af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Possibly register extensions:
    plot = xr.core.utils.UncachedAccessor(ArgoFloatPlot)
