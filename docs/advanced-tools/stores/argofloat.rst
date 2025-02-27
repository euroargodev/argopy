.. currentmodule:: argopy

.. _tools-argofloat:

Argo Float store
================

If you are familiar with Argo float WMO numbers, you may be interested in using directly the Argo float store :class:`ArgoFloat`.

This store aims to facilitate all Argo netcdf file load/read operations for a specific float. Whatever the Argo netcdf file location,
local or remote, you can now delegate to **argopy** the burden of transfer protocol and GDAC paths handling. This store is primarily
intended to be used by third party libraries or in workflow by operators and experts.

All float store methods and properties are documented in the :class:`ArgoFloat` API page.

The simplest use case may look like this:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    WMO = 6903091 # Use any float
    ds = ArgoFloat(WMO).open_dataset('prof')

This will trigger download and opening of the ``https://data-argo.ifremer.fr/dac/coriolis/6903091/6903091_prof.nc`` file. You should notice
that :class:`ArgoFloat` automatically determined in which DAC folder to find this float and constructed the appropriate path
toward the requested dataset ``prof``.

Float store creation
--------------------

If a specific host is not provided, :class:`ArgoFloat` will fetch float data from the ``gdac`` global option (which is set to the Ifremer http server by default), but you can use any valid GDAC host and possibly shortcuts as well:

.. ipython:: python
    :okwarning:


    af = ArgoFloat(WMO)
    # or:
    # af = ArgoFloat(WMO, host='/home/ref-argo/gdac')  # Use your local GDAC copy
    # af = ArgoFloat(WMO, host='https')  # Shortcut for https://data-argo.ifremer.fr
    # af = ArgoFloat(WMO, host='ftp')    # shortcut for ftp://ftp.ifremer.fr/ifremer/argo
    # af = ArgoFloat(WMO, host='s3')     # Shortcut for s3://argo-gdac-sandbox/pub


Note that in order to include dataset from the auxiliary GDAC folder, you need to specify it with the ``aux`` argument at
the instanciation of the class:

.. ipython:: python
    :okwarning:

    af = ArgoFloat(WMO, aux=True)
    af

List dataset and loading
------------------------

Once you created an :class:`ArgoFloat` instance, you can list all available dataset with:

.. ipython:: python
    :okwarning:

    af.ls_dataset()

Note that dataset from the auxiliary GDAC folder are included in this store, and referenced with the `_aux` suffix.

So finally, you can open any of these dataset using their keyword:

.. ipython:: python
    :okwarning:

    ds = af.open_dataset('meta') # load <WMO>_meta.nc
    # or:
    # ds = af.open_dataset('prof') # load <WMO>_prof.nc
    # ds = af.open_dataset('tech') # load <WMO>_tech.nc
    # ds = af.open_dataset('Rtraj') # load <WMO>_Rtraj.nc

Note that you can open a dataset lazily, this is explained in the :ref:`lazy-argofloat` documentation page.


Integration within **argopy**
-----------------------------

The :class:`ArgoFloat` class is further used in **argopy** in the :class:`ArgoIndex` iterator.

.. ipython:: python
    :okwarning:

    from argopy import ArgoIndex

    # Make a search on Argo index of profiles:
    idx = ArgoIndex().search_lat_lon([-70, -55, 20, 30], nrows=100)

    # Then iterate over floats matching the results:
    for float in idx.iterfloats():
        # 'float' is an ArgoFloat instance
        ds = float.open_dataset('meta')
        print(float.WMO, ds['LAUNCH_DATE'].data)
