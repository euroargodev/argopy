.. currentmodule:: argopy

.. _tools-argofloat:

Argo Float store
================

If you are familiar with Argo float WMO numbers, you may be interested in using directly the Argo float store :class:`ArgoFloat`.

This store aims to facilitate all Argo netcdf file load/read operations for a specific float. Whatever the Argo netcdf file location,
local or remote, you can now delegate to **argopy** the burden of transfer protocol and GDAC paths handling.

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
the instantiation of the class:

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

.. note::
    The :meth:`ArgoFloat.open_dataset` also support for direct file loading as a `netCDF4 Dataset object <https://unidata.github.io/netcdf4-python/#netCDF4.Dataset>`_. Just use the `netCDF4=True` option.

    .. ipython:: python
        :okwarning:

        af.open_dataset('meta', netCDF4=True)

Integration within **argopy**
-----------------------------

The :class:`ArgoFloat` class is further used in **argopy** in the :class:`ArgoIndex` iterator.

.. ipython:: python
    :okwarning:

    from argopy import ArgoIndex

    # Make a search on Argo index of profiles:
    idx = ArgoIndex().query.lon_lat([-70, -55, 20, 30], nrows=100)

    # Then iterate over ArgoFloat matching the results:
    for a_float in idx.iterfloats():
        ds = a_float.open_dataset('meta')
        print(a_float.WMO, ds['LAUNCH_DATE'].data)

.. _argofloat-visu:

Plotting features
-----------------
.. currentmodule:: argopy

The :class:`ArgoFloat` class come with a :class:`ArgoFloat.plot` accessor than can take several methods to quickly visualize data from the float:

Check all the detailed arguments on the API reference :class:`ArgoFloat.plot`.

.. tabs::

    .. tab:: Simple trajectory

        .. code-block:: python

                from argopy import ArgoFloat
                af = ArgoFloat(6903262)

                af.plot.trajectory()
                # af.plot.trajectory(figsize=(18,18), padding=[1, 5])

        .. image:: ../../_static/ArgoFloat_trajectory.png

    .. tab:: Data along trajectory

        .. code-block:: python

                from argopy import ArgoFloat
                af = ArgoFloat(6903262)

                af.plot.map('TEMP', pres=450, cmap='Spectral_r')

        .. image:: ../../_static/ArgoFloat_TEMP.png

        .. code-block:: python

                from argopy import ArgoFloat
                af = ArgoFloat(6903262)

                af.plot.map('PROFILE_PSAL_QC')

        .. image:: ../../_static/ArgoFloat_PROFILE_PSAL_QC.png

    .. tab:: Data as a function of pressure

        .. code-block:: python

                from argopy import ArgoFloat
                af = ArgoFloat(6903262)

                af.plot.scatter('TEMP')

        .. image:: ../../_static/ArgoFloat_TEMPscatter.png

        Plotting QC will automatically select the appropriate colormap:

        .. code-block:: python

                af.plot.scatter('PSAL_QC')

        .. image:: ../../_static/ArgoFloat_PSAL_QC.png

        Note that by default, variables are loaded from the `prof` netcdf dataset, but variables from other netcdf dataset can also be plotted if the appropriate dataset is indicated with the `ds` argument:

        .. code-block:: python

                from argopy import ArgoFloat
                af = ArgoFloat(6903262)

                af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

        .. image:: ../../_static/ArgoFloat_MEASUREMENT_CODE.png
