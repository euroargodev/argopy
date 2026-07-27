.. currentmodule:: argopy

.. _tools-argofloat:

Argo Float store
================

If you are familiar with Argo float WMO numbers, you may be interested in using directly the Argo float store :class:`ArgoFloat`.

This store aims to facilitate all Argo files load and read operations for a specific float.
Whatever the Argo file location, local or remote, you can delegate to **Argopy** the burden of transfer protocol and GDAC paths handling.

All float store methods and properties are documented in the :class:`ArgoFloat` API page.

The simplest use case may look like this:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    WMO = 3902492 # Use any float
    ds = ArgoFloat(WMO).open_dataset('prof')

This will trigger download and opening of the ``https://data-argo.ifremer.fr/dac/bodc/3902492/3902492_prof.nc`` file. You should notice that :class:`ArgoFloat` automatically determined in which DAC folder to find this float and constructed the appropriate path toward the requested dataset ``prof``.

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


Note that in order to include datasets from the auxiliary GDAC folder, you need to specify it with the ``aux`` argument at
the instantiation of the class:

.. ipython:: python
    :okwarning:

    af = ArgoFloat(WMO, aux=True)
    af

List and load datasets
----------------------

.. note::
    We refer to a *dataset* any file that is *not* under the float ``profiles`` folder of the GDAC.

Listing
^^^^^^^

Once you created an :class:`ArgoFloat` instance, you can list all available datasets with :meth:`ArgoFloat.ls_datasets`:

.. ipython:: python
    :okwarning:

    af.ls_datasets()

Note that because we used ``aux=True``, datasets from the auxiliary GDAC folder are included in this store, and referenced with the ``_aux`` suffix.

Data loading
^^^^^^^^^^^^

So finally, you can open any of these datasets using the corresponding keyword:

.. ipython:: python
    :okwarning:

    ds = af.open_dataset('prof') # load <WMO>_prof.nc
    print(ds)

    # or:
    # ds = af.open_dataset('meta') # load <WMO>_meta.nc
    # ds = af.open_dataset('tech') # load <WMO>_tech.nc
    # ds = af.open_dataset('Rtraj') # load <WMO>_Rtraj.nc

.. note::
    The :meth:`ArgoFloat.open_dataset` also support for direct file loading as a `netCDF4 Dataset object <https://unidata.github.io/netcdf4-python/#netCDF4.Dataset>`_. Just use the ``netCDF4=True`` option.

    .. ipython:: python
        :okwarning:

        af.open_dataset('meta', netCDF4=True)


List and load profiles
----------------------

.. note::
    We refer to a *profile* any file that *is* under the float ``profiles`` folder of the GDAC and is technically a mono-profile file.

Listing
^^^^^^^

Once you created an :class:`ArgoFloat` instance, you can list all available profile files with :meth:`ArgoFloat.ls_profiles`:


.. ipython:: python
    :okwarning:

    af.ls_profiles()

This method return a dictionary with all available mono-cycle profile files (everything under the ``profiles`` sub-folder).
For each profile file, there is a key to refer to it and to be used by :meth:`ArgoFloat.open_profile` (see below). Keys follow this convention:

* keys are integer for 'core' and ascending profile files (eg: 12 for 'R6903076_012.nc'),
* keys are string for all other profile files, with the following convention:

    * ends with a 'D' for 'core' descending profile files (eg: '1D' for 'R6903076_001D.nc'),
    * starts with a 'B' for BGC ascending profile files (eg: 'B12' for 'BD6903091_012.nc'),
    * starts with a 'B' and ends with a 'D' for BGC descending profile files (eg: 'B12D' for 'BD6903091_012D.nc'),
    * starts with a 'S' for Synthetic profile files (eg: 'S134' for 'S6903091_134.nc').
    * starts with a 'S' and ends with a 'D' for Synthetic descending profile files (eg: 'S2D' for 'SR3902492_002D.nc').

* Data from the auxiliary folder have regular keys with ``aux`` appended at the end of the key (eg: '11aux' for 'aux/coriolis/2903797/profiles/R2903797_011_aux.nc').

Note that since mono-cycle profile files are either 'R' for real-time or 'D' for adjusted or delayed-mode data, there is no need to select one or the other, they can't exist at the same time.

A more verbose description of all available profiles is provided with the :meth:`ArgoFloat.profiles_to_dataframe` method:

.. ipython:: python
    :okwarning:

    af.profiles_to_dataframe()

Data loading
^^^^^^^^^^^^

For a single cycle
""""""""""""""""""

To load a single mono-profile file, one will use the :meth:`ArgoFloat.open_profile` method with one of the key returned by :meth:`ArgoFloat.ls_profiles`:


.. ipython:: python
    :okwarning:

    ds = af.open_profile(6) # cycle number 6, core ascending file

    # or:
    ds = af.open_profile('1D') # cycle number 1, descending core file
    ds = af.open_profile('B4') # cycle number 4, BGC file
    ds = af.open_profile('B2D') # cycle number 2, descending BGC file
    ds = af.open_profile('S8') # cycle number 8, BGC Synthetic file
    ds = af.open_profile('S2D') # cycle number 2, BGC Synthetic descending file


For a collection of cycles
""""""""""""""""""""""""""

You may also need to load a collection of profile files, for a selection of cycles (or all), for a specific dataset (eg: all Synthetic files) and for a specific direction (ascendant or descend).

To do so, **Argopy** provide the :meth:`ArgoFloat.open_profiles` method.

To load a specific list of cycle numbers, just provide them as a first argument. If no other named arguments are provided, this will load (in parallel) all 'core' and ascending mono-profile files for the specified cycle numbers:

.. ipython:: python
    :okwarning:

        ds_list = af.open_profiles([1,2])

The method will return a list of :class:`xarray.Dataset`, one for each of the cycle numbers. Since data loading is done in parallel to improve performances, there is no guarantee that datasets will be ordered similarly to the list of cycle numbers.

Also note that if you don't provide cycle numbers, by default **Argopy** will load all of them (which can be time consuming for some floats).

If you want to load only descending profile files, use the ``direction`` argument:

.. ipython:: python
    :okwarning:

        ds_list = af.open_profiles([1,2], direction='D')

If you want to load only BGC ``B`` profile files, use the ``dataset`` argument (use ``S`` for BGC Synthetic files):

.. ipython:: python
    :okwarning:

        ds_list = af.open_profiles([1,2,3], dataset='B')

Note that the ``dataset`` and ``direction`` arguments can be used together and the availability of the ``progress=True`` argument to get a visual feedback of the processing.


.. tip::

    If you get lost or want to double check on the netcdf files that are loaded, you can check the absolute path of the data source in the `encoding` attribute of a :class:`xarray.Dataset`:

    .. ipython:: python
        :okwarning:

        ds_list[0].encoding['source']


Cycles batch processing
^^^^^^^^^^^^^^^^^^^^^^^

You can provide a pre-processing function to the :meth:`ArgoFloat.open_profiles` method that will be applied to each of the mono-profile files before being returned. Since there is no concatenation at the end of the process (the method return a list), the pre-processing function can return a modified dataset or anything else.

For instance we could want to gather some properties of each profile in a dictionary to create a dataframe with the end results:

.. ipython:: python
    :okwarning:

    import numpy as np

    def ds2dict(ds_profile):
        return {'cycle_number': ds_profile['CYCLE_NUMBER'].values[0],
            'posqc': ds_profile['POSITION_QC'].values[0],
            'time': ds_profile['JULD'].values[0],
            'lon': ds_profile['LONGITUDE'].values[0],
            'lat': ds_profile['LATITUDE'].values[0],
            'max_pres': ds_profile['PRES'].max().values[np.newaxis][0],
           }

    data = af.open_profiles(direction='A', preprocess=ds2dict)
    data[0]

We can finally create a :class:`pandas.DataFrame`:

.. ipython:: python
    :okwarning

    import pandas as pd

    df = pd.DataFrame(data).sort_values(by='cycle_number').reset_index(drop=1)
    df

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


Support for configuration parameters access
-------------------------------------------

The :class:`ArgoFloat` object comes with 2 extensions dedicated to easily access/read configuration parameters in operation and at launch-time.

When Argopy is **offline**, parameters reading rely on a local meta data netcdf file, and when Argopy is **online**, it relies on the Euro-Argo Fleet-Monitoring web-API.


Operational configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

These are facilitated access to the ``CONFIG_PARAMETER_NAME`` and ``CONFIG_PARAMETER_VALUE`` netcdf parameters.

Get list of parameters and missions:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    af = ArgoFloat(3902492)

    # Total number and list of configuration parameters:
    af.config.n_params
    af.config.parameters

    # Total number and list of missions:
    af.config.n_missions
    af.config.missions

    # Get a dictionary mapping cycle on mission numbers:
    af.config.cycles


Or directly read parameter values:

.. ipython:: python
    :okwarning:

    # Read one parameter value, with explicit or implicit parameter name:
    # ('CONFIG_' is not mandatory, but string is case-sensitive)
    af.config['CONFIG_CycleTime_seconds']
    af.config['CycleTime_seconds']

    # Read parameter value for one or more mission numbers:
    # (! 2nd index is not 0-based, it's an integer key to look for in mission numbers)
    af.config['CycleTime_seconds', 1]
    af.config['CycleTime_seconds', 1:3]

    # Read parameter value for one or more cycle numbers:
    # (! 2nd index is not 0-based, it's an integer key to look for in cycle numbers)
    af.config.for_cycles('CycleTime_seconds', 1)
    af.config.for_cycles('CycleTime_seconds', [5, 6])

And parameters can also be exported to :class:`pandas.DataFrame`:

.. ipython:: python
    :okwarning:

    af.config.to_dataframe()
    # or:
    # af.config.to_dataframe(missions=1)
    # af.config.to_dataframe(missions=[1, 2])


Launch-time configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

These are facilitated access to the ``LAUNCH_CONFIG_PARAMETER_NAME`` and ``LAUNCH_CONFIG_PARAMETER_VALUE`` netcdf parameters.

Get list of parameters:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    af = ArgoFloat(3902492)

    # Total number and list of launch parameters:
    af.launchconfig.n_params
    af.launchconfig.parameters

Or directly read parameter values:

.. ipython:: python
    :okwarning:

    # Read one parameter value, with explicit or implicit parameter name:
    # ('CONFIG_' is not mandatory, but string is case-sensitive)
    af.launchconfig['CONFIG_CycleTime_seconds']
    af.launchconfig['CycleTime_seconds']

And launch parameters can also be exported to :class:`pandas.DataFrame`:

.. ipython:: python
    :okwarning:

    af.launchconfig.to_dataframe()



.. note::

    Note the **tab completion for parameter names** is available when executed with ipython (eg: jupyter notebooks) to easily get parameter names: just press tab when typing ``af.launchconfig['`` and ``af.config['``:

    .. image:: https://github.com/user-attachments/assets/826e073f-1685-4e48-88bc-991fc90aae74
        :width: 592

    .. image:: https://github.com/user-attachments/assets/1abd8516-3467-45e8-964f-07b4a80bf91e
        :width: 414

