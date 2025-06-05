Manipulating data
=================

.. contents::
   :local:

.. currentmodule:: xarray

Once you fetched data, **argopy** comes with a handy :class:`xarray.Dataset` accessor ``argo`` to perform specific manipulation of the data. This means that if your dataset is named `ds`, then you can use `ds.argo` to access more **argopy** functions. The full list is available in the API documentation page :ref:`Dataset.argo (xarray accessor)`.

Let's start with standard import:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher

Transformation
--------------

Points vs profiles
^^^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

By default, fetched data are returned as a 1D array collection of measurements:

.. ipython:: python
    :okwarning:

    f = DataFetcher().region([-75,-55,30.,40.,0,100., '2011-01-01', '2011-01-15'])
    ds_points = f.data
    ds_points

If you prefer to work with a 2D array collection of vertical profiles, simply transform the dataset with :meth:`Dataset.argo.point2profile`:

.. ipython:: python
    :okwarning:

    ds_profiles = ds_points.argo.point2profile()
    ds_profiles

You can simply reverse this transformation with the :meth:`Dataset.argo.profile2point`:

.. ipython:: python
    :okwarning:

    ds = ds_profiles.argo.profile2point()
    ds

Pressure levels: Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

Once your dataset is a collection of vertical **profiles**, you can interpolate variables on standard pressure levels using :meth:`Dataset.argo.interp_std_levels` with your levels as input:

.. ipython:: python
    :okwarning:

    ds_interp = ds_profiles.argo.interp_std_levels([0,10,20,30,40,50])
    ds_interp

Note on the linear interpolation process : 
    - Only profiles that have a maximum pressure higher than the highest standard level are selected for interpolation.
    - Remaining profiles must have at least five data points to allow interpolation.
    - For each profile, shallowest data point is repeated to the surface to allow a 0 standard level while avoiding extrapolation.

Pressure levels: Group-by bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

If you prefer to avoid interpolation, you can opt for a pressure bins grouping reduction using :meth:`Dataset.argo.groupby_pressure_bins`. This method can be used to subsample and align an irregular dataset (pressure not being similar in all profiles) on a set of pressure bins. The output dataset could then be used to perform statistics along the N_PROF dimension because N_LEVELS will corresponds to similar pressure bins.

To illustrate this method, let's start by fetching some data from a low vertical resolution float:

.. ipython:: python
    :okwarning:

    f = DataFetcher(src='erddap', mode='expert').float(2901623)  # Low res float
    ds = f.data

Let's now sub-sample these measurements along 250db bins, selecting values from the **deepest** pressure levels for each bins:

.. ipython:: python
    :okwarning:

    bins = np.arange(0.0, np.max(ds["PRES"]), 250.0)
    ds_binned = ds.argo.groupby_pressure_bins(bins=bins, select='deep')
    ds_binned

See the new ``STD_PRES_BINS`` variable that hold the pressure bins definition.

The figure below shows the sub-sampling effect:

.. code-block:: python

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cmocean

    fig, ax = plt.subplots(figsize=(18,6))
    ds.plot.scatter(x='CYCLE_NUMBER', y='PRES', hue='PSAL', ax=ax, cmap=cmocean.cm.haline)
    plt.plot(ds_binned['CYCLE_NUMBER'], ds_binned['PRES'], 'r+')
    plt.hlines(bins, ds['CYCLE_NUMBER'].min(), ds['CYCLE_NUMBER'].max(), color='k')
    plt.hlines(ds_binned['STD_PRES_BINS'], ds_binned['CYCLE_NUMBER'].min(), ds_binned['CYCLE_NUMBER'].max(), color='r')
    plt.title(ds.attrs['Fetched_constraints'])
    plt.gca().invert_yaxis()

.. image:: ../../_static/groupby_pressure_bins_select_deep.png

The bin limits are shown with horizontal red lines, the original data are in the background colored scatter and the group-by pressure bins values are highlighted in red marks

The ``select`` option can take many different values, see the full documentation of :meth:`Dataset.argo.groupby_pressure_bins` , for all the details. Let's show here results from the ``random`` sampling:

.. code-block:: python

    ds_binned = ds.argo.groupby_pressure_bins(bins=bins, select='random')

.. image:: ../../_static/groupby_pressure_bins_select_random.png


Filters
^^^^^^^

.. currentmodule:: xarray

If you fetched data with the ``expert`` mode, you may want to use *filters* to help you curate the data.

- **QC flag filter**: :meth:`Dataset.argo.filter_qc`. This method allows you to filter measurements according to QC flag values. This filter modifies all variables of the dataset.
- **Data mode filter**: :meth:`Dataset.argo.datamode.filter`. This method allows you to filter variables according to their data mode.
- **OWC variables filter**: :meth:`Dataset.argo.filter_scalib_pres`. This method allows you to filter variables according to OWC salinity calibration software requirements. This filter modifies pressure, temperature and salinity related variables of the dataset.


Complementary data
------------------

TEOS-10 variables
^^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

You can compute additional ocean variables from `TEOS-10 <http://teos-10.org/>`_. The default list of variables is: 'SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP' ('SOUND_SPEED', 'CNDC' are optional). `Simply raise an issue to add a new one <https://github.com/euroargodev/argopy/issues/new/choose>`_.

This can be done using the :meth:`Dataset.argo.teos10` method and indicating the list of variables you want to compute:

.. ipython:: python
    :okwarning:

    ds = DataFetcher().float(2901623).to_xarray()
    ds.argo.teos10(['SA', 'CT', 'PV'])

.. ipython:: python
    :okwarning:

    ds['SA']

.. _complement-canyon-med:

Nutrient and carbonate system variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

For BGC, it may be possible to complement a dataset with predictions of the water-column nutrient concentrations and carbonate system variables.

This is currently possible for data located in the Mediterranean Sea using the the CANYON-MED model. CANYON-MED is a Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea [1]_ [2]_. When using this method, please cite the paper [1]_.

This model is available in **argopy** as an extension to the ``argo`` accessor: :class:`Dataset.argo.canyon_med`. It can be used to predict PO4, NO3, DIC, SiOH4, AT and pHT.

As an example, let's load one float data with oxygen measurements:

.. ipython:: python
    :okwarning:

    ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').float(1902605)
    ds = ArgoSet.to_xarray()

We can then predict all possible variables:

.. ipython:: python
    :okwarning:

    ds.argo.canyon_med.predict()

or select variables to predict, like PO4:

.. ipython:: python
    :okwarning:

    ds = ds.argo.canyon_med.predict('PO4')
    ds['PO4']


.. _complement-optical-modeling:

Optical modeling
^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

This extension provides methods to compute standard variables from optical modeling of the upper ocean. This feature is available in **argopy** as an extension to the ``argo`` accessor: :class:`Dataset.argo.optic`. It can be used to:

- compute the depth of the euphotic zone, from PAR: :class:`Dataset.argo.optic.Zeu`
- compute the first optical depth, from depth of the euphotic zone: :class:`Dataset.argo.optic.Zpd`
- compute the depth where PAR reaches some threshold value: :class:`Dataset.argo.optic.Z_iPAR_threshold`
- search and qualify Deep Chlorophyll Maxima: :class:`Dataset.argo.optic.DCM`

As an example, let's load one BGC float data with DOWNWELLING_PAR measurements:

.. ipython:: python
    :okwarning:

    ds = DataFetcher(ds='bgc', mode='expert', params='DOWNWELLING_PAR').float(6901864).data
    dsp = ds.argo.point2profile()

.. currentmodule:: argopy

Note that we could have loaded these data with a :class:`ArgoFloat`:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(6901864).open_dataset('Sprof')

We can then simply call on the extension methods to add variables to the dataset:

.. ipython:: python
    :okwarning:

    dsp.argo.optic.Zeu()
    dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
    dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_max=50.)

    dsp.argo.optic.Zpd()

    dsp.argo.optic.Z_iPAR_threshold(threshold=15.)

For the Deep Chlorophyll Maxima diagnostic, we need CHLA data, so let's load data from another BGC float:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(1902303).open_dataset('Sprof')
    dsp.argo.optic.DCM()

Data models
-----------

.. currentmodule:: argopy

By default **argopy** will provide users with :class:`xarray.Dataset` or :class:`pandas.DataFrame`.

For your own analysis, you may prefer to switch from one to the other. This is all built in **argopy**, with the :meth:`DataFetcher.to_dataframe` and :meth:`DataFetcher.to_xarray` methods.

.. ipython:: python
    :okwarning:

    DataFetcher().profile(6902746, 34).to_dataframe()


Note that internally, **argopy** also work with :class:`pyarrow.Table`.

Saving data
===========

.. currentmodule:: xarray

Once you have your Argo data as a :class:`xarray.Dataset`, you can simply use the awesome export possibilities of `xarray <http://xarray.pydata.org>`_ like :meth:`xarray.Dataset.to_netcdf` or :meth:`xarray.Dataset.to_zarr`.

Note that we provide a dedicated method to export an Argo :class:`xarray.Dataset` to zarr that will handle data type casting and compression easily, the :meth:`Dataset.argo.to_zarr` method:

.. code-block:: python
    :caption: Dataset export to zarr

    from argopy import DataFetcher
    ds = DataFetcher(src='gdac').float(6903091).to_xarray()
    # then:
    ds.argo.to_zarr("6903091_prof.zarr")
    # or:
    ds.argo.to_zarr("s3://argopy/sample-data/6903091_prof.zarr")

References
----------
.. [1] Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2020). A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 7. doi:10.3389/fmars.2020.00620.

.. [2] Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2021). Corrigendum: A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 8. doi:10.3389/fmars.2021.650509.