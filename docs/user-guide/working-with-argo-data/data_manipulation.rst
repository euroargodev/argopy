Manipulating data
=================

.. currentmodule:: xarray

Once you fetched Argo data, **argopy** comes with a handy :class:`xarray.Dataset` accessor ``argo`` to perform specific manipulation of the data. This means that if your dataset is named ``ds``, then you can use ``ds.argo`` to access more **argopy** functions. The full list is available in the API documentation page :ref:`Dataset.argo (xarray accessor)`.

In this section, we present how **argopy** can help in manipulating Argo measurements and parameters.

.. contents::
   :local:


Points vs profiles
------------------

.. currentmodule:: xarray

By default, fetched data are returned as a 1D array collection of measurements:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher

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
------------------------------

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
------------------------------

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
-------

.. currentmodule:: xarray

If you fetched data with the ``expert`` mode, you may want to use *filters* to help you curate the data.

- **QC flag filter**: :meth:`Dataset.argo.filter_qc`. This method allows you to filter measurements according to QC flag values. This filter modifies all variables of the dataset.
- **Data mode filter**: :meth:`Dataset.argo.datamode.filter`. This method allows you to filter variables according to their data mode.
- **OWC variables filter**: :meth:`Dataset.argo.filter_scalib_pres`. This method allows you to filter variables according to OWC salinity calibration software requirements. This filter modifies pressure, temperature and salinity related variables of the dataset.

