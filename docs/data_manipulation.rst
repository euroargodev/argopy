Manipulating data
=================

Once you fetched data, **argopy** comes with a handy :class:`xarray.Dataset` accessor namespace ``argo`` to perform specific manipulation of the data. This means that if your dataset is `ds`, then you can use `ds.argo` to access more **argopy** functions.

Let's start with standard import:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher as ArgoDataFetcher


Transformation
--------------

Points vs profiles
~~~~~~~~~~~~~~~~~~

Fetched data are returned as a 1D array collection of measurements:

.. ipython:: python
    :okwarning:

    argo_loader = ArgoDataFetcher().region([-75,-55,30.,40.,0,100., '2011-01-01', '2011-01-15'])
    ds_points = argo_loader.to_xarray()
    ds_points

If you prefer to work with a 2D array collection of vertical profiles, simply transform the dataset with :meth:`argopy.xarray.ArgoAccessor.point2profile`:

.. ipython:: python
    :okwarning:

    ds_profiles = ds_points.argo.point2profile()
    ds_profiles

You can simply reverse this transformation with the :meth:`argopy.xarray.ArgoAccessor.profile2point`:

.. ipython:: python
    :okwarning:

    ds = ds_profiles.argo.profile2point()
    ds

Interpolation to standard levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your dataset is a collection of vertical **profiles**, you can interpolate variables on standard pressure levels using :meth:`argopy.xarray.ArgoAccessor.interp_std_levels` with your levels as input :

.. ipython:: python
    :okwarning:

    ds_interp = ds_profiles.argo.interp_std_levels([0,10,20,30,40,50])
    ds_interp

Note on the linear interpolation process : 
    - Only profiles that have a maximum pressure higher than the highest standard level are selected for interpolation.
    - Remaining profiles must have at least five data points to allow interpolation.
    - For each profile, shallowest data point is repeated to the surface to allow a 0 standard level while avoiding extrapolation.

Filters
~~~~~~~

If you fetched data with the ``expert`` mode, you may want to use
*filters* to help you curate the data.

[To be added]

Complementary data
------------------

TEOS-10 variables
~~~~~~~~~~~~~~~~~

You can compute additional ocean variables from `TEOS-10 <http://teos-10.org/>`_. The default list of variables is: 'SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP' ('SOUND_SPEED' is optional). `Simply raise an issue to add a new one <https://github.com/euroargodev/argopy/issues/new/choose>`_.

This can be done using the :meth:`argopy.xarray.ArgoAccessor.teos10` method and indicating the list of variables you want to compute:

.. ipython:: python
    :okwarning:

    ds = ArgoDataFetcher().float(2901623).to_xarray()
    ds.argo.teos10(['SA', 'CT', 'PV'])
    ds

.. ipython:: python
    :okwarning:

    ds['SA']

Data models
-----------

By default **argopy** works with `xarray.DataSet <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ and comes with the accessor namespace ``argo`` (see `here for more on xarray accessor <http://xarray.pydata.org/en/stable/internals/extending-xarray.html>_`).

For your own analysis, you may prefer to work with a `Pandas dataframe <https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe>`_. 

.. ipython:: python
    :okwarning:

    df = ArgoDataFetcher().profile(6902746, 34).to_dataframe()
    df

but keep in mind that this is merely a short cut for the :meth:`xarray.Dataset.to_dataframe` method.

Saving data
===========

Once you have your Argo data as :class:`xarray.Dataset`, simply use the awesome possibilities of `xarray <http://xarray.pydata.org>`_ like :meth:`xarray.Dataset.to_netcdf` or :meth:`xarray.Dataset.to_zarr`.
