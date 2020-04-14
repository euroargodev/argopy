Usage
#####

Fetching Argo Data
==================

To access Argo data, you need to use a data fetcher. You can import and instantiate the default argopy data fetcher
like:

.. ipython:: python

    from argopy import DataFetcher as ArgoDataFetcher
    argo_loader = ArgoDataFetcher()

Then, you can request data for a specific space/time domain, for a given float or for a given vertical profile.

For a space/time domain
-----------------------

Use the fetcher access point :meth:`argopy.DataFetcher.region` to specify a domain and chain with the :meth:`argopy.DataFetcher.to_xarray` to get the data returned as :class:`xarray.Dataset`.

For instance, to retrieve data from 85W to 45W, 10N to 20N and 0db to 10db (the first surface values):

.. ipython:: python

    ds = argo_loader.region([-85, -45, 10, 20, 0, 10]).to_xarray()
    ds

You will have noticed that we without constraints on time, the fetcher will return all data available in this region. If you want to specify a time range, add date strings to the region definition list.

For instance, to retrieve data between January 15th, 2011 and the end of April, 2011:

.. ipython:: python

    ds = argo_loader.region([-85, -45, 10, 20, 0, 10, '2011-01-15', '2011-05']).to_xarray()
    ds

Note here that the last date is not included and that the time delta depends on the way you format the string. For instance '2011-05-15' will include data up to '2011-05-14', or '2011-05-28T12:00:00' up to the 11th hour of '2011-05-28'.

For one or more floats
----------------------

If you know the Argo float unique identifier number called a `WMO number <https://www.wmo.int/pages/prog/amp/mmop/wmo-number-rules.html>`_ you can use the fetcher access point :meth:`argopy.DataFetcher.float` to specify the float WMO platform number and chain with the :meth:`argopy.DataFetcher.to_xarray` to get the data returned as :class:`xarray.Dataset`.

For instance, to retrieve data for float WMO *6902746*:

.. ipython:: python

    ds = argo_loader.float(6902746).to_xarray()
    ds

To fetch data for a collection of floats, input them in a list:

.. ipython:: python

    ds = argo_loader.float([6902746, 6902755]).to_xarray()
    ds

For one or more profiles
------------------------

Use the fetcher access point :meth:`argopy.DataFetcher.profile` to specify the float WMO platform number and the profile cycle number to retrieve profiles for, then chain with the :meth:`argopy.DataFetcher.to_xarray` to get the data returned as :class:`xarray.Dataset`.

For instance, to retrieve data for the 12th profile of float WMO 6902755:

.. ipython:: python

    ds = argo_loader.profile(6902755, 12).to_xarray()
    ds

To fetch data for more than one profile, input them in a list:

.. ipython:: python

    ds = argo_loader.profile(6902755, np.arange(1,12)).to_xarray()
    ds