.. _data_fetching:

Fetching Argo data
==================

To access Argo data, you need to use a data fetcher. You can import and instantiate the default argopy data fetcher
like this:

.. ipython:: python

    from argopy import DataFetcher as ArgoDataFetcher
    argo_loader = ArgoDataFetcher()
    argo_loader

Then, you can request data for a specific **space/time domain**, for a given **float** or for a given vertical **profile**.

If you fetch a lot of data, you may want to look at the :ref:`performances` section.

For a space/time domain
-----------------------

Use the fetcher access point :meth:`argopy.DataFetcher.region` to specify a domain and chain with the :meth:`argopy.DataFetcher.to_xarray` to get the data returned as :class:`xarray.Dataset`.

For instance, to retrieve data from 75W to 45W, 20N to 30N, 0db to 10db and from January to May 2011:

.. ipython:: python

    ds = argo_loader.region([-75, -45, 20, 30, 0, 10, '2011-01-01', '2011-06']).to_xarray()
    ds

Note that:

- the constraints on time is not mandatory: if not specified, the fetcher will return all data available in this region.

- the last time bound is exclusive: that's why here we specify June to retrieve data collected in May.

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

    ds = argo_loader.profile(6902755, [3, 12]).to_xarray()
    ds
