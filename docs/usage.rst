.. currentmodule:: argopy

Usage
=====

To get access to Argo data, all you need is 2 lines of codes:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher as ArgoDataFetcher
    ds = ArgoDataFetcher().region([-75, -45, 20, 30, 0, 100, '2011-01', '2011-06']).to_xarray()

In this example, we used a :class:`DataFetcher` to get data for a given space/time region.
We retrieved all Argo data measurements from 75W to 45W, 20N to 30N, 0db to 100db and from January to May 2011 (the max date is exclusive).
Data are returned as a collection of measurements in a :class:`xarray.Dataset`:

.. ipython:: python
    :okwarning:

    ds

.. currentmodule:: xarray

Fetched data are returned as a 1D array collection of measurements. If you prefer to work with a 2D array collection of vertical profiles, simply transform the dataset with the :class:`xarray.Dataset` accessor method :meth:`Dataset.argo.point2profile`:

.. ipython:: python
    :okwarning:

    ds = ds.argo.point2profile()
    ds

You can also fetch data for a specific float using its `WMO number <https://www.wmo.int/pages/prog/amp/mmop/wmo-number-rules.html>`_:

.. ipython:: python
    :okwarning:

    ds = ArgoDataFetcher().float(6902746).to_xarray()

or for a float profile using the cycle number:

.. ipython:: python
    :okwarning:

    ds = ArgoDataFetcher().profile(6902755, 12).to_xarray()
