.. _starting:

Getting started with <img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/>

Import the **argopy** data fetcher:

.. ipython:: python

    from argopy import DataFetcher as ArgoDataFetcher

Then, to get access to Argo data, all you need is 1 line of code:

.. ipython:: python

    ds = ArgoDataFetcher().region([-75, -45, 20, 30, 0, 100, '2011', '2012']).to_xarray()

In this example, we used a data fetcher to get data for a given space/time region.
We retrieved all Argo data measurements from 75W to 45W, 20N to 30N, 0db to 100db and from January to May 2011 (the max date is exclusive).
Data are returned as a collection of measurements in a `xarray.Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`.

.. ipython:: python

    ds

Fetched data are returned as a 1D array collection of measurements.

If you prefer to work with a 2D array collection of vertical profiles, simply transform the dataset with the `xarray.Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>` accessor method `argo.point2profile <https://argopy.readthedocs.io/en/latest/api.html#argopy.ArgoAccessor.point2profile>`:

.. ipython:: python

    ds = ds.argo.point2profile()
    ds

You can also fetch data for a specific float using its [WMO number](<https://www.wmo.int/pages/prog/amp/mmop/wmo-number-rules.html):

.. ipython:: python

    ds = ArgoDataFetcher().float(6902746).to_xarray()

or for a float profile using the cycle number:

.. ipython:: python

    ds = ArgoDataFetcher().profile(6902755, 12).to_xarray()