.. _metadata_fetching:

Fetching Argo meta-data
=======================

Since the Argo measurements dataset is quite complex, it comes with a collection of index files, or lookup tables with meta data. These index help you determine what you can expect before retrieving the full set of measurements. **argopy** has a specific fetcher for index:

.. ipython:: python
    :okwarning:

    from argopy import IndexFetcher as ArgoIndexFetcher
    index_loader = ArgoIndexFetcher()

You can use the Index fetcher with the ``region`` or ``float`` access points, similarly to data fetching:

.. ipython:: python
    :okwarning:

    idx = index_loader.float(5902404).load()
    idx.index

Alternatively, you can use :meth:`argopy.IndexFetcher.to_dataframe()`.
