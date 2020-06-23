.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. currentmodule:: argopy

.. autosummary::
    :toctree: generated/

    argopy.fetchers.ArgoDataFetcher
    argopy.fetchers.ArgoDataFetcher.region
    argopy.fetchers.ArgoDataFetcher.float
    argopy.fetchers.ArgoDataFetcher.profile
    argopy.fetchers.ArgoDataFetcher.to_xarray
    argopy.fetchers.ArgoDataFetcher.to_dataframe

    argopy.fetchers.ArgoIndexFetcher
    argopy.fetchers.ArgoIndexFetcher.region
    argopy.fetchers.ArgoIndexFetcher.float
    argopy.fetchers.ArgoIndexFetcher.to_xarray
    argopy.fetchers.ArgoIndexFetcher.to_dataframe
    argopy.fetchers.ArgoIndexFetcher.to_csv
    argopy.fetchers.ArgoIndexFetcher.plot

    argopy.options.set_options

    argopy.tutorial.open_dataset

    argopy.utilities.show_versions
    argopy.utilities.clear_cache
    argopy.utilities.open_etopo1
    argopy.utilities.list_available_data_src
    argopy.utilities.list_available_index_src

    argopy.xarray.ArgoAccessor.point2profile

    argopy.plotters.open_dashboard

    argopy.stores.fsspec_wrappers.filestore
    argopy.stores.fsspec_wrappers.httpstore
    argopy.stores.fsspec_wrappers.memorystore
    argopy.stores.argo_index.indexstore
    argopy.stores.argo_index.indexfilter_wmo
    argopy.stores.argo_index.indexfilter_box