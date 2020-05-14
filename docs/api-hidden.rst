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

    argopy.xarray.ArgoAccessor.point2profile