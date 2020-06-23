.. currentmodule:: argopy

#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

Top-levels functions
====================

.. autosummary::
    :toctree: generated/

    argopy.DataFetcher
    argopy.IndexFetcher

Search entries
--------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.region
   argopy.DataFetcher.float
   argopy.DataFetcher.profile

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.region
   argopy.IndexFetcher.float

I/O and Data formats
--------------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.to_xarray
   argopy.DataFetcher.to_dataframe

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.to_xarray
   argopy.IndexFetcher.to_dataframe
   argopy.IndexFetcher.to_csv

Visualisation
-------------

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.plot
   argopy.dashboard

Helpers
-------

.. autosummary::
   :toctree: generated/

   argopy.set_options
   argopy.clear_cache
   argopy.tutorial.open_dataset

Low-level functions
===================

.. autosummary::
    :toctree: generated/

    argopy.show_versions
    argopy.utilities.open_etopo1
    argopy.utilities.list_available_data_src

Internals
=========

File systems
------------

.. autosummary::
    :toctree: generated/

    argopy.stores.filestore
    argopy.stores.httpstore
    argopy.stores.memorystore
    argopy.stores.indexstore
    argopy.stores.indexfilter_wmo
    argopy.stores.indexfilter_box

Xarray *argo* name space
==========================

.. automodule:: argopy.xarray

.. autoclass:: argopy.ArgoAccessor()
    :members: