.. currentmodule:: argopy

#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

Data Fetchers
=============

.. autosummary::
    :toctree: generated/

    argopy.DataFetcher

Search entries
--------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.region
   argopy.DataFetcher.float
   argopy.DataFetcher.profile

Data formats
------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.to_xarray
   argopy.DataFetcher.to_dataframe

Index/meta-data fetchers
========================

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher

Search entries
--------------

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.region
   argopy.IndexFetcher.float

Data formats
------------

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.to_xarray
   argopy.IndexFetcher.to_dataframe
   argopy.IndexFetcher.to_csv

Helpers
-------

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.plot

Low-level functions
===================

.. autosummary::
    :toctree: generated/

    argopy.tutorial.open_dataset
    argopy.set_options
    argopy.show_versions
    argopy.utilities.open_etopo1
    argopy.utilities.urlopen
    argopy.utilities.list_available_data_src

Xarray *argo* name space
==========================

.. automodule:: argopy.xarray

.. autoclass:: argopy.ArgoAccessor()
    :members: