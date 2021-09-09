.. currentmodule:: argopy

#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

Top-levels functions
====================

Fetchers
--------

.. autosummary::
    :toctree: generated/

    argopy.DataFetcher
    argopy.IndexFetcher

Fetcher access points
---------------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.region
   argopy.DataFetcher.float
   argopy.DataFetcher.profile

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.region
   argopy.IndexFetcher.float

Fetching methods
----------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.load
   argopy.DataFetcher.to_xarray
   argopy.DataFetcher.to_dataframe
   argopy.DataFetcher.to_index

.. autosummary::
   :toctree: generated/

   argopy.IndexFetcher.load
   argopy.IndexFetcher.to_xarray
   argopy.IndexFetcher.to_dataframe
   argopy.IndexFetcher.to_csv

Fetched data visualisation
--------------------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.plot
   argopy.IndexFetcher.plot
   argopy.dashboard


Fetcher properties
------------------

.. autosummary::
   :toctree: generated/

   argopy.DataFetcher.uri
   argopy.DataFetcher.data
   argopy.DataFetcher.index
   argopy.IndexFetcher.index


Helpers
-------

.. autosummary::
   :toctree: generated/

   argopy.status
   argopy.set_options
   argopy.clear_cache
   argopy.tutorial.open_dataset

Low-level functions
===================

.. autosummary::
    :toctree: generated/

    argopy.show_versions
    argopy.utilities.list_available_data_src
    argopy.utilities.list_available_index_src

Internals
=========

File systems
------------

.. autosummary::
    :toctree: generated/

    argopy.stores.filestore
    argopy.stores.httpstore
    argopy.stores.memorystore

.. autosummary::
    :toctree: generated/

    argopy.stores.indexstore
    argopy.stores.indexfilter_wmo
    argopy.stores.indexfilter_box

Fetchers
--------

ERDDAP
^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.erddap_data.ErddapArgoDataFetcher
    argopy.data_fetchers.erddap_data.Fetch_wmo
    argopy.data_fetchers.erddap_data.Fetch_box

Local FTP
^^^^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.localftp_data.LocalFTPArgoDataFetcher
    argopy.data_fetchers.localftp_data.Fetch_wmo
    argopy.data_fetchers.localftp_data.Fetch_box

Argovis
^^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.argovis_data.ArgovisDataFetcher
    argopy.data_fetchers.argovis_data.Fetch_wmo
    argopy.data_fetchers.argovis_data.Fetch_box

Xarray *argo* name space
==========================

.. automodule:: argopy.xarray

.. autoclass:: argopy.ArgoAccessor()
    :members: