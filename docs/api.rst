.. currentmodule:: argopy

#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

.. contents:: Table of contents:
   :local:

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

Helpers and utilities
---------------------

.. autosummary::
   :toctree: generated/

   argopy.status
   argopy.set_options
   argopy.show_options
   argopy.show_versions
   argopy.clear_cache
   argopy.TopoFetcher
   argopy.tutorial.open_dataset

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

Utilities
---------

Lister
^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.utilities.list_available_data_src
    argopy.utilities.list_available_index_src
    argopy.utilities.list_standard_variables
    argopy.utilities.list_multiprofile_file_variables

Formatter
^^^^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.utilities.format_oneline
    argopy.utilities.Chunker
    argopy.utilities.wmo2box
    argopy.utilities.groupby_remap
    argopy.utilities.linear_interpolation_remap

Checker
^^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.utilities.check_localftp
    argopy.utilities.is_box
    argopy.utilities.is_indexbox
    argopy.utilities.is_wmo
    argopy.utilities.check_wmo


Data Fetchers
-------------

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

Plotters
--------

.. autosummary::
   :toctree: generated/

    argopy.plotters.plot_trajectory
    argopy.plotters.bar_plot
    argopy.plotters.open_dashboard


Xarray *argo* name space
==========================

.. automodule:: argopy.xarray

.. autoclass:: argopy.ArgoAccessor()
    :members: