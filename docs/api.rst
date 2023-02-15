#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

.. contents::
   :local:

Top-levels functions
====================

.. currentmodule:: argopy

Fetchers
--------

.. autosummary::
    :toctree: generated/

    DataFetcher
    IndexFetcher

Fetcher access points
---------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.region
   DataFetcher.float
   DataFetcher.profile

.. autosummary::
   :toctree: generated/

   IndexFetcher.region
   IndexFetcher.float
   IndexFetcher.profile

Fetcher methods
---------------

.. autosummary::
   :toctree: generated/

   DataFetcher.load
   DataFetcher.to_xarray
   DataFetcher.to_dataframe
   DataFetcher.to_index

.. autosummary::
   :toctree: generated/

   IndexFetcher.load
   IndexFetcher.to_xarray
   IndexFetcher.to_dataframe
   IndexFetcher.to_csv

Data visualisation
------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.plot
   DataFetcher.dashboard
   IndexFetcher.plot
   dashboard


Fetcher properties
------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.uri
   DataFetcher.data
   DataFetcher.index
   IndexFetcher.index


Helpers
-------

.. autosummary::
   :toctree: generated/

   status
   TopoFetcher
   set_options
   clear_cache
   tutorial.open_dataset

Low-level functions
===================

.. currentmodule:: argopy

.. autosummary::
    :toctree: generated/

    show_versions
    utilities.list_available_data_src
    utilities.list_available_index_src
    utilities.get_coriolis_profile_id
    utilities.get_ea_profile_page
    utilities.ArgoNVSReferenceTables


Dataset.argo (xarray accessor)
==============================

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor.rst

   Dataset.argo

This accessor extends :py:class:`xarray.Dataset`. Proper use of this accessor should be like:

.. code-block:: python

   >>> import xarray as xr         # first import xarray
   >>> import argopy               # import argopy (the dataset 'argo' accessor is registered)
   >>> from argopy import DataFetcher
   >>> ds = DataFetcher().float([6902766, 6902772, 6902914, 6902746]).load().data
   >>> ds.argo
   >>> ds.argo.filter_qc()


Data Transformation
-------------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.argo.point2profile
   Dataset.argo.profile2point
   Dataset.argo.interp_std_levels
   Dataset.argo.groupby_pressure_bins

Data Filters
------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.argo.filter_qc
   Dataset.argo.filter_data_mode
   Dataset.argo.filter_scalib_pres

Processing
----------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.teos10
    Dataset.argo.create_float_source

Misc
----

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.uid
    Dataset.argo.cast_types

Internals
=========

.. currentmodule:: argopy

File systems
------------

.. autosummary::
    :toctree: generated/

    argopy.stores.filestore
    argopy.stores.httpstore
    argopy.stores.memorystore
    argopy.stores.ftpstore

Argo index store
----------------

.. autosummary::
    :toctree: generated/

    argopy.stores.indexstore
    argopy.stores.indexfilter_wmo
    argopy.stores.indexfilter_box
    argopy.stores.indexstore_pa
    argopy.stores.indexstore_pd

Fetcher sources
---------------

ERDDAP
^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.erddap_data.ErddapArgoDataFetcher
    argopy.data_fetchers.erddap_data.Fetch_wmo
    argopy.data_fetchers.erddap_data.Fetch_box

GDAC
^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.gdacftp_data.FTPArgoDataFetcher
    argopy.data_fetchers.gdacftp_data.Fetch_wmo
    argopy.data_fetchers.gdacftp_data.Fetch_box

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

    argopy.plot.dashboard
    argopy.plot.plot_trajectory
    argopy.plot.bar_plot
    argopy.plot.open_sat_altim_report

Utilities
---------

.. autosummary::
   :toctree: generated/

    argopy.utilities.check_wmo
    argopy.utilities.check_cyc
    argopy.utilities.float_wmo
    argopy.utilities.Registry
    argopy.utilities.list_standard_variables
    argopy.utilities.list_multiprofile_file_variables
    argopy.utilities.Chunker

    argopy.utilities.isconnected
    argopy.utilities.urlhaskeyword
    argopy.utilities.isalive
    argopy.utilities.isAPIconnected
