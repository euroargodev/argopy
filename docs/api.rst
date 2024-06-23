#############
API reference
#############

This page provides an auto-generated summary of argopy's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

.. contents::
   :local:

Argo Data Fetchers
==================

.. currentmodule:: argopy

.. autosummary::
    :toctree: generated/

    DataFetcher
    IndexFetcher

Data selection methods
----------------------

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

Data access methods
-------------------

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

.. _Fetcher Data Visualisation:

Data visualisation methods
--------------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.plot
   DataFetcher.dashboard
   IndexFetcher.plot


Properties
----------

.. autosummary::
   :toctree: generated/

   DataFetcher.data
   DataFetcher.index
   DataFetcher.domain
   DataFetcher.uri
   IndexFetcher.index

Utilities for Argo related data
===============================

.. autosummary::
   :toctree: generated/

   status
   ArgoIndex
   ArgoDocs
   ArgoDOI
   ArgoNVSReferenceTables
   OceanOPSDeployments
   CTDRefDataFetcher
   TopoFetcher

.. _Module Visualisation:

Data visualisation
==================

Visualisation functions available at the ``argopy`` module level:

.. currentmodule:: argopy

.. autosummary::
   :toctree: generated/

    dashboard
    ArgoColors


All other visualisation functions are in the :mod:`argopy.plot` submodule:

.. currentmodule:: argopy.plot

.. autosummary::
   :toctree: generated/

    open_sat_altim_report
    scatter_map
    bar_plot
    scatter_plot
    latlongrid


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
   Dataset.argo.transform_data_mode


Data Filters
------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.argo.filter_data_mode
   Dataset.argo.filter_qc
   Dataset.argo.filter_scalib_pres
   Dataset.argo.filter_researchmode

Extensions
----------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.teos10
    Dataset.argo.create_float_source
    Dataset.argo.canyon_med

Misc
----

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.index
    Dataset.argo.domain
    Dataset.argo.list_WMO_CYC
    Dataset.argo.uid
    Dataset.argo.cast_types

Utilities
=========

Function under the ``argopy.utils`` submodule.

.. currentmodule:: argopy.utils

.. autosummary::
   :toctree: generated/

    list_available_data_src
    list_available_index_src
    list_standard_variables
    list_multiprofile_file_variables

    check_wmo
    check_cyc

    float_wmo
    Registry

    Chunker

    isconnected
    urlhaskeyword
    isalive
    isAPIconnected

    drop_variables_not_in_all_datasets
    fill_variables_not_in_all_datasets

Argopy helpers
==============
.. currentmodule:: argopy

.. autosummary::
   :toctree: generated/

   set_options
   clear_cache
   tutorial.open_dataset
   show_versions
   xarray.ArgoEngine


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

    ArgoIndex
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


Argovis
^^^^^^^

.. autosummary::
    :toctree: generated/

    argopy.data_fetchers.argovis_data.ArgovisDataFetcher
    argopy.data_fetchers.argovis_data.Fetch_wmo
    argopy.data_fetchers.argovis_data.Fetch_box
