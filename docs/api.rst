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

Data selection methods
----------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.region
   DataFetcher.float
   DataFetcher.profile

Data access methods
-------------------

.. autosummary::
   :toctree: generated/

   DataFetcher.load
   DataFetcher.to_xarray
   DataFetcher.to_dataframe
   DataFetcher.to_index
   DataFetcher.to_dataset

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


Argo file stores
================

.. autosummary::
    :toctree: generated/

    gdacfs

ArgoFloat
---------
.. autosummary::
    :toctree: generated/

    ArgoFloat

List of extensions:

.. currentmodule:: argopy

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor.rst

   ArgoFloat.plot

This extension provides the following **plotting methods** for one Argo float data:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   ArgoFloat.plot.trajectory
   ArgoFloat.plot.map
   ArgoFloat.plot.scatter

ArgoIndex
---------

.. autosummary::
    :toctree: generated/

    ArgoIndex

List of extensions:

.. currentmodule:: argopy

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor.rst

   ArgoIndex.query
   ArgoIndex.plot

**Search on a single property** of a file record:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   ArgoIndex.query.wmo
   ArgoIndex.query.cyc
   ArgoIndex.query.lon
   ArgoIndex.query.lat
   ArgoIndex.query.date
   ArgoIndex.query.params
   ArgoIndex.query.parameter_data_mode
   ArgoIndex.query.profiler_type
   ArgoIndex.query.profiler_label

**Search on at least two properties** of a file record:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   ArgoIndex.query.wmo_cyc
   ArgoIndex.query.lon_lat
   ArgoIndex.query.box
   ArgoIndex.query.compose

**Plotting methods**:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    ArgoIndex.plot.trajectory
    ArgoIndex.plot.bar

Argo meta/related data
======================

.. autosummary::
   :toctree: generated/

   status
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
   >>> import argopy               # import argopy (the dataset 'argo' accessor is then registered)
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
   Dataset.argo.datamode.compute
   Dataset.argo.datamode.filter
   Dataset.argo.datamode.merge
   Dataset.argo.datamode.split


Data Filters
------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.argo.filter_qc
   Dataset.argo.filter_scalib_pres
   Dataset.argo.filter_researchmode
   Dataset.argo.datamode.filter


Extensions
----------
.. currentmodule:: argopy

You can create your own extension to an Argo dataset for specific features. It should be registered by inheriting from :class:`argopy.extensions.ArgoAccessorExtension` and decorated with :class:`argopy.extensions.register_argo_accessor`.

**argopy** comes with the following extensions:

General purposes
^^^^^^^^^^^^^^^^

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.teos10
    Dataset.argo.datamode
    Dataset.argo.create_float_source


BGC specifics
^^^^^^^^^^^^^

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.argo.canyon_med
    Dataset.argo.canyon_med.predict
    Dataset.argo.canyon_med.input_list
    Dataset.argo.canyon_med.output_list

    Dataset.argo.canyon_b
    Dataset.argo.canyon_b.predict
    Dataset.argo.canyon_b.input_list
    Dataset.argo.canyon_b.output_list

    Dataset.argo.content
    Dataset.argo.content.predict
    Dataset.argo.content.input_list
    Dataset.argo.content.output_list

    Dataset.argo.optic
    Dataset.argo.optic.Zeu
    Dataset.argo.optic.Zpd
    Dataset.argo.optic.Z_iPAR_threshold
    Dataset.argo.optic.DCM

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
    Dataset.argo.N_POINTS
    Dataset.argo.to_zarr
    Dataset.argo.reduce_profile


Utilities
=========

Function under the ``argopy.utils`` submodule.

.. currentmodule:: argopy.utils

Lists
-----

.. autosummary::
   :toctree: generated/

    list_available_data_src
    list_available_index_src
    list_standard_variables
    list_multiprofile_file_variables
    list_core_parameters
    list_bgc_s_variables
    list_bgc_s_parameters
    list_radiometry_variables
    list_radiometry_parameters
    list_gdac_servers

Checkers
--------

.. autosummary::
   :toctree: generated/

    check_wmo
    check_cyc
    check_gdac_path

    isconnected
    urlhaskeyword
    isalive
    isAPIconnected


Misc
--------

.. autosummary::
   :toctree: generated/

    float_wmo
    Registry

    Chunker

    drop_variables_not_in_all_datasets
    fill_variables_not_in_all_datasets

    GreenCoding
    Github

    optical_modeling

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

    gdacfs
    stores.ArgoStoreProto
    stores.filestore
    stores.httpstore
    stores.httpstore_erddap
    stores.memorystore
    stores.ftpstore
    stores.s3store
    stores.ArgoKerchunker

Fetcher sources
---------------

ERDDAP
^^^^^^

.. autosummary::
    :toctree: generated/

    data_fetchers.erddap_data.ErddapArgoDataFetcher
    data_fetchers.erddap_data.Fetch_wmo
    data_fetchers.erddap_data.Fetch_box

GDAC
^^^^

.. autosummary::
    :toctree: generated/

    data_fetchers.gdac_data.GDACArgoDataFetcher
    data_fetchers.gdac_data.Fetch_wmo
    data_fetchers.gdac_data.Fetch_box


Argovis
^^^^^^^

.. autosummary::
    :toctree: generated/

    data_fetchers.argovis_data.ArgovisDataFetcher
    data_fetchers.argovis_data.Fetch_wmo
    data_fetchers.argovis_data.Fetch_box
