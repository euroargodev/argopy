.. currentmodule:: argopy
.. _user-mode:

User mode (🏄, 🏊, 🚣)
=======================

.. hint::

    **argopy** manipulates the raw data to make them easier to work with. To make sure you understand the data you're getting, have a look to this section.

.. contents:: Contents
   :local:

**Problem**

For non-experts of the Argo dataset, it can be quite
complicated to get access to Argo measurements. Indeed, the Argo data
set is very complex, with thousands of different variables, tens of
reference tables and a `user manual <https://doi.org/10.13155/29825>`__
more than 100 pages long.

This is mainly due to:

-  Argo measurements coming from many different models of floats or
   sensors,
-  quality control of *in situ* measurements of autonomous platforms
   being really a matter of ocean and data experts,
-  the Argo data management workflow being distributed between more than
   10 Data Assembly Centers all around the world,
-  the Argo autonomous profiling floats, despite quite a simple
   principle of functioning, is a rather complex robot that needs a lot
   of data to be monitored and logged.

**Solution**

In order to ease Argo data analysis for the vast majority of
users, we implemented in **argopy** different levels of verbosity and
data processing to hide or simply remove variables only meaningful to
experts.

.. _user-mode-details:

User mode details
-----------------

**argopy** provides 3 user modes:

- 🏄 **expert** mode return all the Argo data, without any postprocessing,
- 🏊 **standard** mode simplifies the dataset, remove most of its jargon and return *a priori* good data,
- 🚣 **research** mode simplifies the dataset to its heart, preserving only data of the highest quality for research studies, including studies sensitive to small pressure and salinity bias (e.g. calculations of global ocean heat content or mixed layer depth).

In **standard** and **research** modes, fetched data are automatically filtered to account for their quality (using the *quality control flags*) and level of processing by the data centers (considering for each parameter the data mode which indicates if a human expert has carefully looked at the data or not). Both mode return a postprocessed subset of the full Argo dataset.

Hence the main difference between the **standard** and **research** modes is in the level of data quality insurance.
In **standard** mode, only good or probably good data are returned and includes real time data that have been validated automatically but not by a human expert.
The **research** mode is the safer choice, with data of the highest quality, carefully checked in delayed mode by a human expert of the `Argo Data Management Team <http://www.argodatamgt.org>`_.

.. list-table:: Table of **argopy** user mode data processing details for **physical** parameters (``phy`` :ref:`dataset <data-set>`)
    :header-rows: 1
    :stub-columns: 1

    * -
      - ``expert``
      - ``standard``
      - ``research``
    * -
      - 🏄
      - 🏊
      - 🚣
    * - Level of quality (QC flags) retained
      - all
      - good or probably good (QC=[1,2])
      - good (QC=1)
    * - Level of assessment (Data mode) retained
      - all: [R,D,A] modes
      - all: [R,D,A] modes, but PARAM_ADJUSTED and PARAM are merged in a single variable according to the mode
      - best only (D mode only)
    * - Pressure error
      - any
      - any
      - smaller than 20db
    * - Variables returned
      - all
      - all without jargon (DATA_MODE and QC_FLAG are retained)
      - comprehensive minimum

.. admonition:: About the 🟢 **bgc** dataset

    The table of **argopy** user mode data processing details for **biogeochemical** parameters is being defined (:issue:`280`) and will be implemented in a near future release.

How to select a user mode ?
---------------------------

Let's import the **argopy** data fetcher:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher as ArgoDataFetcher


By default, all **argopy** data fetchers are set to work with a
**standard** user mode.

If you want to change the user mode, or to simply makes it explicit in your code, you
can use one of the following 3 methods:

-  the **argopy** global option setter:

.. ipython:: python
    :okwarning:

    argopy.set_options(mode='standard')

-  a temporary **context**:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='expert'):
        ArgoDataFetcher().profile(6902746, 34)

-  or the **fetcher option**:

.. ipython:: python
    :okwarning:

    ArgoDataFetcher(mode='research').profile(6902746, 34)

Example of differences in user modes
------------------------------------

To highlight differences in data returned for each user modes, let’s compare data fetched for one profile.

You will note that the **standard** and **research** modes have fewer variables to let you
focus on your analysis. For **expert**, all Argo variables for you to
work with are here.


.. tabs::

    .. tab:: In **expert** mode:

        .. ipython:: python
            :okwarning:

            with argopy.set_options(mode='expert'):
                ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
                print(ds.data_vars)

    .. tab:: In **standard** mode:

        .. ipython:: python
            :okwarning:

            with argopy.set_options(mode='standard'):
                ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
                print(ds.data_vars)

    .. tab:: In **research** mode:

        .. ipython:: python
            :okwarning:

            with argopy.set_options(mode='research'):
                ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
                print(ds.data_vars)

.. note::

    A note for **expert** users looking at **standard** and **research** mode results: they are no ``PARAM_ADJUSTED`` variables because they've been renamed ``PARAM`` wherever the ``DATA_MODE`` variable was ``ADJUSTED`` or ``DELAYED``.