.. _user-mode:

User mode: expert, standard or research
=======================================

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


User mode details
-----------------

**argopy** provides 3 user modes:

- **expert** mode return all the Argo data, without any postprocessing,
- **standard** mode simplifies the dataset, remove most of its jargon and return *a priori* good data,
- **research** mode simplifies the dataset to its heart, preserving only data of the highest quality for research studies, including studies sensitive to small pressure and salinity bias (e.g. calculations of global ocean heat content or mixed layer depth).

Hence, in **standard** and **research** modes, fetched data are automatically filtered to account for their quality (using the *quality control flags*) and level of processing by the data centers (using each *parameter data mode* indicating if ADMT human experts carefully looked at the data or not). Both mode return a postprocessed subset of the full Argo dataset.

One could conclude that the main difference between the **standard** and **research** modes is in the level of data quality insurance.
In **standard** mode, only good or probably good data are returned, which includes real time data that have been validated automatically but not by a human expert.
The **research** mode is the safer choice, with data of the highest quality, carefully checked by a human expert of the ADMT team.

Table below summarizes the technical differences between each user modes:

.. list-table:: Table of **argopy** user mode data processing details
    :header-rows: 1
    :stub-columns: 1

    * -
      - ``expert``
      - ``standard``
      - ``research``
    * - Level of quality (QC flags) retained
      - all
      - good or probably good
      - good
    * - Level of assessment (Data mode) retained
      - all
      - all, but merged in a single variable
      - best only (delayed)
    * - Pressure error
      - any
      - any
      - smaller than 20db
    * - Variables returned
      - all
      - all but technical
      - comprehensive minimum



How to set the user mode ?
--------------------------

Let's import the **argopy** data fetcher:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher as ArgoDataFetcher


By default, all **argopy** data fetchers are set to work with a
**standard** user mode.

If you want to change the user mode, or simply makes it explicit, you
can use:

-  the **argopy** global option setter:

.. ipython:: python
    :okwarning:

    argopy.set_options(mode='standard')

-  a temporary context:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='standard'):
        ArgoDataFetcher().profile(6902746, 34)

-  the option when instantiating the data fetcher:

.. ipython:: python
    :okwarning:

    ArgoDataFetcher(mode='standard').profile(6902746, 34)

Example of differences in user modes
------------------------------------

To highlight differences in data returned for each user modes, let’s compare data fetched for one profile.

You will note that the **standard** and **research** modes have fewer variables to let you
focus on your analysis. For **expert**, all Argo variables for you to
work with are here.

.. ipython:: python
    :okwarning:

    argopy.set_options(ftp='https://data-argo.ifremer.fr')

In **expert** mode:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='expert'):
        ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
        print(ds.data_vars)

In **standard** mode:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='standard'):
        ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
        print(ds.data_vars)

In **research** mode:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='research'):
        ds = ArgoDataFetcher(src='gdac').profile(6902755, 12).to_xarray()
        print(ds.data_vars)
