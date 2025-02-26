.. currentmodule:: argopy
.. _user-mode:

.. |ds_phy| replace:: 🟡+🔵
.. |ds_bgc| replace:: 🟢
.. |mode_expert| replace:: 🏄
.. |mode_standard| replace:: 🏊
.. |mode_research| replace:: 🚣

User modes (|mode_expert|, |mode_standard|, |mode_research|)
========================

.. hint::

    **argopy** manipulates the raw data to make them easier to work with. To make sure you understand the data you're getting, have a look to this section.

.. contents:: Contents
   :local:

**The problem we're trying to solve**

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

**The solution proposed by argopy**

In order to ease Argo data analysis for the vast majority of
users, we implemented in **argopy** different levels of verbosity and
data processing to hide or simply remove variables only meaningful to
experts.

.. _user-mode-definition:

Definitions
-----------

**argopy** provides 3 user modes:

- 🏄 **expert** mode return all the Argo data, without any postprocessing,
- 🏊 **standard** mode simplifies the dataset, remove most of its jargon and return *a priori* good data,
- 🚣 **research** mode simplifies the dataset to its heart, preserving only data of the highest quality for research studies, including studies sensitive to small pressure and salinity bias (e.g. calculations of global ocean heat content or mixed layer depth).

In **standard** and **research** modes, fetched data are automatically filtered to account for their quality (using the *quality control flags*) and level of processing by the data centers (considering for each parameter the data mode which indicates if a human expert has carefully looked at the data or not). Both modes return a postprocessed subset of the full Argo dataset.

Hence the main difference between the **standard** and **research** modes is in the level of data quality insurance.
In **standard** mode, only good or probably good data are returned and these may include real time data that have been validated automatically but not by a human expert.
The **research** mode is the safer choice, with data of the highest quality, carefully checked in delayed mode by a human expert of the `Argo Data Management Team <http://www.argodatamgt.org>`_.


.. _user-mode-standard:

|mode_standard| Standard mode (default)
---------------------------------------

.. list-table:: Table of **argopy** data processing details in ``standard`` user mode |mode_standard|
    :header-rows: 1
    :stub-columns: 1

    * - Parameters
      - Dataset
      - Level of assessment (data mode)
      - Level of quality (QC flags)
      - Pressure error
      - Return variables
    * - Pressure, temperature, salinity
      - |ds_phy| + |ds_bgc|
      - real time, adjusted and delayed mode data: [R,A,D] modes
      - good or probably good values (QC=[1,2])
      - *not used*
      - all without jargon [a]_
    * - Radiometry parameters [b]_ and BBP700 [c]_
      - |ds_bgc|
      - real time, adjusted and delayed mode data: [R,A,D] modes
      - good or probably good values, estimated or changed values (QC=[1,2,5,8])
      - *not used*
      - all without jargon [a]_
    * - CDOM [d]_
      - |ds_bgc|
      - None allowed
      - None allowed
      - *not used*
      - all without jargon [a]_
    * - All other BGC parameters [e]_
      - |ds_bgc|
      - real time data with adjusted values, delayed mode data: [A,D] modes
      - good or probably good data, estimated or changed values (QC=[1,2,5,8])
      - *not used*
      - all without jargon [a]_


.. [a] The complete list is available with :class:`utils.list_standard_variables`. Note that DATA_MODE/PARAM_DATA_MODE and QC flags variables are retained while PARAM_ADJUSTED and PARAM variables are merged (i.e. PARAM_ADJUSTED is removed).
.. [b] The list of radiometry parameters is available with :class:`utils.list_radiometry_parameters`.
.. [c] Particle backscattering at 700 nanometers
.. [d] Concentration of coloured dissolved organic matter in seawater
.. [e] The complete list of BGC parameters is available with :class:`utils.list_bgc_s_parameters`.


.. tabs::

    .. tab:: |mode_standard| Example with |ds_phy| : core+deep missions

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='standard'):
                ds = argopy.DataFetcher(src='gdac').profile(6902746, 12).to_xarray()
                print(ds.data_vars)


    .. tab:: |mode_standard| Example with |ds_bgc| : BGC mission

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='standard'):
                ds = argopy.DataFetcher(src='erddap', ds='bgc').profile(5903248, 34).to_xarray()
                print(ds.data_vars)



|mode_research| Research mode
-----------------------------

.. list-table:: Table of **argopy** data processing details in ``research`` user mode |mode_research|
    :header-rows: 1
    :stub-columns: 1

    * - Parameters
      - Dataset
      - Level of assessment (data mode)
      - Level of quality (QC flags)
      - Pressure error
      - Return variables
    * - Pressure, temperature, salinity
      - |ds_phy| + |ds_bgc|
      - delayed mode data only: [D] mode
      - good values (QC=[1])
      - smaller than 20db
      - comprehensive minimum [a]_
    * - CDOM [d]_
      - |ds_bgc|
      - None allowed
      - None allowed
      - *not used*
      - comprehensive minimum [a]_
    * - All other BGC parameters [e]_
      - |ds_bgc|
      - delayed mode data only: [D] mode
      - good data, estimated or changed values (QC=[1,5,8])
      - *not used*
      - comprehensive minimum [a]_


.. [a] i.e.: float ID, profile number and direction and all parameter values, including error estimates
.. [b] The list of radiometry parameters is available with :class:`utils.list_radiometry_parameters`
.. [c] Particle backscattering at 700 nanometers
.. [d] Concentration of coloured dissolved organic matter in seawater
.. [e] The complete list of BGC parameters is available with :class:`utils.list_bgc_s_parameters`.


.. tabs::

    .. tab:: |mode_research| Example with |ds_phy| : core+deep missions

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='research'):
                ds = argopy.DataFetcher(src='gdac').profile(6902746, 12).to_xarray()
                print(ds.data_vars)


    .. tab:: |mode_research| Example with |ds_bgc| : BGC mission

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='research'):
                ds = argopy.DataFetcher(src='erddap', ds='bgc').profile(5903248, 34).to_xarray()
                print(ds.data_vars)


|mode_expert| Expert mode
-------------------------

No pre or post processing is performed, this user mode returns all the Argo data as they are in data source.


.. tabs::

    .. tab:: |mode_expert| Example with |ds_phy| : core+deep missions

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='expert'):
                ds = argopy.DataFetcher(src='gdac').profile(6902746, 12).to_xarray()
                print(ds.data_vars)


    .. tab:: |mode_expert| Example with |ds_bgc| : BGC mission

        .. ipython:: python
            :okwarning:

            import argopy
            with argopy.set_options(mode='expert'):
                ds = argopy.DataFetcher(src='gdac', ds='bgc').profile(5903248, 34).to_xarray()
                print(ds.data_vars)



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
