.. currentmodule:: argopy
.. _data-set:

Dataset
#######

|Profile count| |Profile BGC count|

.. |Profile count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json
.. |Profile BGC count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20BGC%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-BGC.json

.. hint::

    **argopy** makes a difference between the physical and biogeochemical parameters. To make sure you understand which data you're getting, have a look at this section.

.. contents:: Contents
   :local:


**Argo data are distributed as a single dataset.** It is referenced at https://doi.org/10.17882/42182.

But they are several Argo `missions <https://argo.ucsd.edu/about/mission>`_ with specific files and parameters that need special handling by **argopy**, namely:

- the core Argo Mission:  from floats that measure temperature, salinity, pressure down to 2000m,
- the `Deep Argo Mission <https://argo.ucsd.edu/expansion/deep-argo-mission>`_:  from floats that measure temperature, salinity, pressure down to 6000m,
- and the `BGC-Argo Mission <https://biogeochemical-argo.org>`_:  from floats that measure temperature, salinity, pressure and oxygen, pH, nitrate, chlorophyll, backscatter, irradiance down to 2000m.


Argo dataset available in **argopy**
************************************

In **argopy** we simply make the difference between physical and biogeochemical parameters in the Argo dataset. This is because the Deep Argo mission data are accessible following the same files and parameters than those from the Core mission. Only BGC-Argo data requires specific files and parameters.

In **argopy** you can thus get access to the following Argo data:

1. the **phy** dataset, for *physical* parameters.
    This dataset provides data from floats that measure temperature, salinity, pressure, without limitation in depth. It is available from all :ref:`Available data sources`.
    Since this is the most common Argo data subset it's selected with the ``phy`` keyword by default in **argopy**.

2. the **bgc** dataset, for *biogeochemical* parameters.
    This dataset provides data from floats that measure temperature, salinity, pressure and oxygen, pH, nitrate, chlorophyll, backscatter, irradiance, without limitation in depth.
    You can select this dataset with the keyword ``bgc`` and methods described below.

Selecting a dataset
*******************

You have several ways to specify which dataset you want to use:

-  **using argopy global options**:

.. ipython:: python
    :okwarning:

    import argopy
    argopy.set_options(dataset='bgc')

-  **with an option in a temporary context**:

.. ipython:: python
    :okwarning:

    import argopy
    with argopy.set_options(dataset='phy'):
        argopy.DataFetcher().profile(6904241, 12)

-  **with the `ds` argument in the data fetcher**:

.. ipython:: python
    :okwarning:

    argopy.DataFetcher(ds='phy').profile(6902746, 34)


.. note::

    In the future, we could consider to add more mission specific keywords for the ``dataset`` option and ``ds`` fetcher argument of :class:`DataFetcher`. This could be *deep* for instance. Please `raise an gitHub "issue" <https://github.com/euroargodev/argopy/issues/new>`_ if you may require such a new feature.

The **bgc** dataset
**********************
.. role:: python(code)
   :language: python

.. important::

    At this time, BGC parameters are only available in ``expert`` :ref:`user mode <user-mode>` and with the ``erddap`` :ref:`data source <data-sources>`.

All **argopy** features work with the **phy** dataset. However, they are some specific methods dedicated to the **bgc** dataset that we now describe.

Specifics in :class:`DataFetcher`
=================================

The `BGC-Argo Mission <https://biogeochemical-argo.org>`_ gathers data from floats that measure temperature, salinity, pressure and oxygen, pH, nitrate, chlorophyll, backscatter, irradiance down to 2000m. However, beyond this short BGC parameter list there exist in the Argo dataset **more than 120 BGC-related variables**. Therefore, in the :class:`DataFetcher` we implemented 2 specific arguments to handle BGC variables: ``params`` and ``measured``.

The ``params`` argument
-----------------------

With a :class:`DataFetcher`, the ``params`` argument can be used to specify which variables will be returned, *whatever their values or availability in BGC floats found in the data selection*.

By default, the ``params`` argument is set to the keyword ``all`` to return *all* variables found in the data selection. But the ``params`` argument can also be a single variable or a list of variables, in which case only these will be returned and all the others discarded.

.. dropdown:: Syntax example
    :icon: code
    :color: light
    :open:

    - To return data from a single BGC parameter, just add it as a string, for instance ``DOXY``
    - To return more than one BGC parameter, give them as a list of strings, for instance ``['DOXY', 'BBP700']``
    - To retrieve all available BGC parameters, you can omit the ``params`` argument (since this is the default value), or give it explicitly as ``all``.

    .. ipython:: python
        :okwarning:

        import argopy
        with argopy.set_options(dataset='bgc', src='erddap', mode='expert'):
            params = 'all'  # eg: 'DOXY' or ['DOXY', 'BBP700']
            f = argopy.DataFetcher(params=params)
            f = f.region([-75, -45, 20, 30, 0, 10, '2021-01', '2021-06'])
            f.load()

        print(f.data.argo, "\n")  # Easy print of N profiles and points

        print(list(f.data.data_vars))  # List of dataset variables

The ``measured`` argument
-------------------------

With a :class:`DataFetcher`, the ``measured`` argument can be used to specify which variables cannot be NaN and must return values. This is very useful to reduce a dataset to points where all or some variables are available.

By default, the ``measured`` argument is set to ``None`` for unconstrained parameter values. To the opposite, the keyword ``all`` requires that all variables found in the data selection cannot be NaNs. In between, you can specific one or more parameters to limit the constrain to a few variables.

.. dropdown:: Syntax example
    :icon: code
    :color: light
    :open:

    Let's impose that some variables cannot be NaNs, for instance ``DOXY`` and ``BBP700``:

    .. ipython:: python
        :okwarning:

        import argopy
        with argopy.set_options(dataset='bgc', src='erddap', mode='expert'):
            f = argopy.DataFetcher(params='all', measured=['DOXY', 'BBP700'])
            f = f.region([-75, -45, 20, 30, 0, 10, '2021-01', '2021-06'])
            f.load()

        print(f.data.argo, "\n")  # Easy print of N profiles and points

        print(list(f.data.data_vars))  # List of dataset variables

    We can see from ``f.data.argo.N_POINTS`` that the dataset is reduced compared to the previous version without constraints on variables **measured**.


Specifics in :class:`ArgoIndex`
===============================

All details and examples of the BGC specifics methods for :class:`ArgoIndex` can be found in: :ref:`metadata-index-bgc`.
