.. _data_set:

Dataset (🟡 🟢 🔵)
===================

|Profile count| |Profile BGC count|

.. |Profile count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json
.. |Profile BGC count| image:: https://img.shields.io/endpoint?label=Number%20of%20Argo%20BGC%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-BGC.json


**Argo data are distributed as a single dataset.** It is referenced at https://doi.org/10.17882/42182.

But they are several Argo `missions <https://argo.ucsd.edu/about/mission>`_ with specific files and parameters that need special handling by **argopy**, namely:

- 🟡 the core Argo Mission:  from floats that measure temperature, salinity, pressure down to 2000m,
- 🟢 the `BGC-Argo Mission <https://biogeochemical-argo.org>`_:  from floats that measure temperature, salinity, pressure and oxygen, pH, nitrate, chlorophyll, backscatter, irradiance down to 2000m,
- 🔵 the `Deep Argo Mission <https://argo.ucsd.edu/expansion/deep-argo-mission>`_:  from floats that measure temperature, salinity, pressure down to 6000m.


Available Argo dataset
----------------------

In **argopy** we simply make the difference between physical and biogeochemical parameters in the Argo dataset. This is because the Deep Argo mission data are accessible following the same files and parameters than those from the Core mission. Only BGC-Argo data requires specific files and parameters.

In **argopy** you can thus get access to the following Argo data:

1. 🟡+ 🔵 the **phy** dataset, for *physical* parameters.
    This dataset provides data from floats that measure temperature, salinity, pressure, without limitation in depth. It is available from all :ref:`Available data sources`.
    Since this is the most common Argo data subset it's selected with the ``phy`` keyword by default in **argopy**.

2. 🟢 the **bgc** dataset, for *biogeochemical* parameters.
    This dataset provides data from floats that measure temperature, salinity, pressure and oxygen, pH, nitrate, chlorophyll, backscatter, irradiance, without limitation in depth.
    You can select this dataset with the keyword ``bgc`` and methods described below.

Selecting a dataset
-------------------

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

    In the future, we could consider to add more mission specific keywords for the ``dataset`` option and ``ds`` fetcher argument of :class:`argopy.DataFetcher`. This could be *deep* for instance. Please `raise an gitHub "issue" <https://github.com/euroargodev/argopy/issues/new>`_ if you may require such a new feature.
