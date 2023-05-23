.. currentmodule:: argopy
.. _metadata_fetching:

Argo meta-data
==============

.. contents::
   :local:

Index of profiles
-----------------
.. currentmodule:: argopy

Since the Argo measurements dataset is quite complex, it comes with a collection of index files, or lookup tables with meta data. These index help you determine what you can expect before retrieving the full set of measurements.

**argopy** provides two methods to work with Argo index files: one is high-level and works like the data fetcher, the other is low-level and works like a "store".

Fetcher: High-level Argo index access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**argopy** has a specific fetcher for index files:

.. ipython:: python
    :okwarning:

    from argopy import IndexFetcher as ArgoIndexFetcher

You can use the Index fetcher with the ``region`` or ``float`` access points, similarly to data fetching:

.. ipython:: python
    :suppress:

    import argopy
    ftproot = argopy.tutorial.open_dataset('gdac')[0]
    argopy.set_options(ftp=ftproot)

.. ipython:: python
    :okwarning:

    idx = ArgoIndexFetcher(src='gdac').float(2901623).load()
    idx.index

Alternatively, you can use :meth:`argopy.IndexFetcher.to_dataframe()`:

.. ipython:: python
    :okwarning:

    idx = ArgoIndexFetcher(src='gdac').float(2901623)
    df = idx.to_dataframe()

The difference is that with the `load` method, data are stored in memory and not fetched on every call to the `index` attribute.

The index fetcher has pretty much the same methods than the data fetchers. You can check them all here: :class:`argopy.fetchers.ArgoIndexFetcher`.


Store: Low-level Argo Index access
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The IndexFetcher shown above is a user-friendly layer on top of our internal Argo index file store. But if you are familiar with Argo index files and/or cares about performances, you may be interested in using directly the Argo index store.
At this point, we implemented an index store that can rely on 2 differents internal storage format for the index: one is using :class:`pyarrow.Table` and the other :class:`pandas.DataFrame`.

All index store methods and properties are fully documented in :class:`argopy.stores.indexstore_pa` and :class:`argopy.stores.indexstore_pd`.

Usage
"""""

First, you should select which internal storage format you want. Don't worry, they both provide the same user API.

.. note::

    To improve performances, we recommend to use the Pyarrow index store :class:`argopy.stores.indexstore_pa`. Loading the full Argo profile index takes about 2/3 secs with Pyarrow, while it can take up to 6/7 secs with Pandas.

.. ipython:: python
    :okwarning:

    from argopy.stores import indexstore_pd as indexstore  # Rely on Pandas
    # or:
    # from argopy.stores import indexstore_pa as indexstore  # Rely on Pyarrow

Then, you create the index store with default or custom options:

.. ipython:: python
    :okwarning:

    idx = indexstore()
    # or:
    # indexstore(host="ftp://ftp.ifremer.fr/ifremer/argo")
    # indexstore(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt")
    # indexstore(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt", cache=True)

You can then trigger loading of the index content:

.. ipython:: python
    :okwarning:

    idx.load()  # Load the full index in memory

Here is the list of methods and properties of the **full index**:

.. code-block:: python

    idx.load(nrows=12)  # Only load the first N rows of the index
    idx.N_RECORDS  # Shortcut for length of 1st dimension of the index array
    idx.to_dataframe(index=True)  # Convert index to user-friendly :class:`pandas.DataFrame`
    idx.to_dataframe(index=True, nrows=2)  # Only returns the first nrows of the index
    idx.index  # internal storage structure of the full index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
    idx.uri_full_index  # List of absolute path to files from the full index table column 'file'


They are several methods to **search** the index, for instance:

.. ipython:: python
    :okwarning:

    idx.search_lat_lon_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

Here the list of all methods to **search** the index:

.. code-block:: python

    idx.search_wmo(1901393)
    idx.search_cyc(1)
    idx.search_wmo_cyc(1901393, [1,12])
    idx.search_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only time is used
    idx.search_lat_lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only lat/lon is used
    idx.search_lat_lon_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition


And finally the list of methods and properties for **search results**:

.. code-block:: python

    idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
    idx.to_dataframe()  # Convert search results to user-friendly :class:`pandas.DataFrame`
    idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results
    idx.to_indexfile("search_index.txt")  # Export search results to Argo standard index file
    idx.search  # Internal table with search results
    idx.uri  # List of absolute path to files from the search results table column 'file'


.. hint::

    The **argopy** index store supports the Bio and Synthetic Profile directory files:

    .. ipython:: python
        :okwarning:

        idx = indexstore(index_file="argo_bio-profile_index.txt").load()
        # idx = indexstore(index_file="argo_synthetic-profile_index.txt").load()
        idx

    This BGC index store comes with an additional search possibility for parameters:

    .. ipython:: python
        :okwarning:

        idx.search_params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])

    .. ipython:: python
        :okwarning:

        idx.to_dataframe()


Index file supported
""""""""""""""""""""

The table below summarize the **argopy** support status of all Argo index files:

.. list-table:: **argopy** GDAC index file support status
    :header-rows: 1
    :stub-columns: 1

    * -
      - Index file
      - Supported
    * - Profile
      - ar_index_global_prof.txt
      - ✅
    * - Synthetic-Profile
      - argo_synthetic-profile_index.txt
      - ✅
    * - Bio-Profile
      - argo_bio-profile_index.txt
      - ✅
    * - Trajectory
      - ar_index_global_traj.txt
      - 🔜
    * - Bio-Trajectory
      - argo_bio-traj_index.txt
      - 🔜
    * - Metadata
      - ar_index_global_meta.txt
      - ❌
    * - Technical
      - ar_index_global_tech.txt
      - ❌
    * - Greylist
      - ar_greylist.txt
      - ❌

Index files support can be added on demand. `Click here to raise an issue if you'd like to access other index files <https://github.com/euroargodev/argopy/issues/new>`_.


Reference tables
----------------
.. currentmodule:: argopy.utilities

The Argo netcdf format is strict and based on a collection of variables fully documented and conventioned. All reference tables can be found in the `Argo user manual <https://doi.org/10.13155/29825>`_.

However, a machine-to-machine access to these tables is often required. This is possible thanks to the work of the **Argo Vocabulary Task Team (AVTT)** that is a team of people responsible for the `NVS <https://github.com/nvs-vocabs>`_ collections under the Argo Data Management Team governance.

.. note::

    The GitHub organization hosting the AVTT is the 'NERC Vocabulary Server (NVS)', aka 'nvs-vocabs'. This holds a list of NVS collection-specific GitHub repositories. Each Argo GitHub repository is called after its corresponding collection ID (e.g. R01, RR2, R03 etc.). `The current list is given here <https://github.com/nvs-vocabs?q=argo&type=&language=&sort=name>`_.

    The management of issues related to vocabularies managed by the Argo Data Management Team is done on this `repository <https://github.com/nvs-vocabs/ArgoVocabs>`_.

**argopy** provides the utility class :class:`ArgoNVSReferenceTables` to easily fetch and get access to all Argo reference tables. If you already know the name of the reference table you want to retrieve, you can simply get it like this:

.. ipython:: python
    :okwarning:

    from argopy import ArgoNVSReferenceTables
    NVS = ArgoNVSReferenceTables()
    NVS.tbl('R01')

The reference table is returned as a :class:`pandas.DataFrame`. If you want the exact name of this table:

.. ipython:: python

    NVS.tbl_name('R01')

On the other hand, if you want to retrieve all reference tables, you can do it with the :meth:`ArgoNVSReferenceTables.all_tbl` method. It will return a dictionary with table short names as key and :class:`pandas.DataFrame` as values.

.. ipython:: python

    all = NVS.all_tbl()
    all.keys()

Deployment Plan
---------------
.. currentmodule:: argopy.utilities

It may be useful to be able to retrieve meta-data from Argo deployments. **argopy** can use the `OceanOPS API for metadata access <https://www.ocean-ops.org/api/swagger/?url=https://www.ocean-ops.org/api/1/oceanops-api.yaml>`_ to retrieve these information. The returned deployment `plan` is a list of all Argo floats ever deployed, together with their deployment location, date, WMO, program, country, float model and current status.

To fetch the Argo deployment plan, **argopy** provides a dedicated utility class: :class:`OceanOPSDeployments` that can be used like this:

.. ipython:: python
    :okwarning:

    from argopy import OceanOPSDeployments

    deployment = OceanOPSDeployments()

    df = deployment.to_dataframe()
    df

:class:`OceanOPSDeployments` can also take an index box definition as argument in order to restrict the deployment plan selection to a specific region or period:

.. code-block:: python

    deployment = OceanOPSDeployments([-90, 0, 0, 90])
    # deployment = OceanOPSDeployments([-20, 0, 42, 51, '2020-01', '2021-01'])
    # deployment = OceanOPSDeployments([-180, 180, -90, 90, '2020-01', None])

Note that if the starting date is not provided, it will be set automatically to the current date.

Last, :class:`OceanOPSDeployments` comes with a plotting method:

.. code-block:: python

    fig, ax = deployment.plot_status()

.. image:: _static/scatter_map_deployment_status.png


.. note:: The list of possible deployment status name/code is given by:

    .. code-block:: python

        OceanOPSDeployments().status_code

    =========== == ====
    Status      Id Description
    =========== == ====
    PROBABLE    0  Starting status for some platforms, when there is only a few metadata available, like rough deployment location and date. The platform may be deployed
    CONFIRMED   1  Automatically set when a ship is attached to the deployment information. The platform is ready to be deployed, deployment is planned
    REGISTERED  2  Starting status for most of the networks, when deployment planning is not done. The deployment is certain, and a notification has been sent via the OceanOPS system
    OPERATIONAL 6  Automatically set when the platform is emitting a pulse and observations are distributed within a certain time interval
    INACTIVE    4  The platform is not emitting a pulse since a certain time
    CLOSED      5  The platform is not emitting a pulse since a long time, it is considered as dead
    =========== == ====

