.. currentmodule:: argopy

.. _tools-argoindex:

Argo Index store
================

If you are familiar with Argo index csv files, you may be interested in using directly the Argo index store :class:`ArgoIndex`.

If Pyarrow is installed, this store will rely on :class:`pyarrow.Table` as internal storage format for the index, otherwise it will fall back on :class:`pandas.DataFrame`. Loading the full Argo profile index takes about 2/3 secs with Pyarrow, while it can take up to 6/7 secs with Pandas.

All index store methods and properties are documented in the :class:`ArgoIndex` API page.

.. contents::
   :local:

Index file supported
--------------------

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
    * - Metadata
      - ar_index_global_meta.txt
      - ✅
    * - Auxiliary
      - etc/argo-index/argo_aux-profile_index.txt
      - ✅
    * - Trajectory
      - ar_index_global_traj.txt
      - ❌
    * - Bio-Trajectory
      - argo_bio-traj_index.txt
      - ❌
    * - Technical
      - ar_index_global_tech.txt
      - ❌
    * - Greylist
      - ar_greylist.txt
      - ❌

Index files support can be added on demand. `Click here to raise an issue if you'd like to access other index files <https://github.com/euroargodev/argopy/issues/new>`_.

.. _metadata-index:

Create an Index store
---------------------

You can create an index store with two arguments: a GDAC host (local or remote) and an index file name. Both arguments have default values to the http Ifremer GDAC and core index.

.. ipython:: python
    :okwarning:

    from argopy import ArgoIndex

    idx = ArgoIndex()
    # or:
    # ArgoIndex(index_file="argo_bio-profile_index.txt")
    # ArgoIndex(index_file="bgc-s")  # can use keyword instead of file name: core, bgc-b, bgc-b

    # ArgoIndex(host="ftp://ftp.ifremer.fr/ifremer/argo")
    # ArgoIndex(host="https://data-argo.ifremer.fr", index_file="core")
    # ArgoIndex(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt", cache=True)

You can also use the following shortcuts:

.. list-table:: Table of :class:`ArgoIndex` host name shortcuts
    :header-rows: 1
    :stub-columns: 1

    * - Full host
      - Shortcut
    * - ``https://data-argo.ifremer.fr``
      - ``http`` or ``https``
    * - ``https://usgodae.org/pub/outgoing/argo``
      - ``us-http`` or ``us-https``
    * - ``ftp://ftp.ifremer.fr/ifremer/argo``
      - ``ftp``
    * - ``s3://argo-gdac-sandbox/pub/idx``
      - ``s3`` or ``aws``

.. list-table:: Table of :class:`ArgoIndex` index file name shortcuts
    :header-rows: 1
    :stub-columns: 1

    * - Index file
      - Shortcut
    * - ``ar_index_global_prof.txt``
      - ``core``
    * - ``argo_bio-profile_index.txt``
      - ``bgc-b``
    * - ``argo_synthetic-profile_index.txt``
      - ``bgc-s``
    * - ``ar_index_global_meta.txt``
      - ``meta``
    * - ``etc/argo-index/argo_aux-profile_index.txt``
      - ``aux``


Load the Index
--------------

All :class:`ArgoIndex` methods will automatically trigger the index content download if necessary.

You can use the ``cache=True`` option if you want to limit the number of download to only one per day (the default cache expiration delay is 24 hours). **There is no caching by default**.

You can also trigger the index content download in memory with:

.. ipython:: python
    :okwarning:

    idx.load()  # Load the full index in memory

    # or
    # idx.load(nrows=1000)  # Only load the first N rows of the index

Full Index properties
---------------------

Once you loaded data to a :class:`ArgoIndex` instance, the following attributes and methods are available:

.. code-block:: python

    # Shortcut for length of 1st dimension of the index array
    idx.N_RECORDS

    # Convert index to a user-friendly Argo csv-like :class:`pandas.DataFrame`
    idx.to_dataframe(index=True)

    # Only returns the first nrows of the index
    idx.to_dataframe(index=True, nrows=2)

    # Access internal storage structure of the full index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
    idx.index

    # List of absolute path to files from the full index table column 'file'
    idx.uri_full_index

You can also list unique properties, see :ref:`List unique Index properties` below.

Search the Index
----------------

.. currentmodule:: argopy

If you need to reduce the list of files from an index, notably those matching a set of search criteria, you can use the :class:`ArgoIndex.query` extension.

For instance, to reduce the list of files to those with latitude, longitude and date within a rectangular box:

.. ipython:: python
    :okwarning:

    idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

Here is the list of methods available to search an index:

.. code-block:: python

    idx.query.wmo(1901393)
    idx.query.wmo([6902915, 1901393])
    idx.query.cyc(1)
    idx.query.cyc([1, 12])
    idx.query.wmo_cyc(1901393, [1,12])

    idx.query.lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only lat/lon is used
    idx.query.lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only lat/lon is used
    idx.query.date([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only time is used

    idx.query.wmo_cyc(1901393, [1,12])
    idx.query.lon_lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only lat/lon is used
    idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition

    idx.query.params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])  # Only for BGC profile index
    idx.query.parameter_data_mode({'BBP700': 'D'})  # Only for BGC profile index

    idx.query.profiler_type(845)
    idx.query.profiler_type([845, 856])
    idx.query.profiler_label('NINJA')
    idx.query.profiler_label(['NINJA', 'SOLO-D deep'])

    idx.query.institution_code('IF')
    idx.query.institution_code(['IF', 'JA'])
    idx.query.institution_name('Canada')
    idx.query.institution_name(['Canada', 'INCOIS'])
    idx.query.dac('coriolis')
    idx.query.dac(['meds', 'aoml'])

You will note that the space/time search methods ``lon``, ``lat``, ``date``, ``lon_lat`` and ``box`` all take the same argument that is a list with [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max] values.

If you need to compose a query with several search criteria, you can use the :meth:`ArgoIndex.query.compose` method like this:

.. code-block:: python

    idx.query.compose({'box': BOX, 'wmo': WMOs})
    idx.query.compose({'box': BOX, 'params': 'DOXY'})
    idx.query.compose({'box': BOX, 'params': (['DOXY', 'DOXY2'], {'logical': 'and'})})
    idx.query.compose({'params': 'DOXY', 'profiler_label': 'ARVOR'})


Once you performed a query on a :class:`ArgoIndex` instance, the following attributes and methods are available:

.. code-block:: python

    idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
    idx.to_dataframe()  # Convert search results to a user-friendly Argo csv-like :class:`pandas.DataFrame`
    idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results
    idx.to_indexfile("search_index.txt")  # Export search results to Argo standard index file format
    idx.search  #  internal storage structure of the search-reduced index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
    idx.uri  # List of absolute path to files from the search results table column 'file'

List unique Index properties
----------------------------

It is often useful to be able to list unique occurrences of some index properties. These are available:

.. code-block:: python

    idx.read_wmo()
    idx.read_dac_wmo()
    idx.read_params()
    idx.read_domain()
    idx.read_files()

    idx.records_per_wmo()

.. hint::

    Each of these methods will use the search result by default, and if no search was ran, will fall back on using the full index.

.. _metadata-index-bgc:

Specifics to **bgc** Index
--------------------------

The **argopy** index store supports the Bio, Synthetic and Auxiliary Profile directory files:

.. ipython:: python
    :okwarning:

    idx = ArgoIndex(index_file="bgc-b").load()
    # idx = ArgoIndex(index_file="bgc-s").load()
    idx

All methods presented :ref:`above <metadata-index>` are valid with a BGC index, but a BGC index store comes with additional search possibilities for parameters and parameter data modes.

Two specific index variables are only available with BGC-Argo index files: ``PARAMETERS`` and ``PARAMETER_DATA_MODE``. We thus implemented the :meth:`ArgoIndex.query.params` and :meth:`ArgoIndex.query.parameter_data_mode` methods. These method allow to search for (i) profiles with one or more specific parameters and (ii) profiles with parameters in one or more specific data modes.

.. dropdown:: Syntax for  :meth:`ArgoIndex.query.params`
    :icon: code
    :color: light
    :open:

    .. tab-set::

        .. tab-item:: 1. Load a BGC index

            .. ipython:: python
                :okwarning:

                from argopy import ArgoIndex
                idx = ArgoIndex(index_file='bgc-s').load()
                idx

        .. tab-item:: 2. Search for BGC parameters

            You can search for one parameter:

            .. ipython:: python
                :okwarning:

                idx.query.params('DOXY')

            Or you can search for several parameters:

            .. ipython:: python
                :okwarning:

                idx.query.params(['DOXY', 'CDOM'])

            Note that a multiple parameters search will return profiles with *all* parameters. To search for profiles with *any* of the parameters, use:

            .. ipython:: python
                :okwarning:

                idx.query.params(['DOXY', 'CDOM'], logical='or')


.. dropdown:: Syntax for  :meth:`ArgoIndex.query.parameter_data_mode`
    :icon: code
    :color: light
    :open:

    .. tab-set::

        .. tab-item:: 1. Load a BGC index

            .. ipython:: python
                :okwarning:

                from argopy import ArgoIndex
                idx = ArgoIndex(index_file='bgc-b').load()
                idx

        .. tab-item:: 2. Search for BGC parameter data mode

            You can search one mode for a single parameter:

            .. ipython:: python
                :okwarning:

                idx.query.parameter_data_mode({'BBP700': 'D'})

            You can search several modes for a single parameter:

            .. ipython:: python
                :okwarning:

                idx.query.parameter_data_mode({'DOXY': ['R', 'A']})

            You can search several modes for several parameters:

            .. ipython:: python
                :okwarning:

                idx.query.parameter_data_mode({'BBP700': 'D', 'DOXY': 'D'}, logical='and')

            And mix all of these as you wish:

            .. ipython:: python
                :okwarning:

                idx.query.parameter_data_mode({'BBP700': ['R', 'A'], 'DOXY': 'D'}, logical='or')
