.. currentmodule:: argopy

.. _tools-argoindex:

Argo Index store
================

If you are familiar with Argo index csv files, you may be interested in using directly the Argo index store :class:`ArgoIndex`.

If Pyarrow is installed, this store will rely on :class:`pyarrow.Table` as internal storage format for the index, otherwise it will fall back on :class:`pandas.DataFrame`. Loading the full Argo profile index takes about 2/3 secs with Pyarrow, while it can take up to 6/7 secs with Pandas.

All index store methods and properties are documented in the :class:`ArgoIndex` API page.

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

Usage
-----

You create an index store with default or custom options:

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

Note that you can use GDAC host shortcut names:

- ``https://data-argo.ifremer.fr``, shortcut with ``http`` or ``https``
- ``https://usgodae.org/pub/outgoing/argo``, shortcut with ``us-http`` or ``us-https``
- ``ftp://ftp.ifremer.fr/ifremer/argo``, shortcut with ``ftp``
- ``s3://argo-gdac-sandbox/pub/idx``, shortcut with ``s3`` or ``aws``

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

    idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

Here the list of all methods to **search** the index:

.. code-block:: python

    idx.query.wmo(1901393)
    idx.query.cyc(1)
    idx.query.wmo_cyc(1901393, [1,12])
    idx.query.date([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only time is used
    idx.query.lat_lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition, only lat/lon is used
    idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
    idx.query.params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])  # Only for BGC profile index
    idx.query.parameter_data_mode({'BBP700': 'D'})  # Only for BGC profile index
    idx.query.profiler_type(845)
    idx.query.profiler_label('NINJA')

and this is how to compose several search criteria:

.. code-block:: python

    idx.query.compose({'box': BOX, 'wmo': WMOs})
    idx.query.compose({'box': BOX, 'params': 'DOXY'})
    idx.query.compose({'box': BOX, 'params': (['DOXY', 'DOXY2'], {'logical': 'and'})})
    idx.query.compose({'params': 'DOXY', 'profiler_label': 'ARVOR'})


At last, here is the list of methods and properties for **search results**:

.. code-block:: python

    idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
    idx.to_dataframe()  # Convert search results to user-friendly :class:`pandas.DataFrame`
    idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results
    idx.to_indexfile("search_index.txt")  # Export search results to Argo standard index file
    idx.search  # Internal table with search results
    idx.uri  # List of absolute path to files from the search results table column 'file'


.. _metadata-index-bgc:

Usage with **bgc** index
------------------------

The **argopy** index store supports the Bio, Synthetic and Auxiliary Profile directory files:

.. ipython:: python
    :okwarning:

    idx = ArgoIndex(index_file="argo_bio-profile_index.txt").load()
    # idx = ArgoIndex(index_file="argo_synthetic-profile_index.txt").load()
    idx

.. hint::

    In order to load one BGC-Argo profile index, you can use either ``bgc-b`` or ``bgc-s`` keywords to load the ``argo_bio-profile_index.txt`` or ``argo_synthetic-profile_index.txt`` index files.

All methods presented :ref:`above <metadata-index>` are valid with BGC index, but a BGC index store comes with additional search possibilities for parameters and parameter data modes.

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
