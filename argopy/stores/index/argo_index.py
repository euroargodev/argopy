import importlib


if importlib.util.find_spec("pyarrow") is not None:
    from .implementations.pyarrow.index import indexstore
else:
    from .implementations.pandas.index import indexstore


class ArgoIndex(indexstore):
    """Argo GDAC index store

    If Pyarrow is available, this class will use :class:`pyarrow.Table` as internal storage format; otherwise, a
    :class:`pandas.DataFrame` will be used.

    Shortcuts for ``host`` argument:

    - ``http`` or ``https`` for ``https://data-argo.ifremer.fr``
    - ``us-http`` or ``us-https`` for ``https://usgodae.org/pub/outgoing/argo``
    - ``ftp`` for ``ftp://ftp.ifremer.fr/ifremer/argo``
    - ``s3`` or ``aws`` for ``s3://argo-gdac-sandbox/pub/idx``

    Shortcuts for ``index_file`` argument:

    - ``core`` for the ``ar_index_global_prof.txt`` index file,
    - ``bgc-b`` for the ``argo_bio-profile_index.txt`` index file,
    - ``bgc-s`` for the ``argo_synthetic-profile_index.txt`` index file,
    - ``aux`` for the ``etc/argo-index/argo_aux-profile_index.txt`` index file.
    - ``meta`` for the ``ar_index_global_meta.txt`` index file.

    Examples
    --------
    .. code-block:: python
        :caption: An index store is instantiated with a host (any access path, local, http or ftp) and an index file

        idx = ArgoIndex()
        idx = ArgoIndex(host="https://data-argo.ifremer.fr")  # Default host
        idx = ArgoIndex(host="ftp://ftp.ifremer.fr/ifremer/argo", index_file="ar_index_global_prof.txt")  # Default index
        idx = ArgoIndex(index_file="bgc-s")  # Use keywords instead of exact file names
        idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="bgc-b", cache=True)  # Use cache for performances
        idx = ArgoIndex(host=".", index_file="dummy_index.txt", convention="core")  # Load your own index

    .. code-block:: python
        :caption: Full index methods and properties

        idx.load()
        idx.load(nrows=12)  # Only load the first N rows of the index
        idx.to_dataframe(index=True)  # Convert index to user-friendly :class:`pandas.DataFrame`
        idx.to_dataframe(index=True, nrows=2)  # Only returns the first nrows of the index
        idx.N_RECORDS  # Shortcut for length of 1st dimension of the index array
        idx.index  # internal storage structure of the full index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        idx.shape  # shape of the full index array
        idx.uri_full_index  # List of absolute path to files from the full index table column 'file'

    .. code-block:: python
        :caption: Search methods

        idx.query.wmo(1901393)
        idx.query.cyc(1)
        idx.query.wmo_cyc(1901393, [1,12])

        idx.query.lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        idx.query.lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        idx.query.date([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        idx.query.lon_lat([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        idx.query.box([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition

        idx.query.params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])  # Take a list of strings, only for BGC index !
        idx.query.parameter_data_mode({'BBP700': 'D', 'DOXY': ['A', 'D']})  # Take a dict.

        idx.query.profiler_type(845)
        idx.query.profiler_label('NINJA')

    .. code-block:: python
        :caption: Composing search methods

        idx.query.compose({'box': BOX, 'wmo': WMOs})
        idx.query.compose({'box': BOX, 'params': 'DOXY'})
        idx.query.compose({'box': BOX, 'params': (['DOXY', 'DOXY2'], {'logical': 'and'})})
        idx.query.compose({'params': 'DOXY', 'profiler_label': 'ARVOR'})

    .. code-block:: python
        :caption: Search result properties and methods

        idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
        idx.search  # Internal table with search results
        idx.uri  # List of absolute path to files from the search results table column 'file'

        idx.run()  # Run the search and save results in cache if necessary
        idx.to_dataframe()  # Convert search results to user-friendly :class:`pandas.DataFrame`
        idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results
        idx.to_indexfile("search_index.txt")  # Export search results to Argo standard index file

    .. code-block:: python
        :caption: List of file properties

        idx.read_wmo()
        idx.read_dac_wmo()
        idx.read_params()
        idx.read_domain()
        idx.read_files()

        idx.records_per_wmo()

    .. code-block:: python
        :caption: Misc

        idx.convention  # What is the expected index format (core vs BGC profile index)
        idx.cname
        idx.domain # the default read_domain() output, as a property
        idx.copy(deep=False)

    .. code-block:: python
        :caption: Iterate on :class:`argopy.ArgoFloat` instance

        for a_float in idx.iterfloats():
            print(a_float.WMO)
            ds = a_float.open_dataset('prof')

    .. code-block:: python
        :caption: Trajectory map

        idx = idx.query.wmo(6903091)
        idx.plot.trajectory()

    .. code-block:: python
        :caption: Trajectory map with custom arguments

        idx = ArgoIndex(index_file='bgc-s')
        idx.query.params('CHLA')

        idx.plot.trajectory(set_global=1,
                            add_legend=0,
                            traj=0,
                            cbar=False,
                            markersize=12,
                            markeredgesize=0.1,
                            dpi=120,
                            figsize=(20,20));

    .. code-block:: python
        :caption: Bar plot

        idx.plot.bar(by='dac', index=1)
        idx.plot.bar(by='profiler')


    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
