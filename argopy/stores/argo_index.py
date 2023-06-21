import importlib

if importlib.util.find_spec("pyarrow") is not None:
    from .argo_index_pa import indexstore_pyarrow as indexstore
else:
    from .argo_index_pd import indexstore_pandas as indexstore

class ArgoIndex(indexstore):
    """Argo GDAC index store

    If Pyarrow is available, this class will use :class:`pyarrow.Table` as internal storage format; otherwise, a
    :class:`pandas.DataFrame` will be used.

    convention_supported = ["ar_index_global_prof", "argo_bio-profile_index", "argo_synthetic-profile_index"]

    Examples
    --------

    An index store is instantiated with the access path (host) and the index file:

    >>> idx = ArgoIndex()
    >>> idx = ArgoIndex(host="ftp://ftp.ifremer.fr/ifremer/argo")
    >>> idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt")
    >>> idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt", cache=True)
    >>> idx = ArgoIndex(host=".", index_file="dummy_index.txt", convention="ar_index_global_prof")

    Full index methods and properties:

    >>> idx.load()
    >>> idx.load(nrows=12)  # Only load the first N rows of the index
    >>> idx.N_RECORDS  # Shortcut for length of 1st dimension of the index array
    >>> idx.index  # internal storage structure of the full index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
    >>> idx.shape  # shape of the full index array
    >>> idx.uri_full_index  # List of absolute path to files from the full index table column 'file'
    >>> idx.to_dataframe(index=True)  # Convert index to user-friendly :class:`pandas.DataFrame`
    >>> idx.to_dataframe(index=True, nrows=2)  # Only returns the first nrows of the index

    Search methods and properties:

    >>> idx.search_wmo(1901393)
    >>> idx.search_cyc(1)
    >>> idx.search_wmo_cyc(1901393, [1,12])
    >>> idx.search_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
    >>> idx.search_lat_lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
    >>> idx.search_lat_lon_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
    >>> idx.search_params(['C1PHASE_DOXY', 'DOWNWELLING_PAR'])  # Take a list of strings, only for BGC index !
    >>> idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
    >>> idx.search  # Internal table with search results
    >>> idx.uri  # List of absolute path to files from the search results table column 'file'
    >>> idx.run()  # Run the search and save results in cache if necessary
    >>> idx.to_dataframe()  # Convert search results to user-friendly :class:`pandas.DataFrame`
    >>> idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results
    >>> idx.to_indexfile("search_index.txt")  # Export search results to Argo standard index file

    Misc:

    >>> idx.convention  # What is the expected index format (core vs BGC profile index)
    >>> idx.cname
    >>> idx.read_wmo
    >>> idx.read_params
    >>> idx.records_per_wmo

    """
    pass
