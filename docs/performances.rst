Performances
============

.. contents::
   :local:

To improve **argopy** data fetching performances, several solutions are available:

-  :ref:`cache` your fetched data, i.e. save your request locally so that you don’t have to fetch it again,
-  Use the :ref:`parallel` argument, i.e. fetch chunks of independent data simultaneously (e.g. to be used with a Dask cluster),
-  Load data lazily with the :ref:`lazy` argument and using `kerchunk <https://fsspec.github.io/kerchunk/>`_.

These solutions are explained in details below.

.. _cache:

Caching
-------

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher


Caching data
~~~~~~~~~~~~

If you want to avoid retrieving the same data several times during a
working session, or if you fetched a large amount of data, you may want
to temporarily save data in a cache file.

You can cache fetched data with the fetchers option ``cache``.

**Argopy** cached data are persistent, meaning that they are stored
locally on files and will survive execution of your script with a new
session. **Cached data have an expiration time of one day**, since this
is the update frequency of most data sources. This will ensure you
always have the last version of Argo data.

All data and meta-data (index) fetchers have a caching system.

The argopy default cache folder is under your home directory at
``~/.cache/argopy``.

But you can specify the path you want to use in several ways:

-  with **argopy** global options:

.. code:: python

   argopy.set_options(cachedir='mycache_folder')

-  in a temporary context:

.. code:: python

   with argopy.set_options(cachedir='mycache_folder'):
       f = DataFetcher(cache=True)

-  when instantiating the data fetcher:

.. code:: python

   f = DataFetcher(cache=True, cachedir='mycache_folder')

.. warning::

  You really need to set the ``cache`` option to ``True``. Specifying only the ``cachedir`` won't trigger caching !

Clearing the cache
~~~~~~~~~~~~~~~~~~

If you want to manually clear your cache folder, and/or make sure your
data are newly fetched, you can do it at the fetcher level with the
``clear_cache`` method.

Start to fetch data and store them in cache:

.. ipython:: python
    :okwarning:

    argopy.set_options(cachedir='mycache_folder')

    fetcher1 = DataFetcher(cache=True).profile(6902746, 34).load()

Fetched data are in the local cache folder:

.. ipython:: python
    :okwarning:

    import os
    os.listdir('mycache_folder')

where we see hash entries for the newly fetched data and the cache
registry file ``cache``.

We can then fetch something else using the same cache folder:

.. ipython:: python
    :okwarning:

    fetcher2 = DataFetcher(cache=True).profile(1901393, 1).load()

All fetched data are cached:

.. ipython:: python
    :okwarning:

    os.listdir('mycache_folder')

Note the new hash file from *fetcher2* data.

It is important to note that we can safely clear the cache from the
first *fetcher1* data without removing *fetcher2* data:

.. ipython:: python
    :okwarning:

    fetcher1.clear_cache()
    os.listdir('mycache_folder')

By using the fetcher level clear cache, you make sure that only data
fetched with it are removed, while other fetched data (with other
fetchers for instance) will stay in place.

If you want to clear the entire cache folder, whatever the fetcher used,
do it at the package level with:

.. ipython:: python
    :okwarning:

    argopy.clear_cache()
    os.listdir('mycache_folder')

.. _parallel:

Parallel data fetching
----------------------

Sometimes you may find that your request takes a long time to fetch, or simply does not even succeed. This is probably
because you’re trying to fetch a large amount of data.

In this case, you can try to let **argopy** chunks your request into smaller pieces and have them fetched in parallel
for you. This is done with the data fetcher argument, or global option, ``parallel``.

Parallelization can be tuned using arguments ``chunks`` and ``chunks_maxsize``.

This goes by default like this:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher

    # Define a box to load (large enough to trigger chunking):
    box = [-60, -30, 40.0, 60.0, 0.0, 100.0, "2007-01-01", "2007-04-01"]
    
    # Instantiate a parallel fetcher:
    f = DataFetcher(parallel=True).region(box)

Note that you can also use the option ``progress`` to display a progress bar during fetching.

Then, simply trigger data fetching as usual:

.. ipython:: python
    :okwarning:

    %%time
    ds = f.to_xarray()  # or .load().data



Parallelization methods
~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: v1.0.0

    All data sources are now compatible with each parallelization methods !


3 methods are available to set-up your data fetching requests in parallel:

1. `multi-threading <https://en.wikipedia.org/wiki/Multithreading_(computer_architecture)>`_ with a :class:`concurrent.futures.ThreadPoolExecutor`,
2. `multi-processing <https://en.wikipedia.org/wiki/Multiprocessing>`_ with a :class:`concurrent.futures.ProcessPoolExecutor`,
3. A `Dask Cluster <https://docs.dask.org/en/stable/deploying.html>`_ identified by its `client <https://distributed.dask.org/en/latest/client.html>`_.

The **argopy** parallelization method is set with the ``parallel`` option (global or of the fetcher), which can take one of the following values:

- a boolean ``True`` or ``False``,
- a string: ``thread`` or ``process``,
- or a Dask ``client`` object.

In the case of setting a ``parallel=True`` boolean value, **argopy** will rely on using the default parallelization method defined by the option ``parallel_default_method``.

You have several ways to specify which parallelization methods you want to use:

-  **using argopy global options**:

.. ipython:: python
    :okwarning:

    argopy.set_options(parallel=True)  # Rq: Fall back on using: parallel_default_method='thread'

-  **in a temporary context**:

.. ipython:: python
    :okwarning:

    with argopy.set_options(parallel='process'):
        fetcher = DataFetcher()

-  **with an argument in the data fetcher**:

.. ipython:: python
    :okwarning:

    fetcher = DataFetcher(parallel='process')


.. caution::
    To parallelize your fetcher is useful to handle large region of data,
    but it can also add a significant overhead on *reasonable* size
    requests that may lead to degraded performances. So, **we do not
    recommend for you to use the parallel option systematically**.

    Benchmarking the current **argopy** processing chain has shown that most of the time necessary to fetch data is
    spent in waiting response for the data server and in merging chunks of data. There is currently no possibility
    to avoid chunks merging and the data server response time is out of scope for **argopy**.

.. caution::
    You may have different dataset sizes with and without the
    ``parallel`` option. This may happen if one of the chunk data
    fetching fails. By default, data fetching of multiple resources fails
    with a warning. You can change this behaviour with the option
    ``errors`` of the ``to_xarray()`` fetcher methods, just set it to
    ``raise`` like this:

    .. code:: python

      DataFetcher(parallel=True).region(BOX).to_xarray(errors='raise')


    You can also use ``silent`` to simply hide all messages during fetching.


Number of chunks: ``chunks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see how many chunks your request has been split into, you can look at
the ``uri`` property of the fetcher, it gives you the list of paths
toward data:

.. ipython:: python
    :okwarning:

    # Create a large box:
    box = [-60, 0, 0.0, 60.0, 0.0, 500.0, "2007", "2010"]

    # Init a parallel fetcher:
    fetcher = DataFetcher(parallel=True).region(box)

    print(len(fetcher.uri))

To control chunking, you can use the ``chunks`` option that specifies the number of chunks in each of the *direction*:

-  ``lon``, ``lat``, ``dpt`` and ``time`` for a **region** fetching,
-  ``wmo`` for a **float** and **profile** fetching.

Example:

.. ipython:: python
    :okwarning:

    fetcher = DataFetcher(parallel=True, chunks={'lon': 5}).region(box)
    len(fetcher.uri) # Check the number of chunks

This creates 195 chunks, and 5 along the longitudinale direction, as
requested.

When the ``chunks`` option is not specified for a given *direction*, it
relies on auto-chunking using pre-defined chunk maximum sizes (see
below). In the case above, auto-chunking appends also along latitude,
depth and time; this explains why we have 195 and not only 5 chunks.

To chunk the request along a single direction, set explicitly all the
other directions to ``1``:

.. ipython:: python
    :okwarning:

    # Init a parallel fetcher:
    fetcher = DataFetcher(parallel=True,
                          chunks={'lon': 5, 'lat':1, 'dpt':1, 'time':1}).region(box)
    
    # Check the number of chunks:
    len(fetcher.uri)

We now have 5 chunks along longitude, check out the URLs parameter in
the list of URIs:

.. ipython:: python
    :okwarning:

    for uri in fetcher.uri:
        print("&".join(uri.split("&")[1:-2])) # Display only the relevant URL part

.. note::
    You may notice that if you run the last command with the `argovis` fetcher, you will still have more than 5 chunks (i.e. 65). This is because `argovis` is limited to 3 months length requests. So, for this request that is 3 years long, argopy ends up with 13 chunks along time, times 5 chunks in longitude, leading to 65 chunks in total.

.. warning::
    The ``gdac`` fetcher and the ``float`` and ``profile`` access points of the ``argovis`` fetcher use a list of resources than are not chunked but fetched in parallel using a batch queue.

Size of chunks: ``chunks_maxsize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default chunk size for each access point dimensions are:

====================== ==================
Access point dimension Maximum chunk size
====================== ==================
🗺 region / **lon**       20 deg
🗺 region / **lat**       20 deg
🗺 region / **dpt**       500 m or db
🗺 region / **time**      90 days
🤖 float / **wmo**        5
⚓ profile / **wmo**      5
====================== ==================

These default values are used to chunk data when the ``chunks``
parameter key is set to ``auto``.

But you can modify the maximum chunk size allowed in each of the
possible directions. This is done with the option
``chunks_maxsize``.

For instance if you want to make sure that your chunks are not larger
then 100 meters (db) in depth (pressure), you can use:

.. ipython:: python
    :okwarning:

    # Create a large box:
    box = [-60, -10, 40.0, 60.0, 0.0, 500.0, "2007", "2010"]
    
    # Init a parallel fetcher:
    fetcher = DataFetcher(parallel=True,
                          chunks_maxsize={'dpt': 100}).region(box)

    # Check number of chunks:
    len(fetcher.uri)

Since this creates a large number of chunks, let’s do this again and
combine with the option ``chunks`` to see easily what’s going on:

.. ipython:: python
    :okwarning:

    # Init a parallel fetcher with chunking along the vertical axis alone:
    fetcher = DataFetcher(parallel=True,
                          chunks_maxsize={'dpt': 100},
                          chunks={'lon':1, 'lat':1, 'dpt':'auto', 'time':1}).region(box)
    
    for uri in fetcher.uri:
        print("http: ... ", "&".join(uri.split("&")[1:-2])) # Display only the relevant URL part


You can see, that the ``pres`` argument of this erddap list of URLs
define layers not thicker than the requested 100db.

With the ``profile`` and ``float`` access points, you can use the
``wmo`` keyword to control the number of WMOs in each chunks.

.. ipython:: python
    :okwarning:

    WMO_list = [6902766, 6902772, 6902914, 6902746, 6902916, 6902915, 6902757, 6902771]
    
    # Init a parallel fetcher with chunking along the list of WMOs:
    fetcher = DataFetcher(parallel=True,
                          chunks_maxsize={'wmo': 3}).float(WMO_list)
    
    for uri in fetcher.uri:
        print("http: ... ", "&".join(uri.split("&")[1:-2])) # Display only the relevant URL part


You see here, that this request for 8 floats is split in chunks with no
more that 3 floats each.

.. warning::

    At this point, there is no mechanism to chunk requests along cycle numbers for the ``profile`` access point. See :issue:`362`.


Dask Cluster example
~~~~~~~~~~~~~~~~~~~~

The ``parallel`` option/argument can directly takes a `Dask Cluster <https://docs.dask.org/en/stable/deploying.html>`_ `client <https://distributed.dask.org/en/latest/client.html>`_ object.

This can go like this:

.. ipython:: python
    :okwarning:

    from dask.distributed import Client
    client = Client(processes=True)
    print(client)

    %%time
    with argopy.set_options(parallel=client):
        f = DataFetcher(src='argovis').region([-75, -70, 25, 40, 0, 1000, '2020-01-01', '2021-01-01'])
        print("%i chunks to process" % len(f.uri))
        print(f)
        ds = f.load().data
        print(ds)


Lazy dataset access
-------------------

.. warning:
    As of February 2025, this feature is considered experimental and could change without any deprecation warnings from
    one release to another. This is part of a wider effort to prepare **argopy** for evolutions of the Argo dataset in
    the cloud (cf the `ADMT working group on Argo cloud format activities <https://github.com/OneArgo/ADMT/issues/5>`_).

This **argopy** feature is implemented with `open_dataset` methods from argopy stores (file, http and s3) and the
class:`ArgoFloat` class. Since this is somehow a low-level implementation whereby users need to work with float data
directly, it is probably targetting users with operator or expert knowledge of Argo.

Contrary to the other performance improvement methods, this one is not accessible with a :class:`DataFetcher`.

**What is a lazy access to a dataset ?**

Lazyness in our use case, relates to limiting data transfert/load to what is really needed for an operation. For instance:

- if you want to work with a single Argo parameter for a given float, you don't need to download from the GDAC server all
 the other parameters,
- if you only are interested in assessing a file content (e.g. number of profiles or vertical levels), you also don't
 need to load anything else than the dimensions of the netcdf files.

Since a regular Argo netcdf is not intendeed to be accessed partially, it is rather tricky to access Argo data lazily.
Hopefully, the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library has been developped precisely for this case.

**Accessing Argo dataset lazilly**

First, not all file access protocols support a range request that is mandatory to access lazilly a netcdf file.
The table below synthesises lazy support for all possible GDAC hosts:

.. list-table:: GDAC hosts support for lazy access to float dataset
    :header-rows: 1
    :stub-columns: 1

    * -
      - Supported
    * - https://data-argo.ifremer.fr
      - ❌
    * - https://usgodae.org/pub/outgoing/argo
      - ❌
    * - ftp://ftp.ifremer.fr/ifremer/argo
      - ❌
    * - s3://argo-gdac-sandbox/pub
      - ✅
    * - a local GDAC copy
      - ✅


**argopy** kerchunk helper
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to access lazily an Argo netcdf files, localy or remotely with a server supporting range requests, the netcdf
content has to be curated in order to make a byte range *catalogue* of its content. To do so, you need to have the
`kerchunk <https://fsspec.github.io/kerchunk/>`_ library installed in your working environment.

We developped a specific class to make this netdf file curation easy for **argopy** users: :class:`stores.ArgoKerchunker`.

A typical use case will be to curate one or a collection of netcdf files and then save byte range *catalogues* (these are
json files with zarr data allowing to directly access part of the netcdf file content) in a specific store.

This can be done on the fly, but it can be time consuming with regard to the size of Argo netcdf datasets (the overhead
of using kerchunk is not worth the download time). On the other hand, it may be interesting to save kerchunk data in a
shared store (local or remote), so that other users will be able to use it. From the user perspective, this has the huge
advantage of not requiring the kerchunk library anymore, since opening lazily a dataset will be done with the zarr engine.

This is demonstrated below

.. ipython:: python
    :okwarning:

    from argopy.stores import ArgoKerchunker

    # Create an instance that will save netcdf byte range *catalogues* on a local folder:
    ak = ArgoKerchunker(store='local', root='~/myshared_kerchunk_data_folder')

    # Let's consider a remote Argo netcdf file from a server supporting lazy access:
    ncfile = "s3://argo-gdac-sandbox/pub/dac/coriolis/6903090/6903090_prof.nc"

    # Compute and save this netcdf byte range *catalogue* for later lazy access:
    js = ak.to_kerchunk(ncfile)


Now, for any user with read access to the `~/myshared_kerchunk_data_folder` folder, lazy access is possible without kerchunk

.. ipython:: python
    :okwarning:

    from argopy import s3store

    # Create an instance where to find netcdf byte range *catalogues*:
    ak = ArgoKerchunker(store='local', root='~/myshared_kerchunk_data_folder')

    # Simply open the netcdf file lazily, giving the appropriate ArgoKerchunker:
    s3store().open_dataset(ncfile, lazy=True, ak=ak)

Laziness with an :class:`ArgoFloat`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low level laziness
~~~~~~~~~~~~~~~~~~

Argo netcdf file kerchunk helper

This class is for expert users who wish to test lazy access to remote netcdf files. If you need to compute kerchunk zarr data on-demand, we don’t recommend to use this method as it shows poor performances on mono or multi profile files. It is more efficient to compute kerchunk zarr data in batch, and then to provide these data to users.

The kerchunk library is required only if you start from scratch and need to extract zarr data from a netcdf file, i.e. execute ArgoKerchunker.translate().