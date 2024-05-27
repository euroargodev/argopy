Performances
============

.. contents::
   :local:

To improve **argopy** data fetching performances (in terms of time of
retrieval), 2 solutions are available:

-  :ref:`cache` fetched data, i.e. save your request locally so that you don’t have to fetch it again,
-  Use :ref:`parallel`, i.e. fetch chunks of independent data simultaneously.

These solutions are explained below.

Note that another solution from standard big data strategies would be to
fetch data lazily. But since (i) **argopy** post-processes raw Argo data
on the client side and (ii) none of the data sources are cloud/lazy
compatible, this solution is not possible (yet).

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher

Cache
-----

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

Sometimes you may find that your request takes a long time to fetch, or
simply does not even succeed. This is probably because you’re trying to
fetch a large amount of data.

In this case, you can try to let argopy chunks your request into smaller
pieces and have them fetched in parallel for you. This is done with the
argument ``parallel`` of the data fetcher and can be tuned using options
``chunks`` and ``chunksize``.

This goes by default like this:

.. ipython:: python
    :okwarning:

    # Define a box to load (large enough to trigger chunking):
    box = [-60, -30, 40.0, 60.0, 0.0, 100.0, "2007-01-01", "2007-04-01"]
    
    # Instantiate a parallel fetcher:
    loader_par = DataFetcher(src='erddap', parallel=True).region(box)

you can also use the option ``progress`` to display a progress bar
during fetching:

.. ipython:: python
    :okwarning:

    loader_par = DataFetcher(src='erddap', parallel=True, progress=True).region(box)
    loader_par

Then, you can fetch data as usual:

.. ipython:: python
    :okwarning:

    %%time
    ds = loader_par.to_xarray()

Number of chunks
~~~~~~~~~~~~~~~~

To see how many chunks your request has been split into, you can look at
the ``uri`` property of the fetcher, it gives you the list of paths
toward data:

.. ipython:: python
    :okwarning:

    for uri in loader_par.uri:
        print("http: ... ", "&".join(uri.split("&")[1:-2]))  # Display only the relevant part of each URLs of URI:

To control chunking, you can use the **``chunks``** option that
specifies the number of chunks in each of the *direction*:

-  ``lon``, ``lat``, ``dpt`` and ``time`` for a **region** fetching,
-  ``wmo`` for a **float** and **profile** fetching.

.. ipython:: python
    :okwarning:

    # Create a large box:
    box = [-60, 0, 0.0, 60.0, 0.0, 500.0, "2007", "2010"]
    
    # Init a parallel fetcher:
    loader_par = DataFetcher(src='erddap', 
                                 parallel=True, 
                                 chunks={'lon': 5}).region(box)
    # Check number of chunks:
    len(loader_par.uri)

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
    loader_par = DataFetcher(src='erddap', 
                                 parallel=True, 
                                 chunks={'lon': 5, 'lat':1, 'dpt':1, 'time':1}).region(box)
    
    # Check number of chunks:
    len(loader_par.uri)

We now have 5 chunks along longitude, check out the URLs parameter in
the list of URIs:

.. ipython:: python
    :okwarning:

    for uri in loader_par.uri:
        print("&".join(uri.split("&")[1:-2])) # Display only the relevant URL part

.. note::

    You may notice that if you run the last command with the `argovis` fetcher, you will still have more than 5 chunks (i.e. 65). This is because `argovis` is limited to 3 months length requests. So, for this request that is 3 years long, argopy ends up with 13 chunks along time, times 5 chunks in longitude, leading to 65 chunks in total.

.. warning::
    The `gdac` fetcher and the `float` and `profile` access points of the `argovis` fetcher use a list of resources than are not chunked but fetched in parallel using a batch queue.

Size of chunks
~~~~~~~~~~~~~~

The default chunk size for each access point dimensions are:

====================== ==================
Access point dimension Maximum chunk size
====================== ==================
region / **lon**       20 deg
region / **lat**       20 deg
region / **dpt**       500 m or db
region / **time**      90 days
float / **wmo**        5
profile / **wmo**      5
====================== ==================

These default values are used to chunk data when the ``chunks``
parameter key is set to ``auto``.

But you can modify the maximum chunk size allowed in each of the
possible directions. This is done with the option
**``chunks_maxsize``**.

For instance if you want to make sure that your chunks are not larger
then 100 meters (db) in depth (pressure), you can use:

.. ipython:: python
    :okwarning:

    # Create a large box:
    box = [-60, -10, 40.0, 60.0, 0.0, 500.0, "2007", "2010"]
    
    # Init a parallel fetcher:
    loader_par = DataFetcher(src='erddap', 
                                 parallel=True, 
                                 chunks_maxsize={'dpt': 100}).region(box)
    # Check number of chunks:
    len(loader_par.uri)

Since this creates a large number of chunks, let’s do this again and
combine with the option ``chunks`` to see easily what’s going on:

.. ipython:: python
    :okwarning:

    # Init a parallel fetcher with chunking along the vertical axis alone:
    loader_par = DataFetcher(src='erddap', 
                                 parallel=True, 
                                 chunks_maxsize={'dpt': 100},
                                 chunks={'lon':1, 'lat':1, 'dpt':'auto', 'time':1}).region(box)
    
    for uri in loader_par.uri:
        print("http: ... ", "&".join(uri.split("&")[1:-2])) # Display only the relevant URL part


You can see, that the ``pres`` argument of this erddap list of URLs
define layers not thicker than the requested 100db.

With the ``profile`` and ``float`` access points, you can use the
``wmo`` keyword to control the number of WMOs in each chunks.

.. ipython:: python
    :okwarning:

    WMO_list = [6902766, 6902772, 6902914, 6902746, 6902916, 6902915, 6902757, 6902771]
    
    # Init a parallel fetcher with chunking along the list of WMOs:
    loader_par = DataFetcher(src='erddap', 
                                 parallel=True, 
                                 chunks_maxsize={'wmo': 3}).float(WMO_list)
    
    for uri in loader_par.uri:
        print("http: ... ", "&".join(uri.split("&")[1:-2])) # Display only the relevant URL part


You see here, that this request for 8 floats is split in chunks with no
more that 3 floats each.

.. warning::

    At this point, there is no mechanism to chunk requests along cycle numbers for the ``profile`` access point.

Parallelization methods
~~~~~~~~~~~~~~~~~~~~~~~

They are 2 methods available to set-up your data fetching requests in
parallel:

1. `Multi-threading <https://en.wikipedia.org/wiki/Multithreading_(computer_architecture)>`__
   for all data sources,
2. `Multi-processing <https://en.wikipedia.org/wiki/Multiprocessing>`__
   for *gdac* with a local host.

Both options use a pool of
`threads <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`__
or
`processes <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`__
managed with the `concurrent futures
module <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`__.

The parallelization method is set with the ``parallel_method`` option of
the fetcher, which can take as values ``thread`` or ``process``.

Methods available for data sources:

=================== ====== ==== =======
**Parallel method** erddap gdac argovis
=================== ====== ==== =======
Multi-threading     X      X    X
Multi-processes            X
=================== ====== ==== =======

Note that you can in fact pass the method directly with the ``parallel``
option, so that in practice, the following two formulations are
equivalent:

.. ipython:: python
    :okwarning:

   DataFetcher(parallel=True, parallel_method='thread')
   DataFetcher(parallel='thread')

Comparison of performances
~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that to compare performances with or without the parallel option,
we need to make sure that data are not cached on the server side. To do
this, we use a very small random perturbation on the box definition,
here on the maximum latitude. This ensures that nearly the same amount of data
will be requested but not cached by the server.

.. ipython:: python
    :okwarning:

    def this_box():
        return [-60, 0, 
               20.0, 60.0 + np.random.randint(0,100,1)[0]/1000, 
               0.0, 500.0, 
               "2007", "2009"]

.. ipython:: python
    :okwarning:

    %%time
    b1 = this_box()
    f1 = DataFetcher(src='argovis', parallel=False).region(b1)
    ds1 = f1.to_xarray()

.. ipython:: python
    :okwarning:

    %%time
    b2 = this_box()
    f2 = DataFetcher(src='argovis', parallel=True).region(b2)
    ds2 = f2.to_xarray()

**This simple comparison hopefully shows that parallel request is significantly
faster than the standard one.**

Warnings
~~~~~~~~

-  Parallelizing your fetcher is useful to handle large region of data,
   but it can also add a significant overhead on *reasonable* size
   requests that may lead to degraded performances. So, we do not
   recommend for you to use the parallel option systematically.

-  You may have different dataset sizes with and without the
   ``parallel`` option. This may happen if one of the chunk data
   fetching fails. By default, data fetching of multiple resources fails
   with a warning. You can change this behaviour with the option
   ``errors`` of the ``to_xarray()`` fetcher methods, just set it to
   ``raise`` like this:

   .. code:: python

      DataFetcher(parallel=True).region(this_box()).to_xarray(errors='raise');

You can also use ``silent`` to simply hide all messages during fetching.
