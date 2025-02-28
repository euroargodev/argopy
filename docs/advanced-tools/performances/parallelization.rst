.. currentmodule:: argopy

.. _parallel:

Parallel data fetching
======================

Sometimes you may find that your request takes a long time to fetch, or simply does not even succeed. This is probably
because you’re trying to fetch a large amount of data.

In this case, you can try to let **argopy** chunks your request into smaller pieces and have them fetched in parallel
for you. This is done with the data fetcher argument, or global option, ``parallel``.

Parallelization can be tuned using arguments ``chunks`` and ``chunks_maxsize``.

This goes by default like this:

.. ipython:: python
    :okwarning:

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
-----------------------

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

    import argopy
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
----------------------------

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
----------------------------------

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


Working with a Dask Cluster
---------------------------

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
        print("\n", f)

        ds = f.load().data
        print("\n", ds)
