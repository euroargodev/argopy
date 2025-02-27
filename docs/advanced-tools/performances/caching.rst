.. currentmodule:: argopy

.. _cache:

Caching
=======

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher


Caching data
------------

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
------------------

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
