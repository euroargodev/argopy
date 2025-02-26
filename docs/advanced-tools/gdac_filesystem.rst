.. currentmodule:: argopy

.. _tools-gdacfs:

GDAC file system
================

**argopy** provides a convenient generic file system for all possible GDAC hosts: local, http, ftp and s3. The :class:`stores.gdacfs`
will help you to separate the logic in your code between file access and specific host protocol handling.

The class will return one of the **argopy** file systems (:class:`stores.filestore`, :class:`stores.httpstore`, :class:`stores.ftpstore` or :class:`stores.s3store`) with a prefixed directory, so that you don't have to include the GDAC path to access resources.

Creation
--------
Just provide a valid GDAC path, local or remote:

.. ipython:: python
    :okwarning:

    from argopy.stores import gdacfs

    fs = gdacfs("https://data-argo.ifremer.fr")

    # or:
    # fs = gdacfs("https://usgodae.org/pub/outgoing/argo")
    # fs = gdacfs("ftp://ftp.ifremer.fr/ifremer/argo")
    # fs = gdacfs("s3://argo-gdac-sandbox/pub")
    # fs = gdacfs("/home/ref-argo/gdac")

    fs

Usage
-----

The file system instance will provide most of the required methods to work with any file on the GDAC, without the burden of handling paths and protocols.

.. ipython:: python
    :okwarning:

    fs.glob("dac/aoml/13857/*_meta.nc")

    fs.info("dac/aoml/13857/13857_meta.nc")

    ds = fs.open_dataset("dac/coriolis/6903091/profiles/R6903091_001.nc")

    with fs.open("ar_index_this_week_meta.txt", "r") as f:
        data = f.readlines()
