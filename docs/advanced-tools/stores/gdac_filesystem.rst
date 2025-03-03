.. currentmodule:: argopy

.. _tools-gdacfs:

Store for GDAC files
====================

**argopy** provides a convenient store for all GDAC files, a generic file system, for all possible GDAC hosts. The :class:`gdacfs` class will help you to separate the logic in your code between data access and specific host protocol handling.

This class will return one of the **argopy** file systems (:class:`stores.filestore`, :class:`stores.httpstore`, :class:`stores.ftpstore` or :class:`stores.s3store`) with a prefixed directory, so that you don't have to include the GDAC root path to access files.

This generic file system is based on the fact that the Argo dataset is organised similarly whatever the GDAC host (local copy, http, ftp and s3). For instance, all GDAC hosts have a `dac` folder, or netcdf files path have similar patterns like `dac/<DAC>/<WMO>/<WMO>_prof.nc`. Therefore, you can separate `file access logic` from the `host protocol handling` in your workflow, making it GDAC agnostic and more modular.

Creation
--------
Just provide a valid GDAC path, local or remote:

.. ipython:: python
    :okwarning:

    from argopy import gdacfs

    fs = gdacfs("https://data-argo.ifremer.fr")
    # or:
    # fs = gdacfs("https://usgodae.org/pub/outgoing/argo")
    # fs = gdacfs("ftp://ftp.ifremer.fr/ifremer/argo")
    # fs = gdacfs("s3://argo-gdac-sandbox/pub")
    # fs = gdacfs("/home/ref-argo/gdac")

    fs

You can also use shortcut for all remote GDACs:

.. ipython:: python
    :okwarning:

    fs = gdacfs("http")    # or "https"    > https://data-argo.ifremer.fr
    # or:
    # fs = gdacfs("us-http") # or "us-https" > https://usgodae.org/pub/outgoing/argo
    # fs = gdacfs("ftp")     #            > ftp://ftp.ifremer.fr/ifremer/argo
    # fs = gdacfs("s3")      # or "aws"   > s3://argo-gdac-sandbox/pub



Usage
-----

The file system instance will provide most of the required methods to work with any file on the GDAC, without the burden of handling access protocols and paths construction: paths must be made relative to the GDAC root folder (which is natively the case in Argo index files).

Here are a few examples of what can be down:

.. ipython:: python
    :okwarning:

    fs.glob("dac/aoml/13857/*_meta.nc")

    fs.info("dac/aoml/13857/13857_meta.nc")

    ds = fs.open_dataset("dac/coriolis/6903091/profiles/R6903091_001.nc")

    with fs.open("ar_index_this_week_meta.txt", "r") as f:
        data = f.readlines()
