.. currentmodule:: argopy

Reference dataset
=================

Since Argo floats are not recovered systematically, the quality control procedure requires historical measurements to
assess float data.

Historical CTD measurements have been curated and formatted by the ADMT for years, and are usually distributed as a
collection of Matlab binary files. This approach has the disadvantage of requiring QC operators to download the entire
reference dataset to start QC procedures, hence with the burden of having to keep the local version up to date with
the last available (and most accurate and exhaustive content) version.

The **argopy** team has work with Coriolis to try to alleviate these limitations by uploading both reference dataset to the
Ifremer erddap server. Using the erddap server, it is possible to download only the required data for a given float and the
last version of the dataset.

Ship-based CTD reference measurements
-------------------------------------

The ship-based CTD reference database is updated on a yearly bases by the Coriolis team or when there are enough new
data to justify an upgrade. The ship-based CTD measurements are provided by the PIs, the ARC, Clivar Hydrographic Center
and NODC/USA.

To access ship-based CTD reference measurements, we provide the :class:`CTDRefDataFetcher` class.

You can use the ``box`` argument to specify a rectangular space/domain to fetch reference data for. Note that this is
a list with minimum/maximum for longitude, then latitude, then pressure and finally time. The constraint on time is not
mandatory.

.. ipython:: python
    :okwarning:

    from argopy import CTDRefDataFetcher

    fetcher = CTDRefDataFetcher(box=[-65, -55, 10,  20, 0, 5000, '20000101', '20050101'])
    ds_ship_ctd_ref = fetcher.to_xarray()
    ds_ship_ctd_ref

Handling user name and password
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the ship-based CTD database may contain CTD data that are not yet freely available, access is restricted to
those performing DMQC on Argo data and a user authentication is required. This can be managed by **argopy**.

To provide user name and a password to access this protected dataset, you must set the **argopy** options ``user``
and ``password``.

These options are set automatically from environment variables ``ERDDAP_USERNAME`` and ``ERDDAP_PASSWORD`` but
they can also be set for the session or within a context like this:

.. code-block:: python

    import argopy
    argopy.set_options(user='jane_doe', password='****')

or

.. code-block:: python

    import argopy
    from argopy import CTDRefDataFetcher

    with argopy.set_options(user='jane_doe', password='****')
        ds_ship_ctd_ref = CTDRefDataFetcher(box=[-65, -55, 10,  20, 0, 5000]).to_xarray()


.. warning::

    If you already have accessed to this dataset from the Ifremer ftp, you can use the same username and password.
    But if you have no idea of what to use, please email codac@ifremer.fr to request access.


Argo-based CTD reference measurements
-------------------------------------

In areas with poor ship-based CTD data, good Argo profiles can be used for delayed mode QC. This Argo-based CTD reference
database is updated by SCRIPPS (J Gilson).

To access this dataset, **argopy** provides the ``ref`` dataset shortname to be used with a standard :class:`DataFetcher`
parameterized to point toward the ``erddap`` data source:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher

    fetcher = DataFetcher(ds='ref', src='erddap').region([-65, -55, 10,  20, 0, 5000])
    ds_argo_ctd_ref = fetcher.to_xarray()
    ds_argo_ctd_ref
