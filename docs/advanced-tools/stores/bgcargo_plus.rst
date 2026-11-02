.. currentmodule:: argopy
.. _bgcargo_plus_store:

BGC-Argo+ product
=================

.. warning::

   BGC-Argo+ is a **third-party, non-official** Argo-based product. It is not managed nor endorsed by the Argo Data
   Management Team (ADMT). It is accessible with :meth:`ArgoFloat.open_product`, following the **Argopy**
   :ref:`policy regarding third-party products <product_distribution>`.

.. admonition:: What is BGC-Argo+?

   The `BGC-Argo+ <https://www.bgc-argo-plus.info>`_ dataset is a quality-controlled,
   outlier-removed compilation of BGC-Argo float data curated by the `HI-Cycles
   <https://hi-cycles.org>`_ group at the School of Ocean and Earth Science
   and Technology (SOEST), University of Hawaiʻi at Mānoa (Bushinsky et al, 2026, submitted). Individual float files are
   served on the SOEST FTP server::

       ftp://ftp.soest.hawaii.edu/bgc_argo_plus/outliers_removed/<version>/

   Relative to the standard GDAC ``Sprof`` files, BGC-Argo+ files provide:

   - **Outlier removal** on most BGC variables (oxygen, nitrate, pH).
   - **Consistent variable naming** across all floats.
   - **Version-tagged releases** so that scientific analyses can be reproduced
     against a fixed snapshot.

   References:

   - Dataset: `doi:10.5281/zenodo.19353191 <https://doi.org/10.5281/zenodo.19353191>`_
   - Bushinsky et al. (2026, submitted to ESSD): `doi:10.5194/essd-2026-311 <https://doi.org/10.5194/essd-2026-311>`_

   Contact: Raphaël Bajon (rbajon@hawaii.edu).

Load BGC-Argo+ data
-------------------

Load the BGC-Argo+ file of a float with the :meth:`ArgoFloat.open_product` method:

.. code-block:: python

   from argopy import ArgoFloat

   af = ArgoFloat(6903091)
   ds = af.open_product('BGCArgoPlus')
   ds

Official Argo data is accessible with :meth:`ArgoFloat.open_dataset`, so you can easily compare both:

.. code-block:: python

   ds_sprof = af.open_dataset('Sprof')        # ← official GDAC Sprof
   ds_bgcp  = af.open_product('BGCArgoPlus')  # ← third-party BGC-Argo+ version

Choosing the version
--------------------

The BGC-Argo+ dataset is versioned. Use the ``version`` keyword to choose which version to load:

.. code-block:: python

   af = ArgoFloat(6903091)

   ds = af.open_product('BGCArgoPlus')                          # Default: version from the reference paper
   ds = af.open_product('BGCArgoPlus', version='paper')         # Same as above, v0.1_2026_04
   ds = af.open_product('BGCArgoPlus', version='latest')        # Most recent version on the SOEST server
   ds = af.open_product('BGCArgoPlus', version='v1.0_2026_08')  # A specific version

- ``'paper'`` (the default) is the version described in the reference paper,
  Bushinsky et al. (2026, `doi:10.5194/essd-2026-311 <https://doi.org/10.5194/essd-2026-311>`_):
  :data:`~argopy.stores.float.products.bgcargo_plus.BGCARGO_PLUS_PAPER_VERSION` = ``"v0.1_2026_04"``,
  archived at `doi:10.5281/zenodo.19709012 <https://doi.org/10.5281/zenodo.19709012>`_.
- ``'latest'`` looks up the most recent version available on the SOEST FTP server. This requires a connection
  to the server, and the result will change when a new version is published.

You can list the versions available on the server with:

.. code-block:: python

   from argopy.stores.float.products.bgcargo_plus import bgcargo_plus_versions, bgcargo_plus_latest_version

   bgcargo_plus_versions()        # ['v0.0_2025_09', 'v0.1_2025_12', 'v0.1_2026_04', 'v1.0_2026_08']
   bgcargo_plus_latest_version()  # 'v1.0_2026_08'

Server availability
-------------------

BGC-Argo+ files are served by a single FTP server. If it cannot be reached (server down, network issue), a
:class:`~argopy.stores.float.products.bgcargo_plus.BGCArgoPlusServerError` is raised. You can check the
server beforehand with:

.. code-block:: python

   from argopy.stores.float.products.bgcargo_plus import bgcargo_plus_server_available

   bgcargo_plus_server_available()  # True or False

See also
--------

- :ref:`BGC-Argo+ API reference <api-bgcargo-plus>`.
- :ref:`tools-argofloat`: the parent :class:`~argopy.ArgoFloat` documentation.
- `BGC-Argo+ website <https://www.bgc-argo-plus.info>`_
- `SOEST FTP index <https://ftp.soest.hawaii.edu/bgc_argo_plus/>`_
