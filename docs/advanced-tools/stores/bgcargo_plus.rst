.. _bgcargo_plus_store:

BGC-Argo+ dataset
=================

.. admonition:: What is BGC-Argo+?

   The `BGC-Argo+ <https://www.bgc-argo-plus.info>`_ dataset is a quality-controlled,
   outlier-removed compilation of BGC-Argo float data curated by the `HI-Cycles
   <https://hi-cycles.soest.hawaii.edu>`_ group at the School of Ocean and Earth Science
   and Technology (SOEST), University of Hawaiʻi at Mānoa.  Individual float files are
   served on the SOEST FTP server::

       ftp://ftp.soest.hawaii.edu/bgc_argo_plus/Individual_Floats/outliers_removed/

   Relative to the standard GDAC ``Sprof`` files, BGC-Argo+ files provide:

   - **Outlier removal** on all BGC variables (oxygen, nitrate, pH, chlorophyll,
     backscatter, irradiance).
   - **Consistent variable naming** across all floats.
   - **Version-tagged releases** so that scientific analyses can be reproduced
     against a fixed snapshot.

Usage
-----

The canonical access path is through :class:`argopy.ArgoFloat` by passing the
special dataset name ``"BGCArgoPlus"`` to :meth:`~argopy.ArgoFloat.open_dataset`:

.. code-block:: python

   from argopy import ArgoFloat

   af = ArgoFloat(6903091)
   ds = af.open_dataset('BGCArgoPlus')
   ds

This is entirely analogous to loading a standard GDAC file:

.. code-block:: python

   ds_sprof = af.open_dataset('Sprof')   # ← standard GDAC Sprof
   ds_bgcp  = af.open_dataset('BGCArgoPlus')  # ← BGC-Argo+ version

Pinning the version
-------------------

The BGC-Argo+ dataset is versioned.  By default, argopy uses
:data:`~argopy.stores.float.bgcargo_plus.BGCARGO_PLUS_DEFAULT_VERSION`
(currently ``"v0.1_2025_12"``).  You can pin to any available version with the
``bgcplus_version`` keyword:

.. code-block:: python

   ds = ArgoFloat(6903091).open_dataset('BGCArgoPlus', bgcplus_version='v0.1_2025_12')

.. tip::

   Pin the version in any scientific workflow that requires reproducibility across
   time.  Future releases of argopy will update the default version when a new
   BGC-Argo+ snapshot is published.

Low-level access
----------------

For fine-grained control you can use :class:`~argopy.stores.float.bgcargo_plus.BGCArgoPlusStore`
directly:

.. code-block:: python

   from argopy.stores.float.bgcargo_plus import BGCArgoPlusStore

   store = BGCArgoPlusStore(6903091)
   print(store.url)   # shows the full FTP URL
   ds = store.open_dataset()

The store wraps :class:`argopy.stores.ftpstore` and therefore supports the same
``cache``, ``cachedir``, ``lazy``, and ``timeout`` options:

.. code-block:: python

   store = BGCArgoPlusStore(6903091, cache=True, cachedir='/tmp/argopy_cache')
   ds = store.open_dataset(lazy=True)  # kerchunk-based lazy access

API reference
-------------

.. autofunction:: argopy.stores.float.bgcargo_plus.bgcargo_plus_url

.. autoclass:: argopy.stores.float.bgcargo_plus.BGCArgoPlusStore
   :members: url, open_dataset
   :undoc-members:
   :show-inheritance:

.. rubric:: Module-level constants

.. autodata:: argopy.stores.float.bgcargo_plus.BGCARGO_PLUS_FTP_HOST
.. autodata:: argopy.stores.float.bgcargo_plus.BGCARGO_PLUS_DEFAULT_VERSION
.. autodata:: argopy.stores.float.bgcargo_plus.BGCARGO_PLUS_PATH_TEMPLATE

See also
--------

- :ref:`argofloat_store` — the parent :class:`~argopy.ArgoFloat` documentation.
- `BGC-Argo+ website <https://www.bgc-argo-plus.info>`_
- `SOEST FTP index <https://ftp.soest.hawaii.edu/bgc_argo_plus/>`_
