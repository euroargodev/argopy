.. currentmodule:: argopy.related

Reference tables and vocabulary server
--------------------------------------

The Argo netcdf format is strict and based on a collection of variables fully documented and conventioned. All reference tables can be found in the `Argo user manual <https://doi.org/10.13155/29825>`_.

However, a machine-to-machine access to these tables is often required. This is possible thanks to the work of the **Argo Vocabulary Task Team (AVTT)** that is a team of people responsible for the `NVS <https://github.com/nvs-vocabs>`_ collections under the Argo Data Management Team governance.

.. note::

    The GitHub organization hosting the AVTT is the 'NERC Vocabulary Server (NVS)', aka 'nvs-vocabs'. This holds a list of NVS collection-specific GitHub repositories. Each Argo GitHub repository is called after its corresponding collection ID (e.g. R01, RR2, R03 etc.). `The current list is given here <https://github.com/nvs-vocabs?q=argo&type=&language=&sort=name>`_.

    The management of issues related to vocabularies managed by the Argo Data Management Team is done on this `repository <https://github.com/nvs-vocabs/ArgoVocabs>`_.

**argopy** provides the utility class :class:`ArgoNVSReferenceTables` to easily fetch and get access to all Argo reference tables. If you already know the name of the reference table you want to retrieve, you can simply get it like this:

.. ipython:: python
    :okwarning:

    from argopy import ArgoNVSReferenceTables
    NVS = ArgoNVSReferenceTables()
    NVS.tbl('R01')

The reference table is returned as a :class:`pandas.DataFrame`. If you want the exact name of this table:

.. ipython:: python

    NVS.tbl_name('R01')

**If you don't know the reference table ID**, you can search for a word in tables title and/or description with the ``search`` method:

.. ipython:: python

    id_list = NVS.search('sensor')

This will return the list of reference table ids matching your search. It can then be used to retrieve table information:

.. ipython:: python

    [NVS.tbl_name(id) for id in id_list]


The full list of all available tables is given by the :meth:`ArgoNVSReferenceTables.all_tbl_name` property. It will return a dictionary with table IDs as key and table name, definition and NVS link as values. Use the :meth:`ArgoNVSReferenceTables.all_tbl` property to retrieve all tables.

.. ipython:: python

    NVS.all_tbl_name
