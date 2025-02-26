.. currentmodule:: argopy
.. _metadata_fetching:

Argo meta-data
==============

.. contents::
   :local:

Reference tables and vocabulary server
--------------------------------------
.. currentmodule:: argopy.related

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


Deployment Plan
---------------
.. currentmodule:: argopy.related

It may be useful to be able to retrieve meta-data from Argo deployments. **argopy** can use the `OceanOPS API for metadata access <https://www.ocean-ops.org/api/swagger/?url=https://www.ocean-ops.org/api/1/oceanops-api.yaml>`_ to retrieve these information. The returned deployment `plan` is a list of all Argo floats ever deployed, together with their deployment location, date, WMO, program, country, float model and current status.

To fetch the Argo deployment plan, **argopy** provides a dedicated utility class: :class:`OceanOPSDeployments` that can be used like this:

.. ipython:: python
    :okwarning:

    from argopy import OceanOPSDeployments

    deployment = OceanOPSDeployments()

    df = deployment.to_dataframe()
    df

:class:`OceanOPSDeployments` can also take an index box definition as argument in order to restrict the deployment plan selection to a specific region or period:

.. code-block:: python

    deployment = OceanOPSDeployments([-90, 0, 0, 90])
    # deployment = OceanOPSDeployments([-20, 0, 42, 51, '2020-01', '2021-01'])
    # deployment = OceanOPSDeployments([-180, 180, -90, 90, '2020-01', None])

Note that if the starting date is not provided, it will be set automatically to the current date.

Last, :class:`OceanOPSDeployments` comes with a plotting method:

.. code-block:: python

    fig, ax = deployment.plot_status()

.. image:: _static/scatter_map_deployment_status.png


.. note:: The list of possible deployment status name/code is given by:

    .. code-block:: python

        OceanOPSDeployments().status_code

    =========== == ====
    Status      Id Description
    =========== == ====
    PROBABLE    0  Starting status for some platforms, when there is only a few metadata available, like rough deployment location and date. The platform may be deployed
    CONFIRMED   1  Automatically set when a ship is attached to the deployment information. The platform is ready to be deployed, deployment is planned
    REGISTERED  2  Starting status for most of the networks, when deployment planning is not done. The deployment is certain, and a notification has been sent via the OceanOPS system
    OPERATIONAL 6  Automatically set when the platform is emitting a pulse and observations are distributed within a certain time interval
    INACTIVE    4  The platform is not emitting a pulse since a certain time
    CLOSED      5  The platform is not emitting a pulse since a long time, it is considered as dead
    =========== == ====

ADMT Documentation
------------------
.. currentmodule:: argopy.related

More than 20 pdf manuals have been produced by the Argo Data Management Team. Using the :class:`ArgoDocs` class, it's easy to navigate this great database.

If you don't know where to start, you can simply list all available documents:

.. ipython:: python
    :okwarning:

    from argopy import ArgoDocs

    ArgoDocs().list

Or search for a word in the title and/or abstract:

.. ipython:: python
    :okwarning:

    results = ArgoDocs().search("oxygen")
    for docid in results:
        print("\n", ArgoDocs(docid))

Then using the Argo doi number of a document, you can easily retrieve it:

.. ipython:: python
    :okwarning:

    ArgoDocs(35385)

and open it in your browser:

.. ipython:: python
    :okwarning:

    # ArgoDocs(35385).show()
    # ArgoDocs(35385).open_pdf(page=12)

GDAC snapshot and DOI
---------------------

[TBC]

