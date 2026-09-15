.. currentmodule:: argopy


Argo vocabulary and Reference
=============================

The Argo GDAC database and netcdf format are strict and based on a collection of parameters fully documented and conventioned. Most of these parameters are allowed to take only referenced values organised in tables and related with miscellaneous mappings. All reference tables can be found in the `Argo user manual <https://doi.org/10.13155/29825>`_.

Argo references (tables, values, mappings) are machine-to-machine accessible from a `NVS server <https://vocab.nerc.ac.uk>`_ where the **Argo Vocabulary Task Team (AVTT)** maintains up-to-date the entire Argo vocabulary.

**Argopy** provides a facilitated access to the Argo vocabulary, specifically:

- :ref:`argoreferencevalue`, aka NVS "concept", eg: value "AANDERAA_OPTODE_3930" is one documented and possible value for the "SENSOR_MODEL" parameter in netcdf files,
- :ref:`argoreferencetable`, aka NVS "vocabulary", eg: table "R27" documents the list of possible values for the "SENSOR_MODEL" parameter in netcdf files, like "AANDERAA_OPTODE_3930",
- :ref:`argoreferencemapping`, aka a NVS "mapping", eg: values from the "R27/SENSOR_MODEL" table like "AANDERAA_OPTODE" are *related* to values from the "R25/SENSOR" table like "OPTODE_DOXY" and are *narrower* concept of values from the "R26/SENSOR_MAKER" table like "AANDERAA".

**Argopy** provides specific utility classes for each of these objects that are detailed below.

.. note::

    The AVTT work is managed on Github where a list of NVS collection-specific repositories is hosted under the `nvs-vocabs organisation <https://github.com/nvs-vocabs>`_.

    The management of issues related to vocabularies is done on this `repository <https://github.com/nvs-vocabs/ArgoVocabs>`_.


Let's get started by importing the **Argopy** APIs documented hereafter:

.. ipython:: python
    :okwarning:

    from argopy import ArgoReferenceValue, ArgoReferenceTable, ArgoReferenceMapping

.. _argoreferencevalue:

Reference **values**
--------------------

**A class to work with an Argo Reference Value, a.k.a., a NVS "concept".**

An Argo Reference Value represents one possible and documented value for an Argo parameter. For example, ``AANDERAA_OPTODE_3835`` is an Argo Reference Value for the ``SENSOR_MODEL`` parameter. All possible values for this parameter are listed in the **Argo reference table 27** for "Argo sensor models" (:ref:`see below <argoreferencetable>`).

The :class:`ArgoReferenceValue` class holds all the Argo referencing system information:

- the comprehensive logic (e.g. the reference table this value belongs to is automatically determined when possible, hints are return otherwise),
- the value meta-data are in read-only attributes based on NVS data, e.g. ``definition`` and ``version``, ``deprecated``, etc...,
- the value meta-data can be exported, e.g. ``to_dict()`` and ``to_json()``.


.. note::

    NVS (NERC Vocabulary Server) "concept" are formally `SKOS (Simple Knowledge Organization System) "concept" <https://en.wikipedia.org/wiki/Simple_Knowledge_Organization_System#Concepts>`_.


Creation
^^^^^^^^

.. code-block:: python
    :caption: Creation

    from argopy import ArgoReferenceValue

    # One possible value for the Argo parameter 'SENSOR_MODEL':
    arv = ArgoReferenceValue('AANDERAA_OPTODE_3835')

    # For ambiguous value seen in more than one Reference Table
    arv = ArgoReferenceValue('4', reference='RT_QC_FLAG')
    arv = ArgoReferenceValue('4', reference='RR2')

    # From NVS/URN jargon:
    arv = ArgoReferenceValue.from_urn('SDN:R27::AANDERAA_OPTODE_3835')

.. ipython:: python
    :okwarning:

    ArgoReferenceValue('AANDERAA_OPTODE_3835')

Read attributes
^^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Read attributes

    from argopy import ArgoReferenceValue
    arv = ArgoReferenceValue('AANDERAA_OPTODE_3835')

    # All possible attributes are listed in:
    arv.attrs

Reference Value attributes (and their NVS origin):

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Attribute
      - Description / NVS origin
    * - ``arv.name``
      - ``nvs["skos:altLabel"]`` or ``urnparser(id2urn(nvs["@id"]))["termid"]`` if ``altLabel`` is ``None``
    * - ``arv.long_name``
      - ``nvs["skos:prefLabel"]["@value"]``
    * - ``arv.definition``
      - ``nvs["skos:definition"]["@value"]``
    * - ``arv.deprecated``
      - ``nvs["owl:deprecated"]``
    * - ``arv.reference``
      - The reference table this concept belongs to, can be used with :class:`ArgoReferenceTable` (e.g. ``'R27'``)
    * - ``arv.parameter``
      - The netCDF parameter this concept applies to, can be used with :class:`ArgoReferenceTable` (e.g. ``'SENSOR_MODEL'``)

Other, more technical, Reference Value attributes:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Attribute
      - NVS origin
    * - ``arv.version``
      - ``nvs["owl:versionInfo"]``
    * - ``arv.date``
      - ``nvs["dc:date"]``
    * - ``arv.uri``
      - ``nvs["@id"]``
    * - ``arv.urn``
      - ``nvs["skos:notation"]``

Relationships with other Reference Values or Context:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Attribute
      - NVS origin
    * - ``arv.broader``
      - ``nvs["skos:broader"]``
    * - ``arv.narrower``
      - ``nvs["skos:narrower"]``
    * - ``arv.related``
      - ``nvs["skos:related"]``
    * - ``arv.sameas``
      - ``nvs["owl:sameAs"]``
    * - ``arv.context``
      - ``nvs["@context"]``

Additional attributes:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Attribute
      - Description
    * - ``arv.extra``
      - Extra attributes for R03, R14 and R18 values, curated from the value definition string (see examples below)
    * - ``arv.nvs``
      - Raw NVS json data

.. note::

    In IPython environment, like notebooks, **Argopy** provides an auto-completion feature to easily get one of the attributes, just press tab when typing ``arv['``:

    .. image:: ../../_static/ArgoReferenceValue_autocompletion.png
        :width: 289

Extra attributes (R03, R14, R18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometime, the reference value definition is formatted to hold extra-attributes.

For instance, the BBP470 definition reads: ``Particle backscattering at 470 nm wavelength, reported by ECO3 sensor. Local_Attributes:{long_name:Particle backscattering at 470 nanometers; standard_name:-; units:m-1; valid_min:-; valid_max:-; fill_value:99999.f}. Properties:{category:b; data_type:float}``.

**Argopy** automatically infers those extra-attributes like ``Local_Attributes`` and ``Properties`` from this example, and make them available as regular attributes to the :class:`ArgoReferenceValue` instance.

.. code-block:: python
    :caption: Extra attributes (R03, R14, R18)

    from argopy import ArgoReferenceValue

    # For Values from R03 table
    arv = ArgoReferenceValue('BBP470')
    arv.extra
    arv.extra['Local_Attributes'].long_name
    arv.extra['Properties'].category

    # For Values from R14 table
    arv = ArgoReferenceValue('T000015')
    arv.extra
    arv.extra['Template_Values'].unit

    # For Values from R18 table
    arv = ArgoReferenceValue('CB00001')
    arv.extra
    arv.extra['Template_Values'].short_sensor_name

Export methods
^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Export methods

    from argopy import ArgoReferenceValue
    arv = ArgoReferenceValue('AANDERAA_OPTODE_3835')

    # Export to a dictionary:
    arv.to_dict()
    arv.to_dict(keys=['name', 'deprecated'])  # Select attributes to export in dictionary keys

    # Export to json structure:
    arv.to_json()  # In memory
    arv.to_json('reference_value.json')  # To a json file
    arv.to_json('reference_value.json', keys=['name', 'deprecated'])  # Select attributes to export

.. _argoreferencetable:

Reference **tables**
--------------------

**A class to work with an Argo reference table, a.k.a., a NVS "vocabulary".**

For example, the vocabulary for "Argo sensor models" corresponds to the Argo reference table 27 ("R27") and is used to document possible values of the ``SENSOR_MODEL`` parameter in NetCDF files. All possible values for this parameter are instances of :class:`ArgoReferenceValue` (:ref:`see above <argoreferencevalue>`).

The :class:`ArgoReferenceTable` class holds all the Argo referencing system information:

- the comprehensive logic (use ``SENSOR`` in place of ``R25``),
- the table meta-data are in read-only attributes, e.g. ``art.description``, ``art.version``, ``art.date``, etc...,
- the list of table values/rows are accessible through label-based indexing, e.g. ``art['CTD_TEMP_CNDC']``; this means that the instance can be used almost like a dictionary (e.g. for inclusion assertion and iteration),
- values export/search methods, e.g. ``art.search()``, ``art.to_dataframe()``, ``art.to_dict()``.

.. note::

    NVS (NERC Vocabulary Server) "vocabulary" are formally `SKOS (Simple Knowledge Organization System) "collection" <https://en.wikipedia.org/wiki/Simple_Knowledge_Organization_System#Concept_collections>`_.

Creation
^^^^^^^^

.. code-block:: python
    :caption: Creation

    from argopy import ArgoReferenceTable

    # Use an Argo parameter name, documented by one of the Argo reference tables:
    art = ArgoReferenceTable('SENSOR')

    # or a reference table identifier:
    art = ArgoReferenceTable('R25')

    # or a URN:
    art = ArgoReferenceTable.from_urn('SDN:R25::CTD_TEMP')

.. ipython:: python
    :okwarning:

    ArgoReferenceTable('SENSOR')


.. tip::

    If you don't know where to start, you can get a dictionary of all available tables with this class method :meth:`ArgoReferenceTable.valid_identifier`:

    .. ipython:: python
        :okwarning:

        ArgoReferenceTable.valid_identifier()


Read attributes
^^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Read attributes

    from argopy import ArgoReferenceTable
    art = ArgoReferenceTable('SENSOR')

    # All possible attributes are listed in:
    art.attrs


Reference Table attributes:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Attribute
      - Description / NVS origin
    * - ``art.parameter``
      - Name of the netCDF dataset parameter filled with values from this table
    * - ``art.identifier``
      - Reference Table ID, eg: ``R25``
    * - ``art.description``
      - ``[nvs['@graph']['@type']=='skos:Collection']["dc:description"]``
    * - ``art.uri``
      - ``[nvs['@graph']['@type']=='skos:Collection']["@id"]``
    * - ``art.version``
      - ``[nvs['@graph']['@type']=='skos:Collection']['owl:versionInfo']``
    * - ``art.date``
      - ``[nvs['@graph']['@type']=='skos:Collection']['dc:date']``
    * - ``art.nvs``
      - Raw NVS json data

Indexing and values
^^^^^^^^^^^^^^^^^^^

Since :class:`ArgoReferenceTable` represents a list of distinct concepts, **Argopy** uses an dictionary-like indexing where *keys* are the table values name:

.. code-block:: python
    :caption: Indexing and values

    from argopy import ArgoReferenceTable
    art = ArgoReferenceTable('SENSOR')

    # Values (or concept) within this reference table:
    len(art)     # Number of reference values
    art.keys()   # List of reference values name
    art.values() # List of :class:`ArgoReferenceValue`

    # Check for values:
    'CTD_TEMP_CNDC' in art  # Return True

    # Index by value key, like a simple dictionary:
    art['CTD_TEMP_CNDC']  # Return a :class:`ArgoReferenceValue` instance

    # Allows to iterate over all values/concepts:
    for concept in art:
        print(concept.name, concept.urn)

.. note::

    In IPython environment, like notebooks, **Argopy** provides an auto-completion feature to easily get one of the values, just press tab when typing ``art['``:

    .. image:: ../../_static/ArgoReferenceTable_autocompletion.png
        :width: 345

Export methods
^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Export methods

    from argopy import ArgoReferenceTable
    art = ArgoReferenceTable('SENSOR')

    # Export table attributes to a dictionary (this does not export values):
    art.to_dict()
    art.to_dict(keys=['parameter', 'date', 'uri'])  # Select Table attributes to export in dictionary keys

    # Export table values to a pd.DataFrame:
    art.to_dataframe()
    art.to_dataframe(columns=['name', 'deprecated'])  # Select value attributes to export in columns

    # Export table values to a dictionary, using pd.DataFrame:
    art.to_dataframe(columns=['name', 'deprecated']).to_dict(orient='records')

Search a table
^^^^^^^^^^^^^^

It is possible to search the table among any of the :class:`ArgoReferenceValue` attributes:

.. code-block:: python
    :caption: Search a table

    from argopy import ArgoReferenceTable
    art = ArgoReferenceTable('SENSOR')

    # Search methods (return a list of :class:`ArgoReferenceValue` with match):
    art.search(name='RAMSES')         # Search in values name
    art.search(definition='imaging')  # Search in values definition
    art.search(long_name='TriOS')     # Search in values long name

    # Possible change the output format:
    art.search(deprecated=True, output='df')  # To a :class:`pd.DataFrame`


.. _argoreferencemapping:

Reference **value relationships**
---------------------------------

**A class to work with Argo Reference Value Relationships, a.k.a. NVS "mapping".**

More explanation from the `AVTT documentation
<https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#ivb-mappings>`_:

    Mappings are used to inform relationship between concepts. For
    instance, inform all the ``sensor_models`` manufactured by one
    ``sensor_maker``, or all the ``platform_types`` manufactured by one
    ``platform_maker``, etc. They are used by the `FileChecker <https://github.com/OneArgo/ArgoFormatChecker>`_ to ensure
    the consistency between these metadata fields in the Argo dataset.

The relationship is also called a *predicate*. The `AVTT documentation
<https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#ivb-mappings>`_ indicates that there are two kinds of predicates:

- "narrower/broader" when there is a hierarchy between the subject and the object,
- and "related" when the subject is related to the object without strict hierarchy.

.. note::

    NVS (NERC Vocabulary Server) "mapping" are formally `SKOS (Simple Knowledge Organization System) "mapping" <https://en.wikipedia.org/wiki/Simple_Knowledge_Organization_System#Mapping>`_.

Creation
^^^^^^^^

Since a mapping describes the relationship between a *subject* and an *object*, you can create a :class:`ArgoReferenceMapping` by providing them in this order:

.. code-block:: python
    :caption: Creation

    from argopy import ArgoReferenceMapping

    # Use two Argo parameter names, documented by one of the Argo reference tables:
    ArgoReferenceMapping('PLATFORM_MAKER', 'PLATFORM_TYPE')

    # or reference table identifiers:
    ArgoReferenceMapping('R24', 'R23')

Here, ``PLATFORM_MAKER`` is the *subject* and ``PLATFORM_TYPE`` the *object*.

Read attributes
^^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Indexing and values

    from argopy import ArgoReferenceMapping
    arm = ArgoReferenceMapping('R24', 'R23')

    arm.subjects   # Ordered list of unique 'subject' reference values names
    arm.objects    # Ordered list of unique 'object' reference values names
    arm.predicates # Ordered list of unique 'predicate', aka relationships, in this mapping

    arm.sub_id        # ID of the 'subject' reference table
    arm.sub_parameter # Parameter name of the 'subject' reference table

    arm.obj_id        # ID of the 'object' reference table
    arm.obj_parameter # Parameter name of the 'object' reference table


Indexing and values
^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    :caption: Indexing and values

    from argopy import ArgoReferenceMapping
    arm = ArgoReferenceMapping('R24', 'R23')

    len(arm) # Number of relationships

    # Check if a reference value is in this mapping as a subject or an object:
    'SBE' in arm  # Return True

    # Indexing is by subject values:
    arm['SBE']  # Return a dict with predicate as keys and objects as values

    # Iterate over all relationships:
    for relation in arm:
        print(relation['subject'], relation['predicate'])

Export method
^^^^^^^^^^^^^

.. code-block:: python
    :caption: Export method

    from argopy import ArgoReferenceMapping
    arm = ArgoReferenceMapping('R24', 'R23')

    # Export all mapping relationships in a DataFrame:
    arm.to_dataframe()

    # To export mapping using AVTT jargon:
    arm.to_dataframe(raw=True)

.. ipython:: python
    :okwarning:

    ArgoReferenceMapping('R24', 'R23').to_dataframe()


Legacy ``ArgoNVSReferenceTables``
---------------------------------

**Argopy** used to provide the utility class :class:`ArgoNVSReferenceTables` to easily fetch and get access to all Argo reference tables.
This utility is deprecated and will be remove from the library at some point.

**Deprecated API:**
    ``ArgoNVSReferenceTables.tbl('R25')``

**New API** (but not backward compatible because column names have changed):
    ``ArgoReferenceTable('R25').to_dataframe()``
