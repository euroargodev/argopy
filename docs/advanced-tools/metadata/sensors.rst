.. currentmodule:: argopy.related
.. _argosensor:

ArgoSensor
==========

**Simplify Argo float sensor queries with standardized metadata access.**

The :class:`ArgoSensor` class provides direct access to Argo's sensor metadata through Reference Tables R25 (sensor types) and R27 (sensor models), combined with the Euro-Argo fleet-monitoring API. This enables users to:

- Search for floats equipped with specific sensor models
- Retrieve sensor serial numbers across the global array

.. _argosensor-reference-tables:

Access and Search Reference Tables 25 and 27
--------------------------------------------

**Purpose**: Explore the official Argo vocabularies for sensor types (R25) and models (R27).

.. csv-table:: Attributes and Methods
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   ``reference_model``, "Returns Reference Table **R27** (``SENSOR_MODEL``) as a ``pandas.DataFrame``"
   ``reference_model_name``, "List of all sensor model names (e.g., ``'SBE41CP'``)"
   ``reference_sensor``, "Returns Reference Table **R25** (``SENSOR``) as a ``pandas.DataFrame``"
   ``reference_sensor_type``, "List of all sensor types (e.g., ``'CTD'``, ``'OPTODE'``)"
   ``search_model(model, strict)``, "Search R27 for models matching a string (exact or fuzzy)"
   ``model_to_type(model_name)``, "Returns AVTT mapping on sensor type for a given model name"
   ``type_to_model(sensor_type)``, "Returns AVTT mapping on model names for a given sensor type"

**Example**:

.. code-block:: python

    from argopy import ArgoSensor

    # List all CTD sensor models (R27)
    ArgoSensor().reference_model

    # Fuzzy search (default):
    ArgoSensor().search_model('RBR', strict=False)

    # Exact search:
    ArgoSensor().search_model('SBE61_V5.0.12', strict=True)

    # List the sensor type of a given model:
    ArgoSensor().model_to_type('SBE61')

    # List all possible model names of a given sensor type:
    ArgoSensor().type_to_model('FLUOROMETER_CDOM')


.. _argosensor-search-floats:

Search the Argo Array for Floats Equipped with a Given Sensor Model
-------------------------------------------------------------------

**Purpose**: Find WMOs, serial numbers, or any other sensor information floats equipped with a specific sensor model.

.. csv-table:: Attributes and Methods
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   ``search(model, output)``, "Search for floats with a sensor model"
   "", "``output='wmo'``: Returns list of WMOs (e.g., ``[1901234, 1901235]``)"
   "", "``output='sn'``: Returns list of serial numbers (e.g., ``['1234', '5678']``)"
   "", "``output='wmo_sn'``: Returns dict ``{WMO: [serial1, serial2]}``"
   "", "``output='df'``: Returns a :class:`pandas.DataFrame` with WMO, sensor type, model, maker, sn, units, accuracy and resolution

**Example**:

.. code-block:: python

    from argopy import ArgoSensor

    # Get WMOs for all floats with "AANDERAA_OPTODE_4831"
    wmos = ArgoSensor().search("KISTLER_10153PSIA", output="df")

    # Get WMOs for all floats with "AANDERAA_OPTODE_4831"
    wmos = ArgoSensor().search("AANDERAA_OPTODE_4831", output="wmo")

    # Get serial numbers for the "SBE43F_IDO" model
    serials = ArgoSensor().search("SBE43F_IDO", output="sn")

    # Get WMO-serial pairs for "RBR_ARGO3_DEEP6K"
    wmo_sn = sensor_tool.search("RBR_ARGO3_DEEP6K", output="wmo_sn")
    for wmo, sns in list(wmo_sn.items()):  # First float
        print(f"Float {wmo}: Serials {sns}")


Using an Exact Sensor Model Name to Create an Instance
------------------------------------------------------

**Purpose**: Initialize an :class:`ArgoSensor` instance for a specific model to access its metadata and methods directly.

.. csv-table:: Attributes and Methods
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   ``ArgoSensor(model)``, "Create an instance for an exact model name (e.g., ``'SBE41CP'``)"
   ``model``, "Returns a ``SensorModel`` object (name, long_name, definition, URI, deprecated)"
   ``type``, "Returns a ``SensorType`` object (name, long_name, definition, URI, deprecated)"
   ``search(output)``, "Inherits search methods but defaults to the instance's model"

**Example**:

.. code-block:: python

   # Create an instance for the "SBE43F_IDO" sensor model:
   sbe_sensor = ArgoSensor("SBE43F_IDO")

   # Access model metadata
   print(f"Model: {sbe_sensor.model.name}")
   print(f"Type: {sbe_sensor.type.name}")
   print(f"Definition: {sbe_sensor.model.definition}")

   # Search for floats with this model
   df = sbe_sensor.search(output="df")

    # WMO	Type	Model	Maker	SerialNumber	Units	Accuracy	Resolution
    # 0	1901328	IDO_DOXY	SBE43F_IDO	SBE	577	micro moles	5.0	None
    # 1	1901329	IDO_DOXY	SBE43F_IDO	SBE	579	micro moles	5.0	None
    # 2	2900114	IDO_DOXY	SBE43F_IDO	SBE	0164	micromole/kg	NaN	None
    # 3	2900115	IDO_DOXY	SBE43F_IDO	SBE	0188	micromole/kg	NaN	None
    # 4	2900116	IDO_DOXY	SBE43F_IDO	SBE	0179	micromole/kg	NaN	None


Loop Through ArgoFloat Instances for Each Float
-----------------------------------------------

**Purpose**: Iterate over ``ArgoFloat`` instances for floats matching a sensor model.

.. csv-table:: Attributes and Methods
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   ``iterfloats_with(model)``, "Yields :class:`ArgoFloat` instances for floats with the specified sensor model"
   "", "Use ``chunksize`` to process floats in batches"

**Example**:

.. code-block:: python

    from argopy import ArgoSensor
    # Loop through all floats with "RAFOS" sensors
    for afloat in ArgoSensor().iterfloats_with("RAFOS"):
       print(f"Float {afloat.WMO}: Platform type = {afloat.metadata['platform_type']}")

       # Access sensor metadata
       for sensor in afloat.metadata["sensors"]:
           if "RAFOS" in sensor["model"]:
               print(f"  - Maker: {sensor['maker']}, Serial: {sensor['serial']}")

    # Output:
    # Float 1901234: Platform type = APF11
    #   - Maker: Webb Research, Serial: RAFOS-001
