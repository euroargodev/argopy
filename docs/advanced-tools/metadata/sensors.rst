.. currentmodule:: argopy.related
.. _argosensor:

Argo sensor: models and types
=============================

The :class:`ArgoSensor` class provides direct access to Argo's sensor metadata through Reference Tables `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, combined with the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu/>`_. This enables users to:

- Navigate reference tables,
- Search for floats equipped with specific sensor models,
- Retrieve sensor serial numbers across the global array.

.. contents::
   :local:

.. _argosensor-reference-tables:

Work with reference tables on sensors
-------------------------------------

With the :class:`ArgoSensor` class, you can work with official Argo vocabularies for `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_. With these methods, it is easy to look for a specific sensor model name, which can then be used to :ref:`look for floats <argosensor-search-floats>` or :ref:`create a ArgoSensor class <argosensor-exact-sensor>` directly.

.. list-table:: :class:`ArgoSensor` attributes and methods for reference tables
    :header-rows: 1
    :stub-columns: 1

    * - Attribute/Method
      - Description
    * - :attr:`ArgoSensor.reference_model`
      - Returns Reference Table **Sensor Models (R27)** as a :class:`pandas.DataFrame`
    * - :attr:`ArgoSensor.reference_model_name`
      - List of all sensor model names (e.g., ``'SBE41CP'``)
    * - :attr:`ArgoSensor.reference_sensor`
      - Returns Reference Table **Sensor Types (R25)** as a :class:`pandas.DataFrame`
    * - :attr:`ArgoSensor.reference_sensor_type`
      - List of all sensor types (e.g., ``'CTD'``, ``'OPTODE'``)
    * - :attr:`ArgoSensor.r27_to_r25`
      - Dictionary mapping of R27 to R25
    * - :meth:`ArgoSensor.search_model`
      - Search R27 for models matching a string (exact or fuzzy)
    * - :meth:`ArgoSensor.model_to_type`
      - Returns AVTT mapping on sensor type for a given model name
    * - :meth:`ArgoSensor.type_to_model`
      - Returns AVTT mapping on model names for a given sensor type

Examples
^^^^^^^^

- List all CTD sensor models (R27):

.. ipython:: python
    :okwarning:

    from argopy import ArgoSensor

    ArgoSensor().reference_model

- Fuzzy search (default):

.. ipython:: python
    :okwarning:

    ArgoSensor().search_model('SBE61_V5.0.1', strict=False)

- Exact search:

.. ipython:: python
    :okwarning:

    ArgoSensor().search_model('SBE61_V5.0.1', strict=True)

- List the sensor type of a given model:

.. ipython:: python
    :okwarning:

    ArgoSensor().model_to_type('RBR_ARGO3_DEEP6K')

- List all possible model names of a given sensor type:

.. ipython:: python
    :okwarning:

    ArgoSensor().type_to_model('FLUOROMETER_CDOM')


.. _argosensor-search-floats:

Search for Argo floats equipped with a given sensor model
---------------------------------------------------------

In this section we show how to find WMOs, serial numbers for floats equipped with a specific sensor model using the :meth:`ArgoSensor.search` method.

The method takes a model sensor name as input, and possibly 4 values to `output` to determine how to format results:

- ``output='wmo'`` returns a list of WMOs (e.g., ``[1901234, 1901235]``)
- ``output='sn'`` returns a list of serial numbers (e.g., ``['1234', '5678']``)
- ``output='wmo_sn'`` returns a dict mapping serial numbers  to float WMOs (e.g. ``{WMO: [serial1, serial2]}``)
- ``output='df'`` returns a :class:`pandas.DataFrame` with WMO, sensor type, model, maker, serial number, units, accuracy and resolution.

Examples
^^^^^^^^

- Get WMOs for all floats equiped with the "AANDERAA_OPTODE_4831" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("AANDERAA_OPTODE_4831")  # output='wmo' is default


- Get sensor serial numbers for floats equiped with the "SBE43F_IDO" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("SBE43F_IDO", output="sn")

- Get WMO-serial dictionnary for floats equiped with the "RBR_ARGO3_DEEP6K" model:

.. ipython:: python
    :okwarning:

    wmo_sn = ArgoSensor().search("RBR_ARGO3_DEEP6K", output="wmo_sn")
    wmo_sn

.. _argosensor-exact-sensor:

Use an exact sensor model name to create an instance
----------------------------------------------------

You can initialize an :class:`ArgoSensor` instance for a specific model to access its metadata and methods directly.

.. csv-table:: Attributes and Methods
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   :class:`ArgoSensor`, "Create an instance for an exact model name (e.g., ``'SBE41CP'``)"
   :attr:`ArgoSensor.model`, "Returns a :class:`SensorModel` object (name, long_name, definition, URI, deprecated)"
   :attr:`ArgoSensor.type`, "Returns a :class:`SensorType` object (name, long_name, definition, URI, deprecated)"
   :meth:`ArgoSensor.search`, "Inherits search methods but defaults to the instance's model"

Examples
^^^^^^^^

As an example, let's create an instance for the "SBE43F_IDO" sensor model:

.. ipython:: python
    :okwarning:

    sensor = ArgoSensor("SBE43F_IDO")
    sensor

You can then access model metadata:

.. ipython:: python
    :okwarning:

    sensor.model

.. ipython:: python
    :okwarning:

    sensor.type

And you can look for floats equiped:

.. ipython:: python
    :okwarning:

    df = sensor.search(output="df")
    df

Loop Through ArgoFloat Instances for Each Float
-----------------------------------------------

The :meth:`ArgoSensor.iterfloats_with` will yields :class:`argopy.ArgoFloat` instances for floats with the specified sensor model (use ``chunksize`` to process floats in batches).

Example
^^^^^^^

Loop through all floats with "SATLANTIC_PAR" sensors:

.. ipython:: python
    :okwarning:

    for afloat in ArgoSensor().iterfloats_with("SATLANTIC_PAR"):
        print(f"\n-Float {afloat.WMO}: Platform description = {afloat.metadata['platform']['description']}")
        for sensor in afloat.metadata["sensors"]:
            if "SATLANTIC_PAR" in sensor["model"]:
                print(f"  - Sensor Maker: {sensor['maker']}, Serial: {sensor['serial']}")