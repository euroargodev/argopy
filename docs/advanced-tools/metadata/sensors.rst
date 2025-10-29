.. currentmodule:: argopy.related
.. _argosensor:

Argo sensors
============

The :class:`ArgoSensor` class aims to provide user-friendly access to Argo's sensor metadata from:

- NVS Reference Tables `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Manufacturers (R26) <http://vocab.nerc.ac.uk/collection/R26>`_
- the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu/>`_

This enables users to:

- navigate reference tables,
- search for floats equipped with specific sensor models,
- retrieve sensor serial numbers across the global array.
- search for/iterate over floats equipped with specific sensor models.


.. contents::
   :local:

.. _argosensor-reference-tables:

Work with reference tables on sensors
-------------------------------------

With the :class:`ArgoSensor` class, you can work with official Argo vocabularies for `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Manufacturers (R26) <http://vocab.nerc.ac.uk/collection/R26>`_. With these methods, it is easy to look for a specific sensor model name, which can then be used to :ref:`look for floats <argosensor-search-floats>` or :ref:`create a ArgoSensor class <argosensor-exact-sensor>` directly.

.. list-table:: :class:`ArgoSensor` methods for reference tables
    :header-rows: 1
    :stub-columns: 1

    * - Methods
      - Description
    * - :meth:`ArgoSensor.ref.model.to_dataframe`
      - Returns Reference Table **Sensor Models (R27)** as a :class:`pandas.DataFrame`
    * - :meth:`ArgoSensor.ref.type.to_dataframe`
      - Returns Reference Table **Sensor Types (R25)** as a :class:`pandas.DataFrame`
    * - :meth:`ArgoSensor.ref.maker.to_dataframe`
      - Returns Reference Table **Sensor Manufacturers (R26)** as a :class:`pandas.DataFrame`

    * - :meth:`ArgoSensor.ref.model.hint`
      - List of all sensor model names (e.g., ``['AANDERAA_OPTODE', ..., 'RBR_ARGO3_DEEP6K', ..., 'SBE41CP', ...]``)
    * - :meth:`ArgoSensor.ref.type.hint`
      - List of all sensor types (e.g., ``[..., 'CTD_CNDC', ..., 'OPTODE_DOXY', ...]``)
    * - :meth:`ArgoSensor.ref.maker.hint`
      - List of all sensor makers (e.g., ``[..., 'DRUCK', ... 'SEASCAN', ... ]``)

    * - :meth:`ArgoSensor.ref.model.to_type`
      - Returns sensor type of a given model
    * - :meth:`ArgoSensor.ref.type.to_model`
      - Returns model names of a given sensor type

    * - :meth:`ArgoSensor.ref.model.search`
      - Search R27 for models matching a string, can use wildcard (eg: ``'SBE61*'``)

Examples
^^^^^^^^

- List all CTD models from a given type:

.. ipython:: python
    :okwarning:

    from argopy import ArgoSensor

    ArgoSensor().ref.type.to_model('CTD_CNDC')

- Get the sensor type(s) of a given model:

.. ipython:: python
    :okwarning:

    ArgoSensor().ref.model.to_type('RBR_ARGO3_DEEP6K')

- Search all SBE61 versions with a wildcard:

.. ipython:: python
    :okwarning:

    ArgoSensor().ref.model.search('SBE61*')

- Get one model single model description (see also :attr:`ArgoSensor.vocabulary`):

.. ipython:: python
    :okwarning:

    ArgoSensor().ref.model.search('SBE61_V5.0.1')


.. _argosensor-search-floats:

Search for Argo floats equipped with a given sensor model
---------------------------------------------------------

In this section we show how to find WMOs, and possibly more sensor metadata like serial numbers, for floats equipped with a specific sensor model.
This can be done using the :meth:`ArgoSensor.search` method.

The method takes one or more model sensor names as input, and return results in 4 different formats with the ``output`` option:

- ``output='wmo'`` returns a list of WMOs (e.g., ``[1901234, 1901235]``)
- ``output='sn'`` returns a list of serial numbers (e.g., ``['1234', '5678']``)
- ``output='wmo_sn'`` returns a dictionary mapping float WMOs to serial numbers (e.g. ``{1900166: ['0325', '2657720242']}``)
- ``output='df'`` returns a :class:`pandas.DataFrame` with WMO, sensor type, model, maker, serial number, units, accuracy and resolution in columns.

.. note::

    This method can potentially lead to a scan of the entire Argo float collection, which would be very time consuming. Therefore the :meth:`ArgoSensor.search` method takes only exact sensor models and will raise an error if a wildcard is found.

Examples
^^^^^^^^

- Get WMOs for all floats equipped with the "AANDERAA_OPTODE_4831" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("AANDERAA_OPTODE_4831")  # output='wmo' is default


- Get sensor serial numbers for floats equipped with the "SBE43F_IDO" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("SBE43F_IDO", output="sn")

- Get everything for floats equipped with the "RBR_ARGO3_DEEP6K" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("RBR_ARGO3_DEEP6K", output="df")


.. _argosensor-exact-sensor:

Use an exact sensor model name to create a specific ArgoSensor
--------------------------------------------------------------

You can initialize an :class:`ArgoSensor` instance with a specific model to access more metadata.

.. csv-table:: :class:`ArgoSensor` Attributes and Methods for a specific model
   :header: "Attribute/Method", "Description"
   :widths: 30, 70

   :class:`ArgoSensor`, "Create an instance for an exact model name (e.g., ``'SBE43F_IDO'``)"
   :attr:`ArgoSensor.vocabulary`, "Returns a :class:`SensorModel` object with R27 concept vocabulary (name, long_name, definition, URI, deprecated)"
   :attr:`ArgoSensor.type`, "Returns a :class:`SensorType` object with R25 concept (name, long_name, definition, URI, deprecated)"
   :meth:`ArgoSensor.search`, "Inherits search methods but defaults to the instance's model"

Examples
^^^^^^^^

As an example, let's create an instance for the "SBE43F_IDO" sensor model:

.. ipython:: python
    :okwarning:

    sensor = ArgoSensor("SBE43F_IDO")
    sensor

You can then access this model metadata from the NVS vocabulary (Reference table R27):

.. ipython:: python
    :okwarning:

    sensor.vocabulary

from Reference table R25:

.. ipython:: python
    :okwarning:

    sensor.type

And you can look for floats equipped with it:

.. ipython:: python
    :okwarning:

    df = sensor.search(output="df")
    df

Loop through ArgoFloat instances for each float
-----------------------------------------------

The :meth:`ArgoSensor.iterfloats_with` will yields :class:`argopy.ArgoFloat` instances for floats with the specified sensor model (use ``chunksize`` to process floats in batches).

Example
^^^^^^^

Let's try to gather all platform types of WMOs equipped with a list of sensor models:

.. ipython:: python
    :okwarning:

        sensors = ArgoSensor()

        models = ['ECO_FLBBCD_AP2', 'ECO_FLBBCD']
        results = {}
        for af in sensors.iterfloats_with(models):
            if 'meta' in af.ls_dataset():
                platform_type = af.metadata['platform']['type']  # e.g. 'PROVOR_V_JUMBO'
                if platform_type in results.keys():
                    results[platform_type].extend([af.WMO])
                else:
                    results.update({platform_type: [af.WMO]})
            else:
                print(f"No meta file for float {af.WMO}")
        print(results.keys())


Quick float and sensor lookups from the command line
----------------------------------------------------

The :meth:`ArgoSensor.cli_search` function is available to search for Argo floats equipped with a given sensor model from the command line and retrieve a `CLI <https://en.wikipedia.org/wiki/Command-line_interface>`_ friendly list of WMOs or serial numbers.

You will take note that all output *format* are available to determine the information to retrieve, but that the actual standard output will be a serialized string for easy piping to other tools.

Here is an example with the "SATLANTIC_PAR" sensor.

.. code-block:: bash
    :caption: Call :meth:`ArgoSensor.search` from the command line

    python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('SATLANTIC_PAR', output='wmo')"
    python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('SATLANTIC_PAR', output='sn')"
    python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('SATLANTIC_PAR', output='wmo_sn')"
    python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('SATLANTIC_PAR', output='df')"
