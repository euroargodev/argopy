.. currentmodule:: argopy
.. _argosensor:

Argo sensors
============

The :class:`ArgoSensor` class aims to provide user-friendly access to Argo's sensor metadata from:

- NVS Reference Tables `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Manufacturers (R26) <http://vocab.nerc.ac.uk/collection/R26>`_,
- the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu/>`_.

This enables users to:

- navigate reference tables,
- search for floats equipped with specific sensor models,
- retrieve sensor serial numbers across the global array,
- search for/iterate over floats equipped with specific sensor models.


.. contents::
   :local:
   :depth: 1

.. _argosensor-reference-tables:

Working with reference tables
-----------------------------


With the :class:`ArgoSensor` class, you can work with official Argo vocabularies for `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Manufacturers (R26) <http://vocab.nerc.ac.uk/collection/R26>`_. With the following methods, it is easy to look for a specific sensor model name, which can then be used to :ref:`look for floats <argosensor-search-floats>` or :ref:`create a ArgoSensor class <argosensor-exact-sensor>` directly.


.. currentmodule:: argopy

**Content** from reference tables:

.. autosummary::
   :template: autosummary/accessor_method.rst

   ArgoSensor.ref.model.to_dataframe
   ArgoSensor.ref.model.hint

   ArgoSensor.ref.type.to_dataframe
   ArgoSensor.ref.type.hint

   ArgoSensor.ref.maker.to_dataframe
   ArgoSensor.ref.maker.hint

**Mapping** of sensor model vs type:

.. autosummary::
   :template: autosummary/accessor_method.rst

   ArgoSensor.ref.model.to_type
   ArgoSensor.ref.type.to_model

Model **search**:

.. autosummary::
   :template: autosummary/accessor_method.rst

   ArgoSensor.ref.model.search


Examples
^^^^^^^^

- List all CTD models from a given type:

.. ipython:: python
    :okwarning:

    from argopy import ArgoSensor

    models = ArgoSensor().ref.type.to_model('CTD_CNDC')
    print(models)

- Get the sensor type(s) of a given model:

.. ipython:: python
    :okwarning:

    types = ArgoSensor().ref.model.to_type('RBR_ARGO3_DEEP6K')
    print(types)

- Search all SBE61 versions with a wildcard:

.. ipython:: python
    :okwarning:

    ArgoSensor().ref.model.search('SBE61*')

- Get one single model description (see also :attr:`ArgoSensor.vocabulary`):

.. ipython:: python
    :okwarning:

    ArgoSensor().ref.model.search('SBE61_V5.0.1').T


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

    serials = ArgoSensor().search("SBE43F_IDO", output="sn")
    print(serials)

- Get everything for floats equipped with the "RBR_ARGO3_DEEP6K" model:

.. ipython:: python
    :okwarning:

    ArgoSensor().search("RBR_ARGO3_DEEP6K", output="df")


.. _argosensor-exact-sensor:

Use an exact sensor model name to create a specific ArgoSensor
--------------------------------------------------------------

You can initialize an :class:`ArgoSensor` instance with a specific model to access more metadata. In this use-case, you will have the following attributes and methods available:

.. currentmodule:: argopy

.. autosummary::

   ArgoSensor.vocabulary
   ArgoSensor.type
   ArgoSensor.search
   ArgoSensor.iterfloats_with

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

and from Reference table R25:

.. ipython:: python
    :okwarning:

    sensor.type

You can also look for floats equipped with it:

.. ipython:: python
    :okwarning:

    df = sensor.search(output="df")
    df

Loop through ArgoFloat instances for each float
-----------------------------------------------

The :meth:`ArgoSensor.iterfloats_with` will yields :class:`argopy.ArgoFloat` instances for floats with the specified sensor model (use ``chunksize`` to process floats in batches).

Example
^^^^^^^

Let's try to gather all platform types and WMOs for floats equipped with a list of sensor models:

.. code-block:: python

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

    [f"{r:15s}: {len(results[r])} floats" for r in results.keys()]

.. code-block:: python

    ['APEX           : 37 floats',
     'SOLO_BGC_MRV   : 19 floats',
     'PROVOR_V_JUMBO : 43 floats',
     'PROVOR_III     : 206 floats',
     'PROVOR_V       : 38 floats',
     'PROVOR         : 18 floats',
     'PROVOR_IV      : 21 floats',
     'SOLO_BGC       : 4 floats']

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
