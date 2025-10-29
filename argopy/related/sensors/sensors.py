from ...utils import register_accessor
from .spec import ArgoSensorSpec
from .references import SensorReferences


class ArgoSensor(ArgoSensorSpec):
    """Argo sensor(s) helper class

    The :class:`ArgoSensor` class aims to provide direct access to Argo's sensor metadata from:

    - NVS Reference Tables `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_, `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Manufacturers (R26) <http://vocab.nerc.ac.uk/collection/R26>`_
    - `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu/>`_

    This enables users to:

    - navigate reference tables 27, 25 and 26,
    - retrieve sensor serial numbers across the global array,
    - search for/iterate over floats equipped with specific sensor models.


    Examples
    --------
    .. code-block:: python
        :caption: Access reference tables for 'SENSOR_MODEL'/R27, 'SENSOR'/R25 and 'SENSOR_MAKER'/R26

        from argopy import ArgoSensor

        ArgoSensor().ref.model.to_dataframe() # Return reference table R27 with the list of sensor models as a DataFrame
        ArgoSensor().ref.model.to_list()      # Return list of sensor model names (possible values for 'SENSOR_MODEL' parameter)
        ArgoSensor().ref.model.to_type('SBE61') # Return sensor type (R25) of a given model (R27)

        ArgoSensor().ref.type.to_dataframe()  # Return reference table R25 with the list of sensor types as a DataFrame
        ArgoSensor().ref.type.to_list()       # Return list of sensor types (possible values for 'SENSOR' parameter)
        ArgoSensor().ref.type.to_model('FLUOROMETER_CDOM') # Return all possible model names (R27) for a given sensor type (R25)

        ArgoSensor().ref.maker.to_dataframe()  # Return reference table R26 with the list of manufacturer as a DataFrame
        ArgoSensor().ref.maker.to_list()       # Return list of manufacturer names (possible values for 'SENSOR_MAKER' parameter)

    .. code-block:: python
        :caption: Search in 'SENSOR_MODEL'/R27 reference table

        from argopy import ArgoSensor

        ArgoSensor().ref.model.search('RBR')  # Search and return a DataFrame
        ArgoSensor().ref.model.search('RBR', output='name') # Search and return a list of names instead
        ArgoSensor().ref.model.search('SBE61*')  # Use of wildcards
        ArgoSensor().ref.model.search('*Deep*')  # Search is case-insensitive

    .. code-block:: python
        :caption: Search for all Argo floats equipped with one or more exact sensor model(s)

        from argopy import ArgoSensor

        sensors = ArgoSensor()

        # Search and return a list of WMOs equipped
        sensors.search('SBE61_V5.0.2')

        # Search and return a list of sensor serial numbers in Argo
        sensors.search('ECO_FLBBCD_AP2', output='sn')

        # Search and return a list of tuples with WMOs and sensors serial number
        sensors.search('SBE', output='wmo_sn')

        # Search and return a DataFrame with full sensor information from floats equipped
        sensors.search('RBR', output='df')

        # Search multiple models at once
        sensors.search(['ECO_FLBBCD_AP2', 'ECO_FLBBCD'])


    .. code-block:: python
        :caption: Easily loop through :class:`ArgoFloat` instances for each float equipped with a sensor model

        from argopy import ArgoSensor

        sensors = ArgoSensor()

        # Trivial example:
        model = "RAFOS"
        for af in sensors.iterfloats_with(model):
            print(af.WMO)

        # Example to gather all platform types for all WMOs equipped with a list of sensor models
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

    .. code-block:: python
        :caption: Use an exact sensor model name to create an instance

        from argopy import ArgoSensor

        sensor = ArgoSensor('RBR_ARGO3_DEEP6K')

        sensor.vocabulary
        sensor.type

        sensor.search(output='wmo')
        sensor.search(output='sn')
        sensor.search(output='wmo_sn')

    .. code-block:: bash
        :caption: Get serialised search results from the command-line with :class:`ArgoSensor.cli_search`

        python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo')"
        python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='sn')"
        python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo_sn')"
        python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='df')"


    Notes
    -----
    Related ADMT/AVTT work:
        - https://github.com/OneArgo/ADMT/issues/112
        - https://github.com/OneArgo/ArgoVocabs/issues/156
        - https://github.com/OneArgo/ArgoVocabs/issues/157

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_accessor("ref", ArgoSensor)
class References(SensorReferences):
    """An :class:`ArgoSensor` extension dedicated to reference tables appropriate for sensors

    Examples
    --------
    .. code-block:: python
        :caption: Access reference tables for 'SENSOR_MODEL'/R27, 'SENSOR'/R25 and 'SENSOR_MAKER'/R26

        from argopy import ArgoSensor

        ArgoSensor().ref.model.to_dataframe() # Return reference table R27 with the list of sensor models as a DataFrame
        ArgoSensor().ref.model.to_list()      # Return list of sensor model names (possible values for 'SENSOR_MODEL' parameter)
        ArgoSensor().ref.model.to_type('SBE61') # Return sensor type (R25) of a given model (R27)

        ArgoSensor().ref.type.to_dataframe()  # Return reference table R25 with the list of sensor types as a DataFrame
        ArgoSensor().ref.type.to_list()       # Return list of sensor types (possible values for 'SENSOR' parameter)
        ArgoSensor().ref.type.to_model('FLUOROMETER_CDOM') # Return all possible model names (R27) for a given sensor type (R25)

        ArgoSensor().ref.maker.to_dataframe()  # Return reference table R26 with the list of manufacturer as a DataFrame
        ArgoSensor().ref.maker.to_list()       # Return list of manufacturer names (possible values for 'SENSOR_MAKER' parameter)

    .. code-block:: python
        :caption: Search in 'SENSOR_MODEL'/R27 reference table

        from argopy import ArgoSensor

        ArgoSensor().ref.model.search('RBR')  # Search and return a DataFrame
        ArgoSensor().ref.model.search('RBR', output='name') # Search and return a list of names instead
        ArgoSensor().ref.model.search('SBE61*')  # Use of wildcards
        ArgoSensor().ref.model.search('*Deep*')  # Search is case-insensitive

    """

    _name = "ref"
