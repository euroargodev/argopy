.. currentmodule:: argopy.related

Deployment Plan
---------------

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

.. image:: ../../_static/scatter_map_deployment_status.png


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
