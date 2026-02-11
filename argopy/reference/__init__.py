"""

Argo vocabulary and reference framework

Following the AVTT and NVS nomenclature:
 - one NVS "vocabulary" refers to one "reference table".
 - one NVS "concept" refers to one row/value from a "reference table".

For instance:
The vocabulary named "Argo sensor models", corresponds to the reference table "R27" and is
used to fill values for the "SENSOR_MODEL" parameter in netcdf files.

One possible value for "SENSOR_MODEL" is "AANDERAA_OPTODE_3930", which is one NVS "concept", i.e. one entry of the "Argo sensor models" vocabulary


But since the Argo's user manual refers to "Argo reference table", we shall adopt this convention and avoid the NVS jargon of vocabulary, collection, concept.

This sub-module provides classes to work with this framework:

ArgoReference > ArgoReferenceTable > ArgoReferenceValue


.. note::

    For developers and internal use only, we still use the NVS jargon in file naming.
"""

# To be deprecated:
from .reference_tables import ArgoNVSReferenceTables

# New APIs:
from .concept import ArgoReferenceValue
from .vocabulary import ArgoReferenceTable
from .mapping import ArgoReferenceMapping
from .collection import ArgoReference

__all__ = (
    "ArgoNVSReferenceTables",
    "ArgoReferenceTable",
    "ArgoReferenceValue",
    "ArgoReferenceMapping",
    "ArgoReference",
)