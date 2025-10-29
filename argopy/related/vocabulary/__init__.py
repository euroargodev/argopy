"""

Argo vocabulary and reference framework

Following the AVTT and NVS nomenclature:
 - one NVS "vocabulary" refers to one "reference table".
 - one NVS "concept" refers to one row/value from a "reference table".

For instance:
The vocabulary names "Argo sensor models", corresponds to the reference table 27 ("R27") and is
used to fill values for the "SENSOR_MODEL" parameter in netcdf files.

One possible value for "SENSOR_MODEL" is "AANDERAA_OPTODE_3930", which is one NVS "concept", i.e. one entry of the "Argo sensor models" vocabulary


This sub-module provides classes to work with this framework:

ArgoNVSReferenceTables > ArgoVocabulary > ArgoVocabularyConcept

Should this be renamed:
ArgoNVSReferenceTables > ArgoReferenceTable > ArgoReferenceValue ?

Since the Argo's user manual refers to "Argo reference table", we shall adopt the following class naming:

ArgoNVSReferenceTables/ArgoReferenceTableCollection > ArgoReferenceTable > ArgoReferenceValue
"""
from .reference_tables import ArgoNVSReferenceTables
from .vocabulary import ArgoReferenceTable
from .concept import ArgoReferenceValue

__all__ = (
    "ArgoNVSReferenceTables",
    "ArgoReferenceTable",
    "ArgoReferenceValue",
)