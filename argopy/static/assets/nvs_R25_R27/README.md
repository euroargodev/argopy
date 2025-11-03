# Sensor models and types

This folder hold mapping data to get R25 sensor types from R27 sensor models.

Relevant reference from ADMT/AVTT:
- https://github.com/OneArgo/ArgoVocabs/issues/156
- https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#IVb-Mappings


Columns in mapping files are:

``object_NVS_table, object_concept_id, predicate_code, subject_NVS_table, subject_concept_id, modification_type (I for Insertion)``

> A "predicate" indicates the relationship type between the "subject" and the "object"
>
> For "broader/narrower" relationship, the "BRD" predicate code is used 
>
> For "related" relationship, the "MIN" predicate code is used (minor match)

Eg: 
``R27, AANDERAA_OPTODE , MIN, R25, OPTODE_DOXY , I``

This means that 'object=R27(AANDERAA_OPTODE)' is *related* to 'subject=R25(OPTODE_DOXY)'.
