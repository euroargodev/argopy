This is a temporary static set.

It will be deprecated in favor of static:assets:vocabulary:offline:mapping

# Mappings

Description:
https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#IVb-Mappings

Current status:
https://github.com/OneArgo/ArgoVocabs/issues?q=is%3Aissue%20state%3Aopen%20label%3Amappings

From the AVTT:
> Mappings are used to inform relationship between concepts. For instance, inform all the sensor_models manufactured by one sensor_maker, or all the platform_types manufactures by one platform_maker, etc. They are used by the FileChecker to ensure the consistency between these metadata fields in the Argo dataset.

This folder hold mapping data for miscellaneous tables.

|Table|Mapped to| Argopy support status |
|---|---|----------------------|
|**R08**|R23| ❌                    |
|**R15**|RMC, RTV| ❌                    |
|**R23**|R08, R24| ❌                    |
|**R24**|R23| ❌                    |
|**R25**|R27| ✅                     |
|**R26**|R27| ❌                    |
|**R27**|R25, R26| ✅, ❌                 |
|**RMC**|R15| ❌                    |
|**RTV**|R15| ❌                    |


## Examples

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
