
class ArgoVocabulary:
    """A class to work with one Argo reference table, data and meta-data

    Note that following the AVTT and NVS nomenclature, a "vocabulary" refers to a "reference table".

    For instance the vocabulary for "Argo sensor models", corresponds to the reference table 27 ("R27")
    used to fill values for the "SENSOR_MODEL" parameter in netcdf files.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoVocabulary

        av = ArgoVocabulary(25)
        av = ArgoVocabulary('R25')
        av = ArgoVocabulary.for_parameter('SENSOR')  # Use internal parameter-2-reference-table mapping

        # Vocabulary meta-data:
        av.name        # data["@graph"]["dc:alternative"]
        av.description # data["@graph"]["dc:description"]
        av.uri         # data["@graph"]["@id"]
        av.creator
        av.created
        av.version
        av._data       # Raw NVS json data

        # Name of the netcdf dataset parameter filled with these values, (a simple view of the av.name attribute)
        av.parameter  # eg 'SENSOR'

        # Number of concept within this reference table, see ArgoVocabularyConcept
        av.n_concept  # eg

        # Full vocabulary content:
        av.to_dataframe()  # Return a pd.DataFrame with all concepts standard info.: name, long_name, definition, deprecated, uri, urn
        av.to_concept()    # Return a list of ArgoVocabularyConcept, for each row of the table.

        # Search methods:
        av.search(values='RAMSES')       # Search for an exact sring match in altLabel, return an ArgoVocabularyConcept instance
        av.search(definition='imaging')  # Search in definition, return a list of ArgoVocabularyConcept match
        av.search(prefLabel='TriOS')     # Search in prefLabel, return a list of ArgoVocabularyConcept match

    """
    pass