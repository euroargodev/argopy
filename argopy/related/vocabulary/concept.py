from dataclasses import dataclass

@dataclass
class ArgoVocabularyConcept:
    """A class to work with an Argo vocabulary concept

    An Argo vocabulary concept is one possible value of one Argo netcdf parameter, i.e. one row from a NVS reference table.

    This basicaly turns :class:`pd.DataFrame` columns into class attributes.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoVocabularyConcept

        avc = ArgoVocabularyConcept('AANDERAA_OPTODE_3835')  # One possible value for the netcdf parameter 'SENSOR_MODEL'
        avc = ArgoVocabularyConcept.from_urn('SDN:R27::AANDERAA_OPTODE_3835')

        avc.name       # pd.DataFrame['altLabel'] > urnparser(data["@graph"]['skos:notation'])['termid']
        avc.long_name  # pd.DataFrame['prefLabel'] > data["@graph"]["skos:prefLabel"]["@value"]
        avc.definition # pd.DataFrame['definition'] > data["@graph"]["skos:definition"]["@value"]
        avc.deprecated # pd.DataFrame['deprecated'] > data["@graph"]["owl:deprecated"]
        avc.uri        # pd.DataFrame['id'] > data["@graph"]["@id"]
        avc.urn        # pd.DataFrame['urn'] > data["@graph"]["skos:notation"]
        av._data       # Raw NVS json data

        avc.parameter  # The netcdf parameter this concept applies to (eg 'SENSOR_MODEL')
        avc.vocabulary # The reference table (i.e. ArgoVocabulary) this concept belongs to

    """
    pass