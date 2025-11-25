"""
These NVS/Vocabulary utilities should be moved up to the argopy sub-module utils
"""

from functools import lru_cache
from urllib.parse import urlparse

from argopy.utils.accessories import Asset


Vocabulary2Concept = Asset.load('vocabulary:mapping')['data']['Vocabulary2Concept']
Vocabulary2Parameter = Asset.load('vocabulary:mapping')['data']['Vocabulary2Parameter']


@lru_cache
def concept2vocabulary(name: str) -> str | None:
    """Map a 'Reference Value' to a 'Reference Table ID'

    Based on the NVS Vocabulary-to-Concept mapping in static assets

    Examples
    --------
    .. code-block:: python

        concept2vocabulary('FLOAT_COASTAL') # ['R22']

    Returns
    -------
    str | None
        No error is raised, None is returned if the concept is not found in any of the vocabularies
    """
    name = name.strip().upper()
    found = []
    for vocabulary in Vocabulary2Concept:
        if name in Vocabulary2Concept[vocabulary]:
            found.append(vocabulary)
    if len(found) == 0:
        return None
    return found


def check_vocabulary(input: str) -> str | None:
    """Check the input and return a 'Reference Table ID'

    Parameters
    ----------
    input: str
        Reference table name or ID

    Returns
    -------
    str | None
        Reference table ID
    """
    input = input.strip().upper()
    for vocab in Vocabulary2Parameter:
        if input == vocab or input == Vocabulary2Parameter[vocab]:
            return vocab
    return None


def id2urn(uri: str) -> str:
    """Convert a NVS concept ID to a URN

    Examples
    --------
    .. code-block:: python

        id2urn("http://vocab.nerc.ac.uk/collection/R27/current/PAL_UW/") # "SDN:R27::PAL_UW"
    """
    parts = urlparse(uri).path.split("/")
    try:
        listid = [p for p in parts if p.startswith('R')][0]
    except:
        raise ValueError(f"{uri} is not a valid NVS id")
    try:
        termid = parts[1+[parts.index(p) for p in parts if p=='current'][0]]
    except:
        termid = ""
    return f"SDN:{listid}::{termid}"
