from functools import lru_cache

from ....utils import Asset


Vocabulary2Concept = Asset.load('vocabulary:mapping')['data']['Vocabulary2Concept']

@lru_cache
def concept2vocabulary(name: str) -> str | None:
    """Map a 'Reference Value' to a 'Reference Table'

    Based on the NVS Vocabulary-to-Concept mapping in assets

    Returns
    -------
    str | None
        No error is raised, None is returned if the concept is not found in any of the vocabulary
    """
    name = name.strip().upper()
    found = []
    for vocabulary in Vocabulary2Concept:
        if name in Vocabulary2Concept[vocabulary]:
            found.append(vocabulary)
    if len(found) == 0:
        return None
    return found


