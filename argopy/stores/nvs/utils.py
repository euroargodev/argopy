"""
These NVS/Vocabulary utilities should be moved up to the argopy sub-module utils
"""

from functools import lru_cache
from urllib.parse import urlparse
import re
from dataclasses import dataclass
from typing import Any

from argopy.utils.locals import Asset


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

    Parameters
    ----------
    uri: str
        The NVS concept ID (URL-like) to analyze.

    Examples
    --------
    .. code-block:: python

        id2urn("http://vocab.nerc.ac.uk/collection/R27/current/PAL_UW/") # "SDN:R27::PAL_UW"
        id2urn("http://vocab.nerc.ac.uk/collection/P06/current/UUPH/") # "SDN:P06::UUPH"

    Warnings
    --------
    Only R* and P* NVS collections are supported.

    """
    parts = urlparse(uri).path.split("/")
    try:
        listid = [p for p in parts if p.startswith('R') or p.startswith('P')][0]
    except:
        raise ValueError(f"{uri} is not a valid NVS id")
    try:
        termid = parts[1+[parts.index(p) for p in parts if p=='current'][0]]
    except:
        termid = ""
    return f"SDN:{listid}::{termid}"


def extract_local_attributes(s):
    # Define the regex pattern to capture the Local_Attributes section
    pattern = r"Local_Attributes:\s*\{(.*?)\}"

    # Find the Local_Attributes section
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        # raise ValueError("No Local_Attributes section found in the string.")
        return None

    # Extract the content inside the curly braces
    attributes_str = match.group(1)

    # Define the regex pattern to capture key/value pairs
    pair_pattern = r"(\w+):([^;]+)"

    # Find all key/value pairs
    pairs = re.findall(pair_pattern, attributes_str)

    # Convert the pairs into a dictionary
    attributes_dict = {key.strip(): value.strip() for key, value in pairs}

    return attributes_dict

def extract_properties_section(s):
    # Define the regex pattern to capture the Properties section
    pattern = r"Properties:\s*\{(.*?)\}"

    # Find the Properties section
    match = re.search(pattern, s, re.DOTALL)
    if not match:
        # raise ValueError("No Properties section found in the string.")
        return None

    # Extract the content inside the curly braces
    properties_str = match.group(1)

    # Define the regex pattern to capture key/value pairs
    pair_pattern = r"(\w+):([^;]+)"

    # Find all key/value pairs
    pairs = re.findall(pair_pattern, properties_str)

    # Convert the pairs into a dictionary
    properties_dict = {key.strip(): value.strip() for key, value in pairs}

    return properties_dict

@dataclass
class LocalAttributes:
    long_name : str
    standard_name : str
    units : str
    valid_min : str
    valid_max : str
    fill_value : str

@dataclass
class Properties:
    category : str  #
    data_type : str
    extra_dim : Any | None = None

def read_r03definition(definition: str) -> dict[str, str]:
    la, attrs = None, extract_local_attributes(definition)
    if attrs is not None:
        la = LocalAttributes(
            long_name=attrs.get("long_name", ""),
            standard_name=attrs.get("standard_name", ""),
            units=attrs.get("units", ""),
            valid_min=attrs.get("valid_min", ""),
            valid_max=attrs.get("valid_max", ""),
            fill_value=attrs.get("fill_value", "")
        )
    pr, props = None, extract_properties_section(definition)
    if props is not None:
        pr = Properties(
            category=props.get("category", ""),
            data_type=props.get("data_type", ""),
            extra_dim=props.get("extra_dim", None),
        )
    if la is None and pr is None:
        return None
    return {'local_attributes': la, 'properties': pr}
