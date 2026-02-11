"""
These NVS/Vocabulary utilities should be moved up to the argopy submodule utils
"""

from functools import lru_cache
from urllib.parse import urlparse
import re
from dataclasses import dataclass, make_dataclass
from typing import Any
import pandas as pd

from argopy.utils.locals import Asset
from argopy.utils.format import urnparser

Vocabulary2Concept = Asset.load("vocabulary:mapping")["data"]["Vocabulary2Concept"]
Vocabulary2Parameter = Asset.load("vocabulary:mapping")["data"]["Vocabulary2Parameter"]


@lru_cache
def concept2vocabulary(name: str) -> list[str] | None:
    """Map a 'Reference Value' to a 'Reference Table ID'

    Based on the NVS Vocabulary-to-Concept mapping in static assets

    Returns
    -------
    str | None
        No error is raised, None is returned if the concept is not found in any of the vocabularies

    Examples
    --------
    .. code-block:: python

        concept2vocabulary('FLOAT_COASTAL') # ['R22']

    """
    name = name.strip().upper()
    found : list[str] = []
    for vocabulary in Vocabulary2Concept:
        if name in Vocabulary2Concept[vocabulary]:
            found.append(vocabulary)
    if len(found) == 0:
        return None
    return found


def check_vocabulary(input: str) -> str | None:
    """Check the input and return a 'Reference Table ID', even if the input a table parameter name

    Parameters
    ----------
    input: str
        Reference table name or ID

    Returns
    -------
    str | None
        Reference table ID or None if not found.

    Examples
    --------
    .. code-block:: python

        check_vocabulary('R22')  # Return: 'R22'
        check_vocabulary('PLATFORM_FAMILY')  # Return: 'R22'
        check_vocabulary('dummy')  # Return: None

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

    Returns
    -------
    str

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
        # listid = [p for p in parts if p.startswith('R') or p.startswith('P')][0]
        listid = parts[1 + [parts.index(p) for p in parts if p == "collection"][0]]
        if not (listid[0] == "R" or listid[0] == "P"):
            raise ValueError(
                f"{uri} is not a valid NVS id, only R* and P* collections are allowed."
            )
    except:
        raise ValueError(
            f"{uri} is not a valid NVS id, only R* and P* collections are allowed."
        )
    try:
        termid = parts[1 + [parts.index(p) for p in parts if p == "current"][0]]
    except:
        termid = ""
    return f"SDN:{listid}::{termid}"


def url2predicate(uri: str) -> str | None:
    """Return relation predicate from a NVS binding URL value

    Parameters
    ----------
    uri: str

    Returns
    -------
    str | None

    Examples
    --------
    ..code-block: python

        from argopy.stores.nvs.utils import url2predicate

        url2predicate('http://www.w3.org/2002/07/owl#sameAs/') # owl:sameAs
        url2predicate('http://www.w3.org/2004/02/skos/core#narrower/') # skos:narrower

    """
    try:
        return uri.split("/")[-2].replace("#", ":").replace("core", "skos")
    except:
        return None


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


def extract_template_values(s):
    """Extract Template_Values from a concept definition as a dict

    This function should support R14 and R18 concept definitions.
    """
    # Define the regex pattern to capture the Template_Values section
    pattern = r"Template_Values:\s*\{(.*?)\}"

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

    if "unit" in attributes_dict:
        attributes_dict["unit"] = attributes_dict["unit"].strip("[]").split(", ")

    if 'short_sensor_name' in attributes_dict:
        attributes_dict['short_sensor_name'] = attributes_dict['short_sensor_name'].strip("[]").split(", ")

    return attributes_dict


@dataclass
class LocalAttributes:
    long_name: str
    standard_name: str
    units: str
    valid_min: str
    valid_max: str
    fill_value: str


@dataclass
class Properties:
    category: str  #
    data_type: str
    extra_dim: Any | None = None


@dataclass
class TemplateValues:
    pass


def curate_r03definition(definition: str) -> dict[str, LocalAttributes | Properties] | None:
    la, attrs = None, extract_local_attributes(definition)
    if attrs is not None:
        la = LocalAttributes(
            long_name=attrs.get("long_name", ""),
            standard_name=attrs.get("standard_name", ""),
            units=attrs.get("units", ""),
            valid_min=attrs.get("valid_min", ""),
            valid_max=attrs.get("valid_max", ""),
            fill_value=attrs.get("fill_value", ""),
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
    return {"Local_Attributes": la, "Properties": pr}


def curate_r14definition(definition: str) -> dict[str, 'TemplateValues'] | None:
    attrs = extract_template_values(definition)
    if attrs is not None:
        tv = make_dataclass('TemplateValues', [(key, type(val)) for key, val in attrs.items()])(**attrs)
        return {'Template_Values': tv}
    return None


def curate_r18definition(definition: str) -> dict[str, 'TemplateValues'] | None:
    attrs = extract_template_values(definition)
    if attrs is not None:
        tv = make_dataclass('TemplateValues', [(key, type(val)) for key, val in attrs.items()])(**attrs)
        return {'Template_Values': tv}
    return None


def bindings2df(data: list[dict]) -> pd.DataFrame:
    """Transform a list of bindings to a :class:`pd.DataFrame`"""
    id2concept = lambda x: urnparser(id2urn(x))['termid']
    b = []
    for binding in data:
        b.append(
            {
                "subject": id2concept(binding["subj"]["value"]),
                "predicate": url2predicate(binding["pred"]["value"]),
                "object": id2concept(binding["obj"]["value"]),
                "subject_uri": binding["subj"]["value"],
                "object_uri": binding["obj"]["value"],
            }
        )
    df = (
        pd.DataFrame(b)
        .sort_values(by=["subject", "object"], axis=0)
        .reset_index(drop=True)
        .astype("string")
    )
    return df

