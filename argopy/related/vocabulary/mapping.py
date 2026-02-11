import warnings
import pandas as pd
from typing import Any
from copy import deepcopy
import numpy as np

from argopy.options import OPTIONS
from argopy.stores.nvs import NVS
from argopy.stores.nvs.utils import bindings2df, id2urn, url2predicate
from argopy.utils.format import ppliststr, urnparser
from argopy.utils.locals import Asset


# List of valid semantic relations:
valid_relations: list[tuple((str, str))] = [
    ("R08", "R23"),
    ("R24", "R23"),
    ("RMC", "R15"),
    ("RTV", "R15"),
    ("R25", "R27"),
    ("R26", "R27"),
]
# Include reciprocate:
valid_relations.extend([(o, s) for s, o in valid_relations])

id2concept = lambda x: urnparser(id2urn(x))["termid"]

predicate = lambda x: url2predicate(x).split(":")[-1]  # Remove NVS jargon ('skos:', or 'owl:')

class ArgoReferenceMapping:
    """A class to work with Argo Reference Value Relationships, i.e. a NVS "mapping"

    More details from the AVTT documentation:
    https://github.com/OneArgo/ArgoVocabs?tab=readme-ov-file#ivb-mappings

    > Mappings are used to inform relationship between concepts. For instance, inform all the sensor_models manufactured by one sensor_maker, or all the platform_types manufactures by one platform_maker, etc.
    > They are used by the FileChecker to ensure the consistency between these metadata fields in the Argo dataset.

    Examples
    --------
    ..code-block: python
        :caption: Creation

        from argopy import ArgoReferenceMapping

        # Use two Argo parameter names, documented by one of the Argo reference tables:
        ArgoReferenceMapping('PLATFORM_MAKER', 'PLATFORM_TYPE')

        # or reference table identifiers:
        ArgoReferenceMapping('R24', 'R23')

    .. code-block:: python
        :caption: Indexing and values

        from argopy import ArgoReferenceMapping
        arm = ArgoReferenceMapping('R24', 'R23')

        # Relationships within this reference mapping:
        len(arm)     # Number of relationships
        arm.subjects   # Ordered list of unique 'subject' reference values names
        arm.objects    # Ordered list of unique 'object' reference values names
        arm.predicates # Ordered list of unique 'predicate', aka relationships, in this mapping

        # Check if a reference value is in this mapping as a subject or an object:
        'SBE' in arm  # Return True

        # Indexing is by subject values:
        arm['SBE']  # Return a dict with predicate as keys and objects as values

        # Iterate over all relationships:
        for relation in arm:
            print(relation['subject'], relation['predicate'])

    .. code-block:: python
        :caption: Export method

        from argopy import ArgoReferenceMapping
        arm = ArgoReferenceMapping('R24', 'R23')

        # Export all mapping relationships in a DataFrame:
        arm.to_dataframe()

        # To export mapping using AVTT jargon:
        arm.to_dataframe(raw=True)

    """

    __slots__ = (
        "_subjects",
        "_objects",
        "_predicates",
        "_nvs_store",
        "_d",
        "_Vocabulary2Parameter",
        "sub_id",
        "sub_parameter",
        "obj_id",
        "obj_parameter",
        "nvs",
    )

    def __init__(self, sub: str, obj: str, **kwargs):
        # Get an NVS store to retrieve data:
        self._nvs_store: NVS = NVS(nvs=kwargs.get("nvs", OPTIONS["nvs"]))

        # Validate subject and object:
        self._Vocabulary2Parameter: dict[str, str] = Asset.load("vocabulary:mapping")[
            "data"
        ]["Vocabulary2Parameter"]

        if sub in self._Vocabulary2Parameter.keys():
            self.sub_id: str = sub
            self.sub_parameter: str = self._Vocabulary2Parameter[sub]
        elif sub in self._Vocabulary2Parameter.values():
            self.sub_parameter: str = sub
            self.sub_id: str = [
                k for k, v in self._Vocabulary2Parameter.items() if v == sub
            ][0]
        else:
            raise ValueError(
                f"Unknown subject Reference Table '{sub}'. Possible values are: \nIDs like: {ppliststr([k for k in self._Vocabulary2Parameter], last='or')}\nNames like: {ppliststr([k for k in self._Vocabulary2Parameter.values()], last='or')}"
            )

        if obj in self._Vocabulary2Parameter.keys():
            self.obj_id: str = obj
            self.obj_parameter: str = self._Vocabulary2Parameter[obj]
        elif obj in self._Vocabulary2Parameter.values():
            self.obj_parameter: str = obj
            self.obj_id: str = [
                k for k, v in self._Vocabulary2Parameter.items() if v == obj
            ][0]
        else:
            raise ValueError(
                f"Unknown object Reference Table '{obj}'. Possible values are: \nIDs like: {ppliststr([k for k in self._Vocabulary2Parameter], last='or')}\nNames like: {ppliststr([k for k in self._Vocabulary2Parameter.values()], last='or')}"
            )

        if (self.sub_id, self.obj_id) not in valid_relations:
            warnings.warn(
                f"This mapping '{(self.sub_id, self.obj_id)}'is not known to the AVTT ! Known mappings are {valid_relations}"
            )

        # Retrieve NVS raw data
        # We use a deepcopy because we will modify the nvs raw data with complementary data
        self.nvs: dict[str, Any] = deepcopy(
            self._nvs_store.load_mapping(self.sub_id, self.obj_id)
        )

        # Internal placeholders:
        self._subjects: list[str] | None = None
        self._objects: list[str] | None = None
        self._d: dict[str, pd.DataFrame] | None = {}

    def __repr__(self):
        summary = [
            f"<argo.reference.mapping> subject('{self.sub_id}'/'{self.sub_parameter}') vs object('{self.obj_id}'/'{self.obj_parameter}')"
        ]
        summary.append(f"{len(self)} relationships in this mapping")
        return "\n".join(summary)

    @property
    def subjects(self):
        if self._subjects is None:
            self._subjects = np.unique([
                id2concept(binding["subj"]["value"])
                for binding in self.nvs["results"]["bindings"]
            ]).tolist()
            self._subjects.sort()
        return self._subjects

    @property
    def objects(self):
        if self._objects is None:
            self._objects = np.unique([
                id2concept(binding["obj"]["value"])
                for binding in self.nvs["results"]["bindings"]
            ]).tolist()
            self._objects.sort()
        return self._objects

    @property
    def predicates(self):
        if self._predicates is None:
            self._predicates = np.unique([
                predicate(binding["pred"]["value"])
                for binding in self.nvs["results"]["bindings"]
            ]).tolist()
            self._predicates.sort()
        return self._predicates

    def __len__(self):
        return len(self.nvs["results"]["bindings"])

    def __iter__(self):
        for sub in self.subjects:
            results = {'subject': sub, 'predicate':self[sub]}
            yield results

    def __contains__(self, item):
        return item in self.subjects or item in self.objects

    def __getitem__(self, key: str):
        ref_value: str | None = None
        if key in self.subjects:
            ref_value = key
        if ref_value is not None:
            if self._d.get(ref_value, None) is None:
                data = [
                        b
                        for b in self.nvs["results"]["bindings"]
                        if id2concept(b["subj"]["value"]) == key
                    ]
                results = {}
                for b in data:
                    subj, pred, obj = id2concept(b["subj"]["value"]), predicate(b["pred"]["value"]), id2concept(
                        b["obj"]["value"])
                    if pred in results:
                        results[pred].append(obj)
                    else:
                        results[pred] = [obj]
                for v in results.values():
                    v.sort()
                self._d[ref_value] = results
            return self._d[ref_value]
        raise ValueError(f"Invalid subject mapping value '{key}'")

    def to_dataframe(self, raw:bool = False) -> pd.DataFrame:
        """Return mapping as a :class:`pd.DataFrame`"""
        df = None
        if len(self.nvs["results"]["bindings"]) > 0:
            df = bindings2df(self.nvs["results"]["bindings"])
            if raw:
                df = df.drop(['subject', 'object'], axis=1)
                df = df.rename({'subject_uri': 'subject', 'object_uri': 'object'}, axis=1)
                return df[['subject', 'predicate', 'object']]
            else:
                df = df.drop(['subject_uri', 'object_uri'], axis=1)
                df['predicate'] = df['predicate'].map(lambda x: x.split(":")[-1]) # Remove NVS jargon ('skos:', or 'owl:')
        return df
