from typing import Literal, Any
import logging

from ....errors import OptionValueError, InvalidOption
from ....utils import ListStrProperty
from ..extensions import ArgoIndexSearchValidProto

log = logging.getLogger("argopy.stores.index.valid")


class ArgoIndexSearchValid(ArgoIndexSearchValidProto):
    """Extension providing valid values for search queries and a validator

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoIndex
        idx = ArgoIndex()

        # Get the list of possible values for a given index property
        idx.valid.institution_name

        # Validate one possible value:
        idx.valid('institution_name', 'france')  # True
        idx.valid('institution_name', 'toto')    # False
        idx.valid('institution_name', 'toto', errors='raise') # Raise OptionValueError
    """

    __slots__ = ['_institution_name', '_institution_code']

    def __call__(self, key: str, value: Any, errors: Literal["ignore", "raise"] = "ignore") -> bool | None:
        """
        Parameters
        ----------
        key: str
            The property to evaluate.
        value: Any
            The value to evaluate against the property valid values.
        errors: str, optional, default="raise"
            Define how to handle errors

        Returns
        -------
        bool

        Raises
        ------
        :class:`OptionValueError`, :class:`InvalidOption`
        """
        if getattr(self, key, None) is not None:
            this_prop = getattr(self, f"_{key}", None)
            values = this_prop.values
            if value not in this_prop:
                if errors == "raise":
                    raise OptionValueError(
                        f"'{value}' is not a valid value for '{key}'. Valid values are in: {values}"
                    )
                else:
                    log.info(f"Encountered '{value}' as an invalid value for '{key}'")
                    return False
            return True
        else:
            raise InvalidOption(f"'{key}' is not documented and cannot be evaluated.")

    @property
    def institution_name(self) -> list[str]:
        """List of valid institution names, according to reference table R4

        See Also
        --------
        :meth:`ArgoNVSReferenceTables().tbl('R4')`
        """
        self._institution_name = ListStrProperty([n for n in self._obj._r4.values()])
        return self._institution_name.values

    @property
    def institution_code(self) -> list[str]:
        """List of valid institution codes, according to reference table R4

        See Also
        --------
        :meth:`ArgoNVSReferenceTables().tbl('R4')`
        """
        self._institution_code = ListStrProperty([n for n in self._obj._r4.keys()])
        return self._institution_code.values
