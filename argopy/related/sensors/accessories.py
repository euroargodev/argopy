from typing import Literal
import pandas as pd

from ...utils import NVSrow


# Define some options expected values as tuples
# (for argument validation)
SearchOutput = ("wmo", "sn", "wmo_sn", "df")
Error = ("raise", "ignore", "silent")
Ds = ("core", "deep", "bgc")

# Define Literal types using tuples
# (for typing)
SearchOutputOptions = Literal[*SearchOutput]
ErrorOptions = Literal[*Error]
DsOptions = Literal[*Ds]


class SensorType(NVSrow):
    """One single sensor type data from a R25-"Argo sensor types" row

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoNVSReferenceTables

        sensor_type = 'CTD'

        df = ArgoNVSReferenceTables().tbl(25)
        df_match = df[df["altLabel"].apply(lambda x: x == sensor_type)].iloc[0]

        st = SensorType.from_series(df_match)

        st.name
        st.long_name
        st.definition
        st.deprecated
        st.uri

    """

    reftable = "R25"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorType":
        """Create a :class:`SensorType` from a R25-"Argo sensor models" row"""
        return SensorType(obj)


class SensorModel(NVSrow):
    """One single sensor model data from a R27-"Argo sensor models" row

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoNVSReferenceTables

        sensor_model = 'AANDERAA_OPTODE_4330F'

        df = ArgoNVSReferenceTables().tbl(27)
        df_match = df[df["altLabel"].apply(lambda x: x == sensor_model)].iloc[0]

        sm = SensorModel.from_series(df_match)

        sm.name
        sm.long_name
        sm.definition
        sm.deprecated
        sm.uri
    """

    reftable = "R27"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorModel":
        """Create a :class:`SensorModel` from a R27-"Argo sensor models" row"""
        return SensorModel(obj)

    def __contains__(self, string) -> bool:
        return (
            string.lower() in self.name.lower()
            or string.lower() in self.long_name.lower()
        )
