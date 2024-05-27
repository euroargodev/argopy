import pytest
import types
import numpy as np
import pandas as pd

from argopy.errors import InvalidFetcherAccessPoint
from argopy.utils.chunking import Chunker
from argopy.utils.checkers import is_box


class Test_Chunker:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.WMO = [
            6902766,
            6902772,
            6902914,
            6902746,
            6902916,
            6902915,
            6902757,
            6902771,
        ]
        self.BOX3d = [0, 20, 40, 60, 0, 1000]
        self.BOX4d = [0, 20, 40, 60, 0, 1000, "2001-01", "2001-6"]

    def test_InvalidFetcherAccessPoint(self):
        with pytest.raises(InvalidFetcherAccessPoint):
            Chunker({"invalid": self.WMO})

    def test_invalid_chunks(self):
        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunks='toto')

    def test_invalid_chunksize(self):
        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunksize='toto')

    def test_chunk_wmo(self):
        C = Chunker({"wmo": self.WMO})
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )

        C = Chunker({"wmo": self.WMO}, chunks="auto")
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )

        C = Chunker({"wmo": self.WMO}, chunks={"wmo": 1})
        assert all(
            [all(isinstance(x, int) for x in chunk) for chunk in C.fit_transform()]
        )
        assert len(C.fit_transform()) == 1

        with pytest.raises(ValueError):
            Chunker({"wmo": self.WMO}, chunks=["wmo", 1])

        C = Chunker({"wmo": self.WMO})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )

    def test_chunk_box3d(self):
        C = Chunker({"box": self.BOX3d})
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX3d}, chunks="auto")
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 12, "lat": 1, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lat": 1, "dpt": 1}, chunksize={"lon": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][1] - chunks[0][0] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 1, "lat": 12, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lon": 1, "dpt": 1}, chunksize={"lat": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][3] - chunks[0][2] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 1, "lat": 1, "dpt": 12})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 12

        C = Chunker(
            {"box": self.BOX3d}, chunks={"lon": 1, "lat": 1}, chunksize={"dpt": 10}
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][5] - chunks[0][4] == 10

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 4, "lat": 2, "dpt": 1})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2 * 4

        C = Chunker({"box": self.BOX3d}, chunks={"lon": 2, "lat": 3, "dpt": 4})
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2 * 3 * 4

        with pytest.raises(ValueError):
            Chunker({"box": self.BOX3d}, chunks=["lon", 1])

        C = Chunker({"box": self.BOX3d})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )

    def test_chunk_box4d(self):
        C = Chunker({"box": self.BOX4d})
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker({"box": self.BOX4d}, chunks="auto")
        assert all([is_box(chunk) for chunk in C.fit_transform()])

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 2, "lat": 1, "dpt": 1, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lat": 1, "dpt": 1, "time": 1},
            chunksize={"lon": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][1] - chunks[0][0] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 2, "dpt": 1, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "dpt": 1, "time": 1},
            chunksize={"lat": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][3] - chunks[0][2] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 1, "dpt": 2, "time": 1}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "lat": 1, "time": 1},
            chunksize={"dpt": 10},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert chunks[0][5] - chunks[0][4] == 10

        C = Chunker(
            {"box": self.BOX4d}, chunks={"lon": 1, "lat": 1, "dpt": 1, "time": 2}
        )
        assert all([is_box(chunk) for chunk in C.fit_transform()])
        assert len(C.fit_transform()) == 2

        C = Chunker(
            {"box": self.BOX4d},
            chunks={"lon": 1, "lat": 1, "dpt": 1},
            chunksize={"time": 5},
        )
        chunks = C.fit_transform()
        assert all([is_box(chunk) for chunk in chunks])
        assert np.timedelta64(
            pd.to_datetime(chunks[0][7]) - pd.to_datetime(chunks[0][6]), "D"
        ) <= np.timedelta64(5, "D")

        with pytest.raises(ValueError):
            Chunker({"box": self.BOX4d}, chunks=["lon", 1])

        C = Chunker({"box": self.BOX4d})
        assert isinstance(C.this_chunker, types.FunctionType) or isinstance(
            C.this_chunker, types.MethodType
        )

