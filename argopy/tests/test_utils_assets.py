import pandas as pd
import pytest
from argopy.utils.assets import Asset


class Test_Asset():
    assets = ['gdac_servers.json', 'data_types', 'schema:argo.sensor.schema.json', 'schema:argo.float.schema']
    assets_id = [f"{a}" for a in assets]
    @pytest.mark.parametrize("asset", assets, indirect=False, ids=assets_id)
    def test_load_json(self, asset):
        data = Asset.load(asset)
        assert isinstance(data, dict)

    assets = ['canyon-b:wgts_AT.txt']
    assets_id = [f"{a}" for a in assets]
    @pytest.mark.parametrize("asset", assets, indirect=False, ids=assets_id)
    def test_load_csv(self, asset):
        data = Asset.load(asset)
        assert isinstance(data, pd.DataFrame)

        data = Asset.load(asset, header=None, sep="\t")
        assert isinstance(data, pd.DataFrame)
