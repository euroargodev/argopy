import os

import pandas as pd
import pytest
import io
import argopy
from argopy.utils.locals import modified_environ, Asset


@pytest.mark.parametrize("conda", [False, True],
                         indirect=False,
                         ids=["conda=%s" % str(p) for p in [False, True]])
def test_show_versions(conda):
    f = io.StringIO()
    argopy.show_versions(file=f, conda=conda)
    assert "SYSTEM" in f.getvalue()


def test_modified_environ():
    os.environ["DUMMY_ENV_ARGOPY"] = 'initial'
    with modified_environ(DUMMY_ENV_ARGOPY='toto'):
        assert os.environ['DUMMY_ENV_ARGOPY'] == 'toto'
    assert os.environ['DUMMY_ENV_ARGOPY'] == 'initial'
    os.environ.pop('DUMMY_ENV_ARGOPY')


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
