"""
Mock the Ifremer erddap

For this to work, we need to catch all the possible erddap URI sent from unit tests and return some json or netcdf data
we downloaded once before tests.

This requires to download data using a fully functional erddap server: this is done by this script when executed from 
the command line.

"""
import logging
import pytest
from aioresponses import aioresponses
import numpy as np
import sys
import os
import pickle
import xarray as xr
from urllib.parse import urlparse, parse_qs


log = logging.getLogger("argopy.tests.mocked_erddap")


# @pytest.fixture(scope="module", autouse=True)
@pytest.fixture(scope="module")
def mocked_erddap():
    # uri = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?data_mode,latitude,longitude,position_qc,time,time_qc,direction,platform_number,cycle_number,config_mission_number,vertical_sampling_scheme,pres,temp,psal,pres_qc,temp_qc,psal_qc,pres_adjusted,temp_adjusted,psal_adjusted,pres_adjusted_qc,temp_adjusted_qc,psal_adjusted_qc,pres_adjusted_error,temp_adjusted_error,psal_adjusted_error&platform_number=~%221901393%22&distinct()&orderBy(%22time,pres%22)"
    # with aioresponses() as httpserver:
    #     with open('test_data/ArgoFloats_float_1901393.nc', mode='rb') as file:
    #         data = file.read()
    #     httpserver.get(uri, body=data)
    #
    #     log.info('Mocked IFREMER erddap ready.')
    #     yield httpserver

    DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)).replace("helpers", "test_data")
    # log.info("Tests DATA_FOLDER: %s" % DATA_FOLDER)

    with open(os.path.join(DATA_FOLDER, "erddap_file_index.pkl"), "rb") as f:
        URI = pickle.load(f)
    # log.debug(URI)

    with aioresponses() as httpserver:
        for ressource in URI:
            test_data_file = os.path.join(DATA_FOLDER, "%s.%s" % (ressource['sha'], ressource['ext']))
            with open(test_data_file, mode='rb') as file:
                data = file.read()
            httpserver.get(ressource['uri'], body=data)#, headers={'Content-Type': 'application/x-netcdf'})

            # assert can_be_xr_opened(True, test_data_file)
            # log.debug("%s -> %s" % (parse_qs(ressource['uri']), test_data_file))

            # log.debug(xr.open_dataset(test_data_file, engine='netcdf4'))

        log.info("Mocked IFREMER erddap ready with %i URLs registered." % len(URI))
        yield httpserver


def can_be_xr_opened(src, file):
    try:
        xr.open_dataset(file)
        return src
    except:
        print("This source can't be opened with xarray: %s" % src)
        return src


if __name__ == '__main__':
    import aiohttp
    import asyncio
    import aiofiles

    DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)).replace("helpers", "test_data")
    print("Tests DATA_FOLDER: %s" % DATA_FOLDER)

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from argopy import DataFetcher

    async def fetch_download_links(session: aiohttp.ClientSession):
        # Get the list of URI to download.
        # This should correspond to all the possible erddap requests made during CI tests.
        # And because of the erddap fetcher N_POINT attribute, we also need to fetch ".ncHeader" on top of ".nc" files
        requests_phy = {
            "float": [[1901393], [1901393, 6902746]],
            "profile": [[6902746, 34], [6902746, np.arange(12, 13)], [6902746, [1, 12]]],
            "region": [
                [-20, -16., 0, 1, 0, 100.],
                # [-20, -16., 0, 1, 0, 100., "1997-07-01", "1997-09-01"]
                [-20, -16., 0, 1, 0, 100., "2004-01-01", "2004-01-31"]
            ],
        }
        requests_bgc = {
            "float": [[5903248], [7900596, 2902264]],
            "profile": [[5903248, 34], [5903248, np.arange(12, 14)], [5903248, [1, 12]]],
            "region": [
                [-70, -65, 35.0, 40.0, 0, 10.0],
                [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-1", "2012-12-31"],
            ],
        }
        requests_ref = {
            "region": [
                [-70, -65, 35.0, 40.0, 0, 10.0],
                [-70, -65, 35.0, 40.0, 0, 10.0, "2012-01-01", "2012-12-31"],
            ]
        }
        requests = {'phy': requests_phy, 'bgc': requests_bgc, 'ref': requests_ref}

        nc_file = lambda url, sha, iter: {'uri': url, 'ext': "nc", 'sha': "%s_%03.d" % (sha, iter)}
        ncHeader_file = lambda url, sha, iter: {'uri': url.replace(".nc", ".ncHeader"), 'ext': "ncHeader", 'sha': "%s_%03.d" % (sha, iter)}

        URI = []
        for ds in requests:
            fetcher = DataFetcher(src='erddap', ds=ds)
            for access_point in requests[ds]:
                if access_point == 'profile':
                    for cfg in requests[ds][access_point]:
                        f = fetcher.profile(*cfg)
                        uri = f.uri
                        for ii, url in enumerate(uri):
                            URI.append(nc_file(url, f.fetcher.sha, ii))
                            URI.append(ncHeader_file(url, f.fetcher.sha, ii))
                        # print(ds, access_point, cfg, f.fetcher.sha)
                if access_point == 'float':
                    for cfg in requests[ds][access_point]:
                        f = fetcher.float(cfg)
                        uri = f.uri
                        for ii, url in enumerate(uri):
                            URI.append(nc_file(url, f.fetcher.sha, ii))
                            URI.append(ncHeader_file(url, f.fetcher.sha, ii))
                        # print(ds, access_point, cfg, f.fetcher.sha)
                if access_point == 'region':
                    for cfg in requests[ds][access_point]:
                        f = fetcher.region(cfg)
                        uri = f.uri
                        for ii, url in enumerate(uri):
                            URI.append(nc_file(url, f.fetcher.sha, ii))
                            URI.append(ncHeader_file(url, f.fetcher.sha, ii))
                        # print(ds, access_point, cfg, f.fetcher.sha)

        return URI

    async def place_file(session: aiohttp.ClientSession, source: dict) -> None:
        # print(source['uri'])
        # r = await session.get(source['uri'], ssl=False)
        async with session.get(source['uri'], ssl=False) as r:
            if not r.content_type == 'application/x-netcdf':
                print("Unexpected content type (%s) with this request: %s" % (r.content_type, parse_qs(source['uri'])))

            test_data_file = os.path.join(DATA_FOLDER, "%s.%s" % (source['sha'], source['ext']))
            async with aiofiles.open(test_data_file, 'wb') as f:
                data = await r.content.read(n=-1)  # load all read bytes !
                await f.write(data)
                return can_be_xr_opened(source, test_data_file)

    async def main():
        async with aiohttp.ClientSession() as session:
            urls = await fetch_download_links(session)
            return await asyncio.gather(*[place_file(session, url) for url in urls])

    loop = asyncio.get_event_loop()
    URLS = loop.run_until_complete(main())
    # URLS = [url for url in URLS if url is not None]
    print("Saved %i urls" % len(URLS))

    # Save the URI list in a pickle to be loaded and used by the erddap mocker:
    with open(os.path.join(DATA_FOLDER, "erddap_file_index.pkl"), 'wb') as f:
        pickle.dump(URLS, f)

    # print(URLS)
