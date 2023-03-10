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

log = logging.getLogger("argopy.tests")

# @pytest.fixture(scope="module", autouse=True)
@pytest.fixture(scope="class")
def mocked_erddap():
    uri = 'https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?data_mode,latitude,longitude,position_qc,time,time_qc,direction,platform_number,cycle_number,config_mission_number,vertical_sampling_scheme,pres,temp,psal,pres_qc,temp_qc,psal_qc,pres_adjusted,temp_adjusted,psal_adjusted,pres_adjusted_qc,temp_adjusted_qc,psal_adjusted_qc,pres_adjusted_error,temp_adjusted_error,psal_adjusted_error&platform_number=~%221901393%22&distinct()&orderBy(%22time,pres%22)'
    with aioresponses() as httpserver:
        with open('test_data/ArgoFloats_float_1901393.nc', mode='rb') as file:
            data = file.read()
        httpserver.get(uri, body=data)

        log.info('Mocked IFREMER erddap ready.')
        yield httpserver

import aiohttp
import asyncio

if __name__ == '__main__':
    URI = ['https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?data_mode,latitude,longitude,position_qc,time,time_qc,direction,platform_number,cycle_number,config_mission_number,vertical_sampling_scheme,pres,temp,psal,pres_qc,temp_qc,psal_qc,pres_adjusted,temp_adjusted,psal_adjusted,pres_adjusted_qc,temp_adjusted_qc,psal_adjusted_qc,pres_adjusted_error,temp_adjusted_error,psal_adjusted_error&platform_number=~%221901393%22&distinct()&orderBy(%22time,pres%22)']

    async def main():
        async with aiohttp.ClientSession() as session:
            async with session.get(URI[0]) as resp:
                print(resp.status)
                data = await resp.read()
                with open('../test_data/toto.nc', mode='wb') as file:
                    file.write(data)
                # print(await resp.text())

    asyncio.run(main())

