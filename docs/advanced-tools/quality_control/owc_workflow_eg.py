import os
import shutil
from pathlib import Path

import pyowc as owc
from argopy import DataFetcher

# Define float to calibrate:
FLOAT_NAME = "6903010"

# Set-up where to save OWC analysis results:
results_folder = "./analysis/%s" % FLOAT_NAME
Path(results_folder).mkdir(parents=True, exist_ok=True)
shutil.rmtree(results_folder)  # Clean up folder content
Path(os.path.sep.join([results_folder, "float_source"])).mkdir(
    parents=True, exist_ok=True
)
Path(os.path.sep.join([results_folder, "float_calib"])).mkdir(
    parents=True, exist_ok=True
)
Path(os.path.sep.join([results_folder, "float_mapped"])).mkdir(
    parents=True, exist_ok=True
)
Path(os.path.sep.join([results_folder, "float_plots"])).mkdir(
    parents=True, exist_ok=True
)

# fetch the default configuration and parameters
USER_CONFIG = owc.configuration.load()

# Fix paths to run at Ifremer:
for k in USER_CONFIG:
    if "FLOAT" in k and "data/" in USER_CONFIG[k][0:5]:
        USER_CONFIG[k] = os.path.abspath(USER_CONFIG[k].replace("data", results_folder))

USER_CONFIG["CONFIG_DIRECTORY"] = os.path.abspath("../data/constants")
USER_CONFIG["HISTORICAL_DIRECTORY"] = os.path.abspath(
    "/Volumes/OWC/CLIMATOLOGY/"
)  # where to find ARGO_for_DMQC_2020V03 and CTD_for_DMQC_2021V01 folders
USER_CONFIG["HISTORICAL_ARGO_PREFIX"] = "ARGO_for_DMQC_2020V03/argo_"
USER_CONFIG["HISTORICAL_CTD_PREFIX"] = "CTD_for_DMQC_2021V01/ctd_"
print(owc.configuration.print_cfg(USER_CONFIG))

# Create float source data with argopy:
fetcher_for_real = DataFetcher(src="gdac", cache=True, mode="expert").float(FLOAT_NAME)
fetcher_sample = DataFetcher(src="gdac", cache=True, mode="expert").profile(
    FLOAT_NAME, [1, 2]
)  # To reduce execution time for demo
ds = fetcher_sample.load().data
ds.argo.create_float_source(path=USER_CONFIG["FLOAT_SOURCE_DIRECTORY"], force="default")

# Prepare data for calibration: map salinity on theta levels
owc.calibration.update_salinity_mapping("", USER_CONFIG, FLOAT_NAME)

# Set the calseries parameters for analysis and line fitting
owc.configuration.set_calseries("", FLOAT_NAME, USER_CONFIG)

# Calculate the fit of each break and calibrate salinities
owc.calibration.calc_piecewisefit("", FLOAT_NAME, USER_CONFIG)

# Results figures
owc.plot.dashboard("", FLOAT_NAME, USER_CONFIG)
