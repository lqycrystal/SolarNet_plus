__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, Institute of Automotive Technology TUM"
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import requests
import os
import json
import pandas as pd
import numpy as np

from definitions import pv_system_loss, default_year

def pv_electricity_generation_per_kWp_hourly(longitude, latitude, azimuth, slope, peak_power=1, loss=14, year=2014, dir_pvgis_cache=""):
    """
    This function calls the PVGIS API to determine the hourly energy output of a roof segment for a
    year of weather data, default is 2014, a year with medium yearly solar radiation in bavaria.
    https://ec.europa.eu/jrc/en/PVGIS/docs/noninteractive

    :param longitude: float
        lonigtude of roof segment's location
    :param latitude: float
        latitude of roof segment's location
    :param azimuth: float
        azimuth of roof segment, between -180 (N) and 180 (N), with 0 being South. see PVGIS for definition
    :param slope: float
        roof tilt/slope angle between 0 and 90
    :param peak_power: float
        peak power of the PV system, default: 1 kWp
    :param loss: int
        loss of PV system in percent. default 14
    :param year: int
        year of radiation data. default is 2014

    :return: PV_E_gen_hourly pandas Dataframe
        pandas Dataframe with time and power as columns. Contains the hourly values of one year.
        power is in W
    """

    # round values integer to avoid weird errors
    loss = int(round(loss, 0))
    azimuth = int(round(azimuth, 0))
    slope = int(round(slope, 0))
    peak_power = round(peak_power, 2)

    # create PVGIS call string
    PVGIS_config = (f"lat={latitude: .8f}&lon={longitude: .8f}&pvcalculation=1&peakpower={peak_power}&angle={slope}"
                    f"&aspect={azimuth}&startyear={year}&endyear={year}&mountingplace=building&outputformat=json"
                    f"&loss={loss}")

    # check if result of a configuration has already been requested and saved before:
    pvgis_cache_filepath = os.path.join(dir_pvgis_cache, PVGIS_config + str(".json"))
    if os.path.isfile(pvgis_cache_filepath):
        with open(pvgis_cache_filepath, 'rb') as file:
            pvgis_result = json.load(file)

    # if not: call PVGIS API and save result as json
    else:
        # build query
        query = ("https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?{}".format(PVGIS_config))
        # get result
        pvgis_result = requests.get(query)
        # extract energy generation from result json
        pvgis_result = pvgis_result.json()
        with open(pvgis_cache_filepath, 'w') as file:
            json.dump(pvgis_result, file)

    df = pd.DataFrame(pvgis_result['outputs']['hourly'])
    PV_E_gen_hourly = pd.DataFrame(columns=['time', 'power'])
    PV_E_gen_hourly.time = pd.to_datetime(df.time, format='%Y%m%d:%H%M')
    PV_E_gen_hourly.power = df.P

    return PV_E_gen_hourly


def pv_electricity_generation(location, azimuths, slopes, peak_powers, dir_pvgis_cache):
    # make sure location is in EPSG 4326
    if location.crs != 4326:
        location = location.to_crs(4326)
    # initialize result list
    E_gen_hourly_list = []
    # request the electricity generation per kWp for each roof segment (azimuth)
    for i, azimuth in enumerate(azimuths):
        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(
            location.geometry.x.iloc[0],
            location.geometry.y.iloc[0],
            azimuth,
            slopes[i],
            peak_power=1,
            loss=pv_system_loss*100,
            year=default_year,
            dir_pvgis_cache=dir_pvgis_cache
        )
        # scale generation per kWp to segment's kWp
        PV_E_gen_hourly_segment = PV_E_gen_hourly_kWp.power * peak_powers[i] / 1000
        # electricity generation profile in kW
        E_gen_hourly_list.append(PV_E_gen_hourly_segment)
    # if no electricity generation: fill e_gen with zeroes
    if len(E_gen_hourly_list) == 0:
        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(42, 11, 0, 30, 1, 14, 2014)
        E_gen_hourly_list.append(PV_E_gen_hourly_kWp.power * 0)
    return E_gen_hourly_list
