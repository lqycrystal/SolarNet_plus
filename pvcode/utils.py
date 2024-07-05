__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import datetime as dt
import numpy as np
import geopandas as gpd

def get_progress_string(progress):
    number_of_dashes = 50
    progress_string = '| >'
    progress_string += int(round(progress * number_of_dashes, 0)) * '>' + \
        int(round((1 - progress) * number_of_dashes, 0)) * '-'
    progress_string += ' |'
    return progress_string


def determine_season(date):
    '''
    This function determines the seasons based on a datetime date.
    The definition is based on the VDEW definition:
    https://www.bdew.de/media/documents/1999_Repraesentative-VDEW-Lastprofile.pdf (page 30)

    - Winter: 01.11 - 20.03
    - Summer: 15.05 - 14.09
    - Transition: others: 21.03 - 14.05 and 15.09 - 31.10

    :parameter:
        date: datetime.date
    :return: season_int: int  - 0: winter, 1: transition, 2: summer
    '''

    assert isinstance(date, dt.date), print("date must be datetime.date")

    # check if winter
    if date >= dt.date(date.year, 11, 1) or date <= dt.date(date.year, 3, 20):
        season_int = 0
    # check if summer
    elif dt.date(date.year, 5, 15) <= date <= dt.date(date.year, 9, 14):
        season_int = 1
    # else: transition
    else:
        season_int = 2
    return season_int


def quarter_hourly_to_hourly(quarter_hourly_array):
    '''
    This function takes an np.array of 96 values (quarter hourly profile) and returns an np.array of 24 (hourly) values.
    This is done by averaging the values of each four entries (hours).
    :parameter:
        quarter_hourly_array
    :return:
        hourly_array: np.array
    '''

    # make sure input is numpy array or can be transformed to one
    assert isinstance(quarter_hourly_array, np.ndarray), \
        print("Input as numpy array expected, but got: " + str(type(quarter_hourly_array)))

    # Reshape quarter-hourly array into a 2D array with 4 columns and a row for every hour
    quarter_hourly_reshaped = quarter_hourly_array.reshape(-1, 4)
    # Calculate the mean of each hour
    hourly_array = np.mean(quarter_hourly_reshaped, axis=1)

    return hourly_array


def assign_osm_building_id(gdf, gdf_buildings):
    gdf_new = gpd.sjoin(gdf, gdf_buildings, how='left', predicate='intersects')
    gdf_new = gdf_new[gdf_new["index_right"].isna() == False]
    # drop non-required colums
    drop_cols = [col for col in gdf_new.columns if (col not in gdf.columns and col != "building_id")]
    gdf_new = gdf_new.drop(drop_cols, axis=1)
    return gdf_new

