import os
import cv2
import sys

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString

dir_base = os.path.abspath('..')
if dir_base not in sys.path:
    sys.path.append(dir_base)

from utils import get_progress_string, assign_osm_building_id


from spatial_operations import raster_to_vector, simplify_polygon, find_longest_line, \
    calculate_rotation_angle, select_azimuth, opposite_angle, filter_out_overlapping_polygons



def get_vector_labels(file_path, mask_filename, gdf_images, label_classes, area_treshold, bg_is_0=False):
    '''
    :param file_path: path to file
    :param mask_filename: name of mask, same as in gdf_images
    :param gdf_images: GeoDataFrame with coordinates of masks' bounding box
    :param label_classes: label definition of masks
    :param area_treshold: minimum area of geometries extracted from mask
    :return: GeoDataframe with geometries of extracted mask geometries. same CRS as gdf_images
    '''
    # load mask
    mask = cv2.imread(file_path, 0)
    if mask_filename in list(gdf_images.id):
        # select gdf_image for the bounding box
        gdf_image = gdf_images[gdf_images.id == mask_filename]
        image_bbox = gdf_image.geometry.iloc[0]
        # run transformation
        gdf_labels = raster_to_vector(mask, mask_filename, image_bbox, label_classes, bg_is_0=bg_is_0)

    # clean up dataframe
    gdf_labels = gdf_labels[gdf_labels.geometry.area > area_treshold]
    gdf_labels = gdf_labels.reset_index()
    gdf_labels.crs = gdf_images.crs
    return gdf_labels


def segment_simplify_and_add_azimuth(gdf_segments, visualize=False):
    gdf_original_geom = gdf_segments.copy()

    # find longest line and get azimuth of longest line
    azimuths = []
    simplified_segments = []
    longest_lines = []
    for i, gdf_segment in enumerate(gdf_segments.iloc):
        # simplify roof segment geometry (to be able to align PV modules to roof line)
        segment = gdf_segment.geometry
        simplified_segment = simplify_polygon(segment, tolerance=np.sqrt(segment.area) / 20)
        simplified_segments.append(simplified_segment)

        if gdf_segment["label"] == "flat":
            azimuth = np.nan
        else:
            # calculate rotation from longest line
            longest_line = find_longest_line(simplified_segment)
            longest_lines.append(LineString(longest_line))
            rotation_angle = calculate_rotation_angle(longest_line)
            # there are 2 possible azimuth angles. select azimuth based on class
            azimuth_options = (-rotation_angle, opposite_angle(-rotation_angle))
            azimuth = select_azimuth(gdf_segment["label"], azimuth_options[0], azimuth_options[1])

        azimuths.append(azimuth)

    gdf_segments.loc[:, "azimuth"] = azimuths
    gdf_segments.loc[:, "geometry"] = simplified_segments

    gdf_longest_lines = gpd.GeoDataFrame({"geometry": longest_lines})
    gdf_longest_lines.crs = gdf_segments.crs

    # for debugging: visualization
    if visualize:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='datalim')

        # Draw polygons and longest line
        gdf_segments.plot(ax=ax, color="None", edgecolor="green")
        gdf_longest_lines.plot(ax=ax, edgecolor="blue")

        # Draw arrow pointing in the specified direction
        for gdf_segment in gdf_segments.iloc:
            x = gdf_segment.geometry.centroid.x
            y = gdf_segment.geometry.centroid.y
            if np.isnan(gdf_segment["azimuth"]):
                ax.plot(x, y, "-ro")
            else:
                delta_x = 3 * np.cos(np.radians(-gdf_segment["azimuth"]-90))
                delta_y = 3 * np.sin(np.radians(-gdf_segment["azimuth"]-90))
                ax.arrow(x, y, delta_x, delta_y, head_width=0.2, head_length=0.4, color="purple")

            ax.text(x + 0.1, y + 0.1, gdf_segment["label"], fontsize=11, color='purple')

    return gdf_segments, gdf_original_geom, gdf_longest_lines


def remove_overlapping_segments(gdf_segments, gdf_buildings, percent_overlap=0.1):
    gdf_segments = assign_osm_building_id(gdf_segments, gdf_buildings)

    # loop through geoms of each image
    gdf_keep = gpd.GeoDataFrame()
    gdf_drop = gpd.GeoDataFrame()
    for i, gdf_buildings in enumerate(gdf_buildings.iloc):
        progress_string = get_progress_string(round(i / len(gdf_buildings), 2))
        print(f'Removing overlapping geoms for building {str(i)} {progress_string}')

        gdf_iter = gdf_segments[gdf_segments['building_id'] == gdf_buildings['building_id']]

        gdf_iter = gdf_iter.reset_index()
        gdf_keep_iter, gdf_drop_iter = filter_out_overlapping_polygons(
            gdf_iter, percent_overlap=percent_overlap, keep_mode="larger"
        )
        # append results
        gdf_keep = pd.concat([gdf_keep, gdf_keep_iter])
        gdf_drop = pd.concat([gdf_drop, gdf_drop_iter])
    return gdf_keep, gdf_drop

