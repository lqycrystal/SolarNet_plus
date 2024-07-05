import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from definitions import epsg_metric_germany
from electricity_generation import pv_electricity_generation
from masks_to_vector import get_vector_labels, segment_simplify_and_add_azimuth
from module_placement import module_placement, create_pv_modules_gdf
from spatial_operations import get_image_gdf_in_directory



# --------------------------------------------------------------------------- #
# Set parameters and paths
# --------------------------------------------------------------------------- #
dir_roof_segment_masks ='./gt_test_roof_segment/' # the path of input roof orientation map
dir_roof_superstructure_masks = './gt_test_superstructure/' # the path of input roof superstructure map, instead of detailed 9 classes, here we use the map including only 2 classes: background and superstructure
dir_geotifs = './img_tif/' # the path of geotiff corresponding to input roof orientation map
dir_pvgis_cache = './pvgis_cache/' # the path to store intermediate PVGIS files
dir_results = './outputs/'# the path of output files
if not os.path.isdir(dir_pvgis_cache):
 os.makedirs(dir_pvgis_cache)
if not os.path.isdir(dir_results):
 os.makedirs(dir_results)

# todo: set if results should be visualized for debugging
visualize = False
# todo: code assumes background is 0, if BG is the largest class, set to False
bg_is_0 = True

# todo: check class definition (without background)
segment_classes = ['flat', 'W', 'S', 'E', 'N']
superstructure_classes = ['unknown', 'pvmodule']

# todo: adapt threshold of minimum size of geometries
a_min_segments = 2  # in square meters
a_min_superstructures = 0.5  # in square meters

# todo: set PV module specs
pv_module_peak_power = 400  # in Wp
pv_module_height = 1.7  # in m
pv_module_width = 1  # in m

# todo: set default slope
default_slope = 30  # in degrees

# --------------------------------------------------------------------------- #
# Extract the bounding boxes of the masks from geotifs
# --------------------------------------------------------------------------- #
gdf_images = get_image_gdf_in_directory(dir_geotifs)
gdf_images = gdf_images.to_crs(epsg=epsg_metric_germany)
# --------------------------------------------------------------------------- #
# Run the calculation for each mask in directory
# --------------------------------------------------------------------------- #

mask_filenames = os.listdir(dir_roof_segment_masks)
gdf_segments_all = gpd.GeoDataFrame()
gdf_superstructures_all = gpd.GeoDataFrame()
gdf_modules_all = gpd.GeoDataFrame()
for mask_filename in mask_filenames:
    # --------------------------------------------------------------------------- #
    # Extract information from mask to vector
    # --------------------------------------------------------------------------- #
    # mask to vector segments
    file_path = os.path.join(dir_roof_segment_masks, mask_filename)
    mask_id = mask_filename[:-4]  # mask id is filename without file ending
    gdf_segments = get_vector_labels(
        file_path, mask_id, gdf_images, segment_classes, a_min_segments, bg_is_0=bg_is_0)

    # simplify roof segment polygon and calculate azimuth
    gdf_segments, _, _ = segment_simplify_and_add_azimuth(gdf_segments)
    gdf_segments["slopes"] = [default_slope if np.isnan(i["azimuth"]) == False else 0 for i in gdf_segments.iloc]

    # mask to vector segments
    file_path = os.path.join(dir_roof_superstructure_masks, mask_filename)
    gdf_superstructures = get_vector_labels(
        file_path, mask_id, gdf_images, superstructure_classes, a_min_superstructures, bg_is_0=bg_is_0)
    # --------------------------------------------------------------------------- #
    # Conduct module placement on roof segments considering roof superstructures
    # --------------------------------------------------------------------------- #
    alignment_list, gdf_modules_vertical, gdf_modules_horizontal, azimuth_list = module_placement(
        gdf_segments,
        gdf_segments["azimuth"],
        gdf_segments["slopes"],
        gdf_superstructures,
        pv_module_height,
        pv_module_width
    )
    # create gdf with module geometries
    gdf_modules = create_pv_modules_gdf(alignment_list, gdf_modules_vertical, gdf_modules_horizontal, azimuth_list)
    gdf_modules = gdf_modules.to_crs(epsg_metric_germany)

    # add number of modules and the peak power in kW per segment
    gdf_segments["pv_modules_per_segment"] = [len(mp.geometry.geoms) for mp in gdf_modules.iloc]
    gdf_segments["pv_peak_power_per_segment"] = [len(mp.geometry.geoms) * pv_module_peak_power / 1000 for mp in gdf_modules.iloc]
    gdf_segments["azimuth_incl_flat"] = azimuth_list

    # --------------------------------------------------------------------------- #
    # Calculate the electricity generation of each roof segment
    # --------------------------------------------------------------------------- #
    gs_location = gpd.GeoSeries(gdf_segments.unary_union.centroid)
    gs_location.crs = gdf_images.crs
    electricity_generations = pv_electricity_generation(
        location=gs_location,
        azimuths=gdf_segments["azimuth_incl_flat"],
        slopes=gdf_segments["slopes"],
        peak_powers=gdf_segments["pv_peak_power_per_segment"],
        dir_pvgis_cache=dir_pvgis_cache
    )
    gdf_segments["electricity_generations"] = [np.sum(e) for e in electricity_generations]

    # --------------------------------------------------------------------------- #
    # add results to overall result GeoDataframes
    # --------------------------------------------------------------------------- #
    gdf_segments_all = pd.concat([gdf_segments_all, gdf_segments])
    gdf_superstructures_all = pd.concat([gdf_superstructures_all, gdf_superstructures])
    gdf_modules_all = pd.concat([gdf_modules_all, gdf_modules])

    # --------------------------------------------------------------------------- #
    # Visualize results for debugging purposes
    # --------------------------------------------------------------------------- #
    if visualize:
        fig, ax = plt.subplots()
        gdf_segments.plot(ax=ax, color="blue")
        gdf_superstructures.plot(ax=ax, color="red")
        gdf_modules.plot(ax=ax, color="green")

# --------------------------------------------------------------------------- #
# save results
# --------------------------------------------------------------------------- #

gdf_segments_all.to_csv(os.path.join(dir_results, "gdf_segments_2_gt.csv")) # This is csv format of PV generation of each roof segment

gdf_modules_all.to_file(os.path.join(dir_results, "gdf_modules_2_gt.json"), driver="GeoJSON") # This is geojson format of potential PV module can be installed in the future

