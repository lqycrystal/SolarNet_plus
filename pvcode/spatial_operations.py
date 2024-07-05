__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, "
__credits__ = ["Parts of code by Nils Kemmerzell"]
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import rasterio
import cv2
import os

import numpy as np
import geopandas as gpd
import shapely
from shapely import MultiPolygon, Polygon, MultiLineString, LineString
from shapely.geometry import Polygon, box

from utils import get_progress_string


def find_longest_line(polygon):
    """
    Finds longest line in roof segment to align solar panels.
    """
    ring_xy = polygon.exterior.xy
    x_coords = list(ring_xy[0])
    y_coords = list(ring_xy[1])

    longest_line_length = 0
    longest_line_coords = None

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        try:
            next_x = x_coords[i+1]
            next_y = y_coords[i+1]
        except:
            continue

        length = np.sqrt((next_x - x) ** 2 + (next_y - y) ** 2)
        if length > longest_line_length:
            longest_line_length = length
            longest_line_coords = [(x, y), (next_x, next_y)]

    return longest_line_coords


def azimuth_to_label_class(az, label_classes):
    label_classes = label_classes[:-1]
    if az is None or np.isnan(az):
        az_class = "flat"
    else:
        surplus_angle = 360 / len(label_classes) / 2
        az = az + 180 + surplus_angle
        if az > 360: az -= 360
        az_id = int(np.ceil(az / (360 / len(label_classes))) - 1)
        az_class = label_classes[az_id]
    return az_class


def label_class_to_azimuth(label_class):
    class_to_azimuth_map = {
        'N': 180,
        'NNE': -157.5,
        'NE': -135,
        'ENE': -112.5,
        'E': -90,
        'ESE': -67.5,
        'SE': -45,
        'SSE': -22.5,
        'S': 0,
        'SSW': 22.5,
        'SW': 45,
        'WSW': 67.5,
        'W': 90,
        'WNW': 112.5,
        'NW': 135,
        'NNW': 157.5,
        'flat': np.nan
    }
    azimuth = class_to_azimuth_map[label_class]
    return azimuth


def filter_out_overlapping_polygons(gdf, percent_overlap=0.5, keep_mode="index"):
    # make sure input is valid
    keep_options = ["index", "larger"]
    assert keep_mode in keep_options, print(f"Keep_mode invalid. Please select {[k for k in keep_options]}")

    # Create geoindex
    sindex = gdf.sindex

    # Find intersecting pairs (excluding self-intersecting pairs)
    intersecting_pairs = []
    for i, polygon in enumerate(gdf['geometry']):
        possible_matches_index = list(sindex.intersection(polygon.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        # Exclude self-intersecting pairs
        possible_matches = possible_matches[possible_matches.index != i]
        intersecting_pairs.extend(
            [(i, j) for j in possible_matches_index if polygon.overlaps(gdf['geometry'][j])])

    # Check whether polygons intersect with more than percent_overlap, e.g. 50%
    overlapping_polygons = set()
    for pair in intersecting_pairs:
        intersection_area = gdf['geometry'][pair[0]].intersection(gdf['geometry'][pair[1]]).area
        if intersection_area / min(gdf['geometry'][pair[0]].area,
                                   gdf['geometry'][pair[1]].area) > percent_overlap:
            overlapping_polygons.add(pair[0])
            overlapping_polygons.add(pair[1])

    # Keep polygons with no overlap and one of the overlapping polygons (with the smaller index)
    polygons_to_keep = set()
    for i in range(len(gdf)):
        if i not in overlapping_polygons:
            # Keep polygons with no overlap
            polygons_to_keep.add(i)

    for i, pair in enumerate(intersecting_pairs):
        progress_string = get_progress_string(round(i / len(intersecting_pairs), 2))
        print(f'Solving intersecting geom pairs {str(i)} {progress_string}')

        if pair[0] in overlapping_polygons and pair[1] in overlapping_polygons:
            if keep_mode == "index":
                # Keep only the one with the smaller index for overlapping pairs
                overlapping_polygons_iter = set(pair)
                for other_pair in intersecting_pairs:
                    if pair[0] in other_pair or pair[1] in other_pair:
                        overlapping_polygons_iter.update(other_pair)
                polygons_to_keep.add(min(overlapping_polygons_iter))
                # polygons_to_keep.add(min(pair))

            elif keep_mode == "larger":
                # Keep the largest polygon among all overlapping polygons
                overlapping_polygons_iter = set(pair)
                for other_pair in intersecting_pairs:
                    if pair[0] in other_pair or pair[1] in other_pair:
                        overlapping_polygons_iter.update(other_pair)

                largest_polygon_index = max(overlapping_polygons_iter, key=lambda x: gdf['geometry'][x].area)
                polygons_to_keep.add(largest_polygon_index)
                # larger_polygon_index = max(pair, key=lambda x: gdf['geometry'][x].area)
                # polygons_to_keep.add(larger_polygon_index)

    # Create a new GeoDataFrame with polygons to keep
    gdf_segments_keep = gdf[gdf.index.isin(polygons_to_keep)]
    gdf_segments_to_drop = gdf[gdf.index.isin(polygons_to_keep) == False]

    return gdf_segments_keep, gdf_segments_to_drop


def save_img_as_geotiff(image, boundary, crs, save_path):
    # Define the transformation parameters
    height, width, bands = image.shape

    bbox = boundary.bounds
    transform = rasterio.transform.from_origin(
        bbox[0], bbox[3], (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height)

    # Set up the metadata for the GeoTIFF file
    meta = {
        'driver': 'GTiff',
        'count': bands,  # Number of bands in the image, including alpha channel
        'width': width,
        'height': height,
        'crs': crs,
        'transform': transform,
        'dtype': str(image.dtype),
        'compress': 'lzw'
    }

    # # Save the image as a GeoTIFF file


    # Save the image as a GeoTIFF file
    with rasterio.open(save_path, 'w', **meta) as dst:
        # If 'image' has an alpha channel, include it in the write call
        if bands == 4:
            # Write RGB bands
            dst.write(image[:, :, :3].transpose(2, 0, 1), indexes=[1, 2, 3])
            # Write alpha band
            dst.write(image[:, :, 3]*100, indexes=4)
        else:
            dst.write(image.transpose(2, 0, 1))

    return


def raster_to_vector(mask, id, image_bbox, CLASSES, bg_is_0=False):
    """
    Takes a mask as input and returns a list of shapely polygons
    """

    label_list = []
    geometry_list = []
    image_shape = mask.shape

    for i, class_id in enumerate(CLASSES):
        prediction = np.zeros(image_shape)
        if bg_is_0:
            prediction[mask == i+1] = 1
        else:
            prediction[mask == i] = 1
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            # get the contours from the mask
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Convert the contours list to a Numpy array
            contours = np.array(contours, dtype=object)

            # convert the contours to shapely polygon
            for cnt in contours:
                cnt = cnt.reshape(-1, 2)
                try:
                    shapely_poly = Polygon(cnt)
                except ValueError:
                    continue
                # write the results to lists
                geometry_list.append(shapely_poly)
                label_list.append(CLASSES[i])

    image_bbox_px = box(0, 0, image_shape[0], image_shape[1])

    geometry_list = [convert_geocoord_and_pixel(geom, image_bbox_px, image_bbox, case='px_to_coord')
                     for geom in geometry_list]
    geometry_list = [Polygon(geometry) for geometry in geometry_list]

    gdf_labels = gpd.GeoDataFrame({
        'id': list([id]) * len(label_list),
        'label': label_list,
        'geometry': geometry_list
    })

    return gdf_labels


def switch_coordinates(x, y):
    """ function to switch x and y coordinates"""
    return y, x


def convert_lonlat_to_latlon(obj):
    """function to switch longitude, latitude coordinates to latitude, longitude coordinates"""
    return shapely.ops.transform(switch_coordinates, obj)


def convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target):
    iox_min = img_box_orig.bounds[0]
    iox_max = img_box_orig.bounds[2]
    itx_min = img_box_target.bounds[0]
    itx_max = img_box_target.bounds[2]

    ioy_min = img_box_orig.bounds[1]
    ioy_max = img_box_orig.bounds[3]
    ity_min = img_box_target.bounds[1]
    ity_max = img_box_target.bounds[3]

    point_list = []
    for x, y in coords:
        x_new = itx_min + ((x - iox_min) / (iox_max - iox_min) * (itx_max - itx_min))
        y_new = ity_min + ((ioy_max - y) / (ioy_max - ioy_min) * (ity_max - ity_min))
        point_list.append((x_new, y_new))
    return point_list


def convert_geocoord_and_pixel(geom, img_box_px, img_box_latlon, case='coord_to_px'):
    # todo: describe function
    if case == 'px_to_coord':
        img_box_orig = img_box_px
        img_box_target = img_box_latlon
    elif case == 'coord_to_px':
        img_box_orig = img_box_latlon
        img_box_target = img_box_px
    else:
        print('transformation case not covered: try case=px_to_latlon ')

    if type(geom) == MultiPolygon:
        pol_list = []
        for pol in geom.geoms:
            p_list = convert_points_geocoord_and_pixel(pol.exterior.coords, img_box_orig, img_box_target)
            pol_list.append(Polygon(p_list))
        converted_geom = MultiPolygon(pol_list)
    elif type(geom) == MultiLineString:
        linestring_list = []
        for ls in geom.geoms:
            p_list = convert_points_geocoord_and_pixel(ls.coords, img_box_orig, img_box_target)
            linestring_list.append(LineString(p_list))
        converted_geom = MultiLineString(linestring_list)
    elif type(geom) == Polygon:
        coords = geom.exterior.coords
        point_list = convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target)
        converted_geom = Polygon(point_list)
    elif type(geom) == LineString:
        coords = geom.coords
        point_list = convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target)
        converted_geom = LineString(point_list)
    return converted_geom


def simplify_polygon(polygon, tolerance=1):
    simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=False)
    if type(simplified_polygon) == MultiPolygon:
        simplified_polygon = simplified_polygon.geoms[0]
    if simplified_polygon.is_empty:
        simplified_polygon = polygon.simplify(tolerance=0.5, preserve_topology=False)
        if type(simplified_polygon) == MultiPolygon:
            simplified_polygon = simplified_polygon.geoms[0]
        if simplified_polygon.is_empty:
            simplified_polygon = polygon
    return simplified_polygon


def calculate_rotation_angle(line):
    """
    Calculate angle to rotate modules.
    """
    try:
        slope = (line[1][1]-line[0][1])/(line[1][0]-line[0][0])
        angle_rad = np.arctan(slope)
        angle_deg = angle_rad * 180 / np.pi
    except ZeroDivisionError:
        angle_deg = 90 # happens for east/west
    return angle_deg


def opposite_angle(angle):
    opposite = (angle + 180) % 360
    if opposite > 180:
        opposite -= 360
    return opposite


def select_azimuth(orientation, angle1, angle2, orientation_mapping=None):
    # copy original angle, otherwise this can lead to error with abs(angle) in North case
    orig_angle1 = angle1
    orig_angle2 = angle2

    if orientation_mapping is None:
        orientation_mapping = {'N': 180, 'E': -90, 'S': 0, 'W': 90}

    if orientation == "N":
        angle1 = abs(angle1)
        angle2 = abs(angle2)

    diff1 = abs(angle1 - orientation_mapping[orientation])
    diff2 = abs(angle2 - orientation_mapping[orientation])

    # Choose the angle with the smaller absolute difference
    selected_angle = orig_angle1 if diff1 < diff2 else orig_angle2
    return selected_angle



def get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF, save_to_png_path=[]):
    image_id_list = [id[:-4] for id in os.listdir(DIR_IMAGES_GEOTIFF) if id[-4:] == '.tif']

    # open image
    raster_srcs = [rasterio.open(os.path.join(DIR_IMAGES_GEOTIFF, str(image_id) + ".tif")) for image_id in image_id_list]
    image_bbox_list = []
    image_width_px = []
    image_height_px = []
    print('')
    
    for i, raster_src in enumerate(raster_srcs):
        progress_string = get_progress_string(round(i / len(raster_srcs), 2))
        print('Loading geo_tiffs: ' + progress_string, end="\r")
        #print(i)

        # initialize rgb image with shape of .tif
        if len(save_to_png_path) > 0:
            filename_mask = os.path.join(save_to_png_path, str(image_id_list[i]) + '.png')
            data = raster_src.read()
            img = np.dstack((data[0, :, :], data[1, :, :], data[2, :, :]))
            cv2.imwrite(filename_mask, img)

        band_shape = raster_src.shape
        image_width_px.append(band_shape[1])
        image_height_px.append(band_shape[0])

        # add image bounding box from geotiff to geodataframe
        transform = raster_src.transform
        ulx, uly = transform * (0, 0)
        lrx, lry = transform * (raster_src.width, raster_src.height)

        image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
        image_bbox_list.append(image_bbox)

    gdf_images = gpd.GeoDataFrame({
        'id': image_id_list,
        'geometry': image_bbox_list,
        'image_width_px': image_width_px,
        'image_height_px': image_height_px,
    })
    gdf_images.crs = raster_srcs[0].crs.to_dict()
    return gdf_images