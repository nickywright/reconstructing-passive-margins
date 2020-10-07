import math
import numpy as np
from ptt.utils import points_in_polygons
from ptt.utils import points_spatial_tree
import pygplates

import os
from gprm.utils import inpaint
from netCDF4 import Dataset
import scipy.interpolate as spi

#################################################
# Included functions
#################################################
def sample_grid_using_scipy(x,y,grdfile):
    
    data=Dataset(grdfile,'r')
    try:
        lon = np.copy(data.variables['x'][:])
        lat = np.copy(data.variables['y'][:])
    except:
        lon = np.copy(data.variables['lon'][:])
        lat = np.copy(data.variables['lat'][:])
    
    Zg = data.variables['z'][:]
    
    test = inpaint.fill_ndimage(Zg)
    
    lut=spi.RectBivariateSpline(lon,lat,test.T)
    result = []
    for xi,yi in zip(x,y):
        result.append(lut(xi, yi)[0][0])
            
    return result


def generate_points_grid(grid_spacing_degrees):
    
    if grid_spacing_degrees == 0:
        return
    
    points = []
    
    # Data points start *on* dateline (-180).
    # If 180 is an integer multiple of grid spacing then final longitude also lands on dateline (+180).
    num_latitudes = int(math.floor(180.0 / grid_spacing_degrees)) + 1
    num_longitudes = int(math.floor(360.0 / grid_spacing_degrees)) + 1
    for lat_index in range(num_latitudes):
        lat = -90 + lat_index * grid_spacing_degrees
        
        for lon_index in range(num_longitudes):
            lon = -180 + lon_index * grid_spacing_degrees
            
            points.append(pygplates.PointOnSphere(lat, lon))

    # num_latitudes = int(math.floor(180.0 / grid_spacing_degrees))
    # num_longitudes = int(math.floor(360.0 / grid_spacing_degrees))
    # for lat_index in range(num_latitudes):
    #     # The 0.5 puts the point in the centre of the grid pixel.
    #     # This also avoids sampling right on the poles.
    #     lat = -90 + (lat_index + 0.5) * grid_spacing_degrees
    #     
    #     for lon_index in range(num_longitudes):
    #         # The 0.5 puts the point in the centre of the grid pixel.
    #         # This also avoids sampling right on the dateline where there might be
    #         # age grid or static polygon artifacts.
    #         lon = -180 + (lon_index + 0.5) * grid_spacing_degrees
    #         
    #         points.append(pygplates.PointOnSphere(lat, lon))
    
    return points


def write_xyz_file(output_filename, output_data):
    with open(output_filename, 'w') as output_file:
        for output_line in output_data:
            output_file.write(' '.join(str(item) for item in output_line) + '\n')


def reconstruct_scalar_coverage(uniform_recon_points,spatial_tree_of_uniform_recon_points,
                                static_polygon_features,rotation_model,time):
    print('Reconstruct static polygons...')
    
    # Reconstruct the multipoint feature.
    recon_static_polygon_features = []
    pygplates.reconstruct(static_polygon_features, rotation_model, recon_static_polygon_features, time)
    
    # Extract the polygons and plate IDs from the reconstructed static polygons.
    recon_static_polygons = []
    recon_static_polygon_plate_ids = []
    for recon_static_polygon_feature in recon_static_polygon_features:
        recon_plate_id = recon_static_polygon_feature.get_feature().get_reconstruction_plate_id()
        recon_polygon = recon_static_polygon_feature.get_reconstructed_geometry()
        
        recon_static_polygon_plate_ids.append(recon_plate_id)
        recon_static_polygons.append(recon_polygon)
    
    print('Find static polygons...')
    
    # Find the reconstructed static polygon (plate IDs) containing the uniform (reconstructed) points.
    #
    # The order (and length) of 'recon_point_plate_ids' matches the order (and length) of 'uniform_recon_points'.
    # Points outside all static polygons return a value of None.
    recon_point_plate_ids = points_in_polygons.find_polygons_using_points_spatial_tree(
            uniform_recon_points, spatial_tree_of_uniform_recon_points, recon_static_polygons, recon_static_polygon_plate_ids)
    
    print('Group by polygons...')
    
    # Group recon points with plate IDs so we can later create one multipoint per plate.
    recon_points_grouped_by_plate_id = {}
    for point_index, point_plate_id in enumerate(recon_point_plate_ids):
        # Reject any points outside all reconstructed static polygons.
        if point_plate_id is None:
            continue
        
        # Add empty list to dict if first time encountering plate ID.
        if point_plate_id not in recon_points_grouped_by_plate_id:
            recon_points_grouped_by_plate_id[point_plate_id] = []
        
        # Add to list of points associated with plate ID.
        recon_point = uniform_recon_points[point_index]
        recon_points_grouped_by_plate_id[point_plate_id].append(recon_point)
    
    print('Reverse reconstruct points...')
    
    # Reconstructed points.
    recon_point_lons = []
    recon_point_lats = []
    
    # Present day points associated with reconstructed points.
    point_lons = []
    point_lats = []
    
    # Create a multipoint feature for each plate ID and reverse-reconstruct it to get present-day points.
    #
    # Iterate over key/value pairs in dictionary.
    for plate_id, recon_points_in_plate in recon_points_grouped_by_plate_id.iteritems():
        # Reverse reconstructing a multipoint is much faster than individually reverse-reconstructing points.
        multipoint_feature = pygplates.Feature()
        multipoint_feature.set_geometry(pygplates.MultiPointOnSphere(recon_points_in_plate))
        multipoint_feature.set_reconstruction_plate_id(plate_id)
        
        # Reverse reconstruct the multipoint feature.
        pygplates.reverse_reconstruct(multipoint_feature, rotation_model, time)
        
        # Extract reverse-reconstructed geometry.
        multipoint = multipoint_feature.get_geometry()
        
        # Collect present day and associated reconstructed points.
        for point_index, point in enumerate(multipoint):
            lat, lon = point.to_lat_lon()
            point_lons.append(lon)
            point_lats.append(lat)
            
            recon_point = recon_points_in_plate[point_index]
            recon_lat, recon_lon = recon_point.to_lat_lon()
            recon_point_lons.append(recon_lon)
            recon_point_lats.append(recon_lat)

    return point_lons,point_lats,recon_point_lons,recon_point_lats


def group_points_by_plate_id(points,static_polygon_features):

    # Extract the polygons and plate IDs from the static polygon features.
    static_polygons = []
    static_polygon_plate_ids = []
    for static_polygon_feature in static_polygon_features:
        recon_plate_id = static_polygon_feature.get_reconstruction_plate_id()
        for polygon in static_polygon_feature.get_geometries():
            static_polygons.append(polygon)
            static_polygon_plate_ids.append(recon_plate_id)

    print('Find static polygons...')

    # Find the static polygon (plate IDs) containing the points.
    #
    # The order (and length) of 'point_plate_ids' matches the order (and length) of 'points'.
    # Points outside all static polygons return a value of None.
    point_plate_ids = points_in_polygons.find_polygons(points, static_polygons, static_polygon_plate_ids)

    print('Group by polygons...')

    # Group points with plate IDs so we can later create one multipoint per plate.
    points_grouped_by_plate_id = {}
    for point_index, point_plate_id in enumerate(point_plate_ids):
        # Use plate ID of static polygon (default to zero for points outside all static polygons).
        if point_plate_id is None:
            point_plate_id = 0

        # Add empty list to dict if first time encountering plate ID.
        if point_plate_id not in points_grouped_by_plate_id:
            points_grouped_by_plate_id[point_plate_id] = []

        # Add to list of points associated with plate ID.
        # We add point *index* instead of point so we can match with present day point.
        points_grouped_by_plate_id[point_plate_id].append(point_index)

    return points_grouped_by_plate_id


def reconstruct_point_groups(points,points_grouped_by_plate_id,rotation_model,time,anchor_plate_id=0):

    # Reconstructed points - we'll fill this in below.
    #
    # This will have the same number of points as 'points' (and in same order).
    recon_points = [None] * len(points)

    # Create a multipoint feature for each plate ID and reconstruct it.
    # print(points_grouped_by_plate_id)
    # Iterate over key/value pairs in dictionary.
    for plate_id, point_indices in points_grouped_by_plate_id.items():
        # Convert indices back to points.
        points_in_plate = [points[point_index] for point_index in point_indices]
        
        # Reconstructing a multipoint is much faster than individually reconstructing points.
        multipoint_feature = pygplates.Feature()
        multipoint_feature.set_geometry(pygplates.MultiPointOnSphere(points_in_plate))
        multipoint_feature.set_reconstruction_plate_id(plate_id)
        
        # Reconstruct the multipoint feature.
        recon_multipoint_feature = []
        pygplates.reconstruct(multipoint_feature, rotation_model, recon_multipoint_feature, float(time), anchor_plate_id=anchor_plate_id)
        
        # Should only be one reconstructed geometry (multipoint).
        recon_multipoint = recon_multipoint_feature[0].get_reconstructed_geometry()
        
        # Iterate over points in reconstructed multipoint and place them back into original reconstructed array.
        for multipoint_index, recon_point in enumerate(recon_multipoint):
            # Convert index in reconstructed multipoint into index in original points.
            point_index = point_indices[multipoint_index]
            # Store back into original reconstructed array.
            recon_points[point_index] = recon_point
    
    # Convert recon points to lat/lon.
    recon_point_lons = []
    recon_point_lats = []
    for recon_point in recon_points:
        recon_lat, recon_lon = recon_point.to_lat_lon()
        recon_point_lons.append(recon_lon)
        recon_point_lats.append(recon_lat)

    return recon_point_lons,recon_point_lats

