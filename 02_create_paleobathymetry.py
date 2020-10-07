# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------- This script creates grids of the depth evolution of passive margins -------
# ---
# --- Passive margins are reconstructed through time based on subsidence parameters
# --- from part 1 (01_run_subsidence.py), and the desired plate model, and sedimentation
# --- model (e.g. constant sedimentatation; or sedimentation that keeps up with subsidence
# ---
# --- Inputs:
#           - Rotation file (as .rot)
#           - Static polygon (as .gpml)
#           - Passive margin file (as .gpml)
#             This is a polygon outling which passive margins to reconstruct
#           - 'subsidenceinfo_[date].txt' from part 1 (01_run_subsidence.py)
#           - Sedimentation mode: set as 'Constant' for constant sedimentation, otherwise will
#             use sedimentation that keeps with subsidence
#
# --- Outputs:
#           - Reconstructed basement depth for passive margins regions (as netcdfs)
#           - Reconstructed sediment thickness for passive margin regions (as netcdfs)
#           - Reconstructed paleobathymetry for passive margin regions (as netcdfs)
#
# Created by Simon Williams, John Cannon, Nicky Wright
#
# Copyright (c) 2020 The University of Sydney. All rights reserved.
#
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
# import sys
import multiprocessing
import time
import os
import pygplates
# from ptt.utils import points_spatial_tree
from ptt.utils import points_in_polygons
from ptt.utils.call_system_command import call_system_command
# from gprm.utils.sphere import create_tree_for_spherical_data, sampleOnSphere

# from this folder
# from Package_ReconstructScalarCoverages import sample_grid_using_scipy, generate_points_grid, reconstruct_scalar_coverage, write_xyz_file
from Package_ReconstructScalarCoverages import write_xyz_file
from Package_ReconstructScalarCoverages import group_points_by_plate_id, reconstruct_point_groups
from tectonic_subsidence import evaluate_subsidence_at_time  # run_grid_pip


# ------------------------- Set parameters and files ------------------------------------
# --- Input files
data_dir = './input_data'
rotation_file = '%s/Global_410-0Ma_Rotations_2019_v2.rot' % data_dir
static_polygon_file = '%s/StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons_2019_v1_polygon.shp' % data_dir
passivemarginfile = '%s/passive_margins_present_day.gpml' % data_dir

# --- Parameters
# Set the output grid spacing
grid_spacing = 0.1

# Specify times for calculation
min_time = 0
max_time = 2
time_step = 1

# source_data = './subsidenceinfo_20200224.txt'
source_data = './subsidenceinfo_20201007.txt'       # from part 1

sedimentation_mode = 'Constant'         # Set as 'Constant' for constant, otherwise it will use keep pace
output_directory = 'out_constant'

anchor_plate_id = 0  # Best to keep as 0, this is only in case you really want to change it

# %%
start_time = time.time()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Should not need to modify anything below here
# ---------------------- Read in files ----------------------------------
# --- Read in relevant files to pygplates
print("... Importing input files")
rotation_model = pygplates.RotationModel(rotation_file)
static_polygon_features = pygplates.FeatureCollection(static_polygon_file)

# --- Load points from previous script, with all necessary values defined
print('... Load precomputed points data produced by script 01, using %s' % source_data)
points_data = np.loadtxt(source_data)
longitude = points_data[:, 0]
latitude = points_data[:, 1]
depth = points_data[:, 2]
sedthickness = points_data[:, 3]
rift_start = points_data[:, 4]
rift_end = points_data[:, 5]
beta = points_data[:, 6]

# --- Because we need to reconstruct the points, cookie-cut them to the static
# --- polygons for the selected model
# Create a multipoint from the loaded lat,lons
input_points = pygplates.MultiPointOnSphere(zip(latitude.flatten(), longitude.flatten())).to_lat_lon_point_list()

# --------------------------------------------------------
# --- Polygon test part 1
# Make a list of plate ids for each point in the originally loaded txt file

# First iterate over each polygon, to make a list of plate ids that are passed
# as the appropriate mapping (proxy) value to the 'find_polygons' function
static_polygons = []
static_polygon_plate_ids = []
for static_polygon_feature in static_polygon_features:
    plate_id = static_polygon_feature.get_reconstruction_plate_id()
    spolygon = static_polygon_feature.get_geometry()

    static_polygons.append(spolygon)
    static_polygon_plate_ids.append(plate_id)

# This array lists plate_ids for all input points
input_point_plate_ids = points_in_polygons.find_polygons(input_points,
                                                         static_polygons,
                                                         static_polygon_plate_ids)

# --------------------------------------------------------
# --- Polygon test part 2

# Load a file that contains polygons defining the extent of passive margin
# regions of interest (put another way, it will exclude regions that we are not
# interested in and so do not want to calculate rift-related subsidence for)

# Do a point in polygon test to isolate only those points which are within the
# passive margin polygons extent

passive_margin_polygons = pygplates.FeatureCollection(passivemarginfile)
reconstructed_passive_margin_polygons = []

pygplates.reconstruct(passive_margin_polygons, rotation_model, reconstructed_passive_margin_polygons, 0)
rpolygons = []
for polygon in reconstructed_passive_margin_polygons:
    if polygon.get_reconstructed_geometry():
        rpolygons.append(polygon.get_reconstructed_geometry())

# polygons_containing_points is a list of polygons that each point is in, or
# 'None' if it is not within any polygon, so can be used to determine whether
# to keep or exclude points
polygons_containing_points = points_in_polygons.find_polygons(input_points, rpolygons)

# --------------------------------------------------------
# Make empty arrays in which to store points that are within the passive margin polygons
lat2 = []
lon2 = []
depth2 = []
sedthickness2 = []
rift_start2 = []
rift_end2 = []
beta2 = []
plate_id2 = []

# Iterate over all points in input file,
# only append points that fall within passive margin polygon test,
# also append the plate ids determined from previous step
for index,(pcp,point,point_plate_id) in enumerate(zip(polygons_containing_points,input_points,input_point_plate_ids)):
    if pcp is not None:
        lat2.append(point.get_latitude())
        lon2.append(point.get_longitude())
        depth2.append(depth[index])
        sedthickness2.append(sedthickness[index])
        rift_start2.append(rift_start[index])
        rift_end2.append(rift_end[index])
        beta2.append(beta[index])
        plate_id2.append(point_plate_id)


# clip points is a multipoint that contains just the points in the clipped region
clip_points = pygplates.MultiPointOnSphere(zip(lat2,lon2))

# print info about which sedimentation model is being used outside of the loop
if sedimentation_mode == 'Constant':
    print('Sedimentation model: Constant')
else:
    print('Sedimentation model: keeps up with subsidence')

# --------------------------------------------------------
# Cell defining functions for number-crunching loop
def get_paleobathymetry_snapshot(latitude, longitude,
                                 rift_start, rift_end,
                                 beta, depth, sedthickness,
                                 output_directory,
                                 recon_time):
    print('Working on time %0.2f Ma...' % recon_time)
    # evaluate subsidence
    paleobathymetry = []
    bsmt = []
    riftend = []
    riftstart = []
    sedthick = []
    equal = []
    beta_out = []
    # count = 0

    for plat, plon, prs, pre, pbeta, pBathy, psedThick in zip(latitude, longitude, rift_start, rift_end, beta, depth, sedthickness):

        # If the reconstruction time is greater than the rift start, there is
        # no subsidence
        if recon_time >= prs:
            paleobathymetry.append(0.)
            bsmt.append(0.)  # why was this 999. ???
            riftend.append(pre)
            riftstart.append(prs)
            sedthick.append(0.)
            if pre == prs:
                equalr = 1
            else:
                equalr = 0
            equal.append(equalr)
            beta_out.append(pbeta)

        else:

            if sedimentation_mode == 'Constant':
                # Determine amount of sediment that would have accumulated by
                # this time, based on constant rate of accumulation since rift
                # start time
                time_fraction = (float(prs)-float(recon_time))/float(prs)
                min_thickness = 0.01  # (adding 0.01 meters to stop thickness from ever being zero)
                psedThick_at_time = (psedThick * time_fraction) + min_thickness

            # elif sedimentation_mode is 'Keep_Pace':
            else:
                psedThick_at_time = psedThick

            # If beta is nan, then the subsequent functions won't work. Assume this
            # is because TTS is zero
            if np.isnan(pbeta):
                bsmt_depth = 0.
            else:
                bsmt_depth = evaluate_subsidence_at_time(prs, pre, pbeta, psedThick_at_time, pBathy, recon_time)

            # Also handle cases where pre and prs are the same (which shouldn't
            # happen, but could if inputs contain weirdness)
            # IS THIS USED FOR ANYTHING????
            if pre == prs:
                equalr = 1
            else:
                equalr = 0
            equal.append(equalr)

            bsmt.append(bsmt_depth)
            riftend.append(pre)
            riftstart.append(prs)
            sedthick.append(psedThick_at_time)
            beta_out.append(pbeta)

            # Calculate paleobathymetry, set to zero if negative (e.g. where
            # sediment thickness is overestimated)
            if np.less(float(bsmt_depth), float(psedThick_at_time)):
                paleobathymetry.append(0.)
            else:
                paleobathymetry.append(bsmt_depth - psedThick_at_time)

    recon_point_lons, recon_point_lats = reconstruct_point_groups(clip_points,
                                                                  points_grouped_by_plate_id,
                                                                  rotation_model,
                                                                  recon_time,
                                                                  anchor_plate_id)

    # Write out data into multi-column ascii file
    if not os.path.exists('out_tmp'):
        os.makedirs('out_tmp')
    write_xyz_file('out_tmp/tmp_%0.2f.xyz' % recon_time, zip(recon_point_lons, recon_point_lats, paleobathymetry, bsmt, riftstart, riftend, sedthick, equal, beta_out))

    # pre-processing step (block median) - test if it makes output grid better??
    #call_system_command(['gmt', 'blockmedian', 'out_tmp/tmp_%0.2f.xyz' % recon_time,
    #                     '-Rg', '-I%0.8fd' % grid_spacing, '-i0,1,2',
    #                     '>', 'out_tmp/tmp2_%0.2f.xyz' % recon_time])

    call_system_command(['gmt', 'nearneighbor',
                         'out_tmp/tmp_%0.2f.xyz' % recon_time,
                         '-G%s/paleobathy_%0.2f.nc' % (output_directory,recon_time),
                         '-Rg', '-I%0.8fd' % grid_spacing,
                         '-N4/1','-S%0.8fd' % grid_spacing, '-i0,1,2', '-fg'])
    call_system_command(['gmt', 'nearneighbor',
                         'out_tmp/tmp_%0.2f.xyz' % recon_time,
                         '-G%s/bsmt_%0.2f.nc' % (output_directory,recon_time),
                         '-Rg', '-I%0.8f' % grid_spacing,
                         '-N4/1','-S%0.8fd' % grid_spacing, '-i0,1,3', '-fg'])
    call_system_command(['gmt', 'nearneighbor',
                         'out_tmp/tmp_%0.2f.xyz' % recon_time,
                         '-G%s/sedthick_%0.2f.nc' % (output_directory,recon_time),
                         '-Rg', '-I%0.8f' % grid_spacing,
                         '-N4/1','-S%0.8fd' % grid_spacing, '-i0,1,6', '-fg'])

    #call_system_command(['gmt', 'xyz2grd',
    #                     'out_tmp/tmp2_%0.2f.xyz' % recon_time,
    #                     '-G%s/paleobathy_%0.2f.nc' % (output_directory,recon_time),
    #                     '-Rg', '-I%0.8f' % grid_spacing])

    # clean up
#     call_system_command(['rm', 'out_tmp/tmp_%0.2f.xyz' % recon_time])
    #call_system_command(['rm', 'out_tmp/tmp2_%0.2f.xyz' % recon_time])

    print("--- %s seconds ---" % (time.time() - start_time))


def get_paleobathymetry_snapshot_pool_function(args):
    try:
        return get_paleobathymetry_snapshot(*args)
    except KeyboardInterrupt:
        pass


def get_paleobathymetry_snapshot_parallel(latitude,
                                          longitude,
                                          rift_start,
                                          rift_end,
                                          beta,
                                          depth,
                                          sedthickness,
                                          output_directory,
                                          pool_recon_time_list):

    num_cpus_overall = multiprocessing.cpu_count()
    num_cpus = num_cpus_overall - 2  # don't use all the cpus available

    # Split the workload across the CPUs.
    try:
        pool = multiprocessing.Pool(num_cpus)
        for pool_recon_time in pool_recon_time_list:
            pool_map_async_result = pool.map_async(
                get_paleobathymetry_snapshot_pool_function,
                [(latitude, longitude, rift_start, rift_end, beta, depth, sedthickness, output_directory, pool_recon_time)], 1)  # chunksize
            # print 'Done time %0.2f Ma' % pool_recon_time
    finally:
        pool.close()
        pool.join()
    # Apparently if we use pool.map_async instead of pool.map and then get the results
    # using a timeout, then we avoid a bug in Python where a keyboard interrupt does not work properly.
    # See http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
    try:
        pool_output_datas = pool_map_async_result.get(99999)
    except KeyboardInterrupt:
        return

# --------------------------------------------------------
print('... Grouping points by plate id')
points_grouped_by_plate_id = group_points_by_plate_id(clip_points, static_polygon_features)

print("--- %s seconds ---" % (time.time() - start_time))

if not os.path.exists(output_directory):
    print("... Creating " + str(output_directory) + " now")
    os.makedirs(output_directory)

pool_recon_time_list = np.arange(min_time, max_time, time_step)

get_paleobathymetry_snapshot_parallel(lat2, lon2,
                                      rift_start2, rift_end2,
                                      beta2, depth2, sedthickness2,
                                      output_directory,
                                      pool_recon_time_list)
print('Done!')
