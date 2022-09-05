
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------- This script calculates subsidence parameters for passive margins ----------
# ---
# --- Present-day passive margin regions are used to determine stretching factors and
# --- sediment thickness through time, based on:
#           - timing and duration of rifting (determined from the plate model)
#           - present-day sediment thickness (e.g. GlobSed)
#           - present-day bathymetry (e.g. ETOPO)
#           - present-day basement depth (based on bathymetry and sediemnt thickness)
#           - assumed intial crustal thickness (35 km)
#           - sediment density as a function of depth (from Sawyer, 1985)
# --- Part 2 of this workflow ('02_create_paleobathymetry.py') uses this information to
# --- create grids through time.
#
# --- Inputs:
#           - Rotation file (as .rot)
#           - COB terranes file (as .gpml). All the geometries in this must be polygons
#           - IsoCOB file (as .gpml)
#           - present-day bathymetry/ETOPO file in metres (as .nc or .grd)
#           - present-day sediment thickness file in metres (as .nc or .grd)
#
# --- Outputs:
#           - text file 'subsidenceinfo_[date].txt' with columns:
#          longitude | latitude | depth (m) | sediment thickness (m) | rift start age (Ma) | rift end age (Ma) | stretching factor (beta) | depth mismatch
#
# Created by Simon Williams, John Cannon, Nicky Wright
#
# Copyright (c) 2020 The University of Sydney. All rights reserved.
#
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import xarray as xr
import time
import pygplates
from tectonic_subsidence import grd2multipoint, isocob_rift_times, run_optimisation_for_dataset
from gprm.utils.sphere import sampleOnSphere

# ------------------------- Set parameters and files ------------------------------------
# --- Input files
# data_dir = 'input_data'

# modified for 2022_v2
plate_model_dir = '/Users/nickywright/repos/usyd/EarthBytePlateMotionModel-ARCHIVE/Global_Model_WD_Internal_Release_2022_v2'

rotation_filenames = [
    '%s/Alps_Mesh_Rotations.rot' % plate_model_dir,
    '%s/Andes_Flat_Slabs_Rotations.rot' % plate_model_dir,
    '%s/Andes_Rotations.rot' % plate_model_dir,
    '%s/Australia_Antarctica_Mesh_Rotations.rot' % plate_model_dir,
    '%s/Australia_North_Zealandia_Rotations.rot' % plate_model_dir,
    '%s/Eurasia_Arabia_Mesh_Rotations.rot' % plate_model_dir,
    '%s/Global_250-0Ma_Rotations.rot' % plate_model_dir,
    '%s/Global_410-250Ma_Rotations.rot' % plate_model_dir,
    '%s/North_America_Flat_Slabs_Rotations.rot' % plate_model_dir,
    '%s/North_America_Mesh_Rotations.rot' % plate_model_dir,
    '%s/North_China_Mesh_Rotations.rot' % plate_model_dir,
    '%s/South_Atlantic_Rotations.rot' % plate_model_dir,
    '%s/South_China_DeformingModel.rot' % plate_model_dir,
    '%s/Southeast_Asia_Rotations.rot' % plate_model_dir]

COBterrane_file = '%s/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_COB_Terranes.gpml' % plate_model_dir
isocob_features = '%s/StaticGeometries/AgeGridInput/Global_EarthByte_GeeK07_IsoCOB.gpml' % plate_model_dir

# rotation_file = '%s/Global_410-0Ma_Rotations_2019_v2.rot' % data_dir
# etopo_file = '%s/ETOPO1_ice_smoothed_20km_01d_all_rg.nc' % data_dir  # this has been smoothed using grdfilter
etopo_file = '/Users/nickywright/Data/Bathymetry_Topography/ETOPO/ETOPO1_Ice_g_6m_Rd.nc'
# sedimentthickness_file = '%s/GlobSed-v2_6m_clipped.nc' % data_dir
sedimentthickness_file = '/Users/nickywright/Data/SedimentThickness/GlobSed_package2/GlobSed-v2.nc'


# --- Parameters
lon_min = -180.
lon_max = 180.
lat_min = -90.
lat_max = 90.

sampling_factor = 1     # increase to 'resample' into a coarser grid - useful for testing purposes only

no_iterations = 20      # number of iterations for the workflow. Use a lower number (i.e. 1) for testing, and 20 for the final grid

today = time.strftime("%Y%m%d")  # get today's date in YYYYMMDD format - for creating the output file


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Should not need to modify anything below here
# ---------------------- Read in files ----------------------------------
# --- Read in relevant files to pygplates
print("... Importing input files")
cobter = pygplates.FeatureCollection(COBterrane_file)
rotation_model = pygplates.RotationModel(rotation_filenames)
cob_lines = pygplates.FeatureCollection(isocob_features)

# --------------------------------------------------------
# ---- Get valid time from isocob file
print("... Getting rift times from isocobs")

cob_lines_present = []   # create an empty array to add points to
for cob in cob_lines:
    if cob.get_valid_time()[1] <= 0:
        cob_lines_present.append(cob)

# ---- checking that COB geometries are polygons

# Convert polylines to polygons in the features.
for feature in cobter:
        # polygons = [pygplates.PolygonOnSphere(geometry) for geometry in feature.get_geometries()]
        # changed feature get statement to 'all_geometries' - since feature has issues getting different
        # types of geometries
    for geometry in feature.get_all_geometries():
        polygons = pygplates.PolygonOnSphere(geometry)
        feature.set_geometry(polygons)


# --------------------------------------------------------
# Get bathymetry within bounding box as point data
subsidence_points_lon, subsidence_points_lat, subsidence_points_z = grd2multipoint(etopo_file,
                                                                                   cobter,
                                                                                   rotation_model,
                                                                                   lon_min,
                                                                                   lon_max,
                                                                                   lat_min,
                                                                                   lat_max,
                                                                                   sampling_factor)

# --------------------------------------------------------
# Get Rift Start/End times at pointd by interpolation from IsoCOB properties
pts_lon, pts_lat, pts_re, pts_rs = isocob_rift_times(cob_lines_present, rotation_model)

# interpolate the 'rift end' ages onto chosen grid points
d, l = sampleOnSphere(np.hstack(pts_lat),
                      np.hstack(pts_lon),
                      np.hstack(pts_re),
                      subsidence_points_lat,
                      subsidence_points_lon,
                      n=4)
interp_re = np.hstack(pts_re).ravel()[l]

# interpolate the 'rift start' ages onto chosen grid points
d, l = sampleOnSphere(np.hstack(pts_lat),
                      np.hstack(pts_lon),
                      np.hstack(pts_rs),
                      subsidence_points_lat,
                      subsidence_points_lon,
                      n=4)
interp_rs = np.hstack(pts_rs).ravel()[l]

# --------------------------------------------------------
# --- To do backstripping, need to load in a sediment thickness grid
print("... Getting sediment thickness grid")

# Then, sample the sediment thickness onto the same points that have been isolated by the previous steps
ds_disk2 = xr.open_dataset(sedimentthickness_file)

sedThick = ds_disk2['z']
sedThick = sedThick.sel(lon=slice(lon_min,lon_max)).sel(lat=slice(lat_min,lat_max))[::sampling_factor,::sampling_factor]
# coord_keys = sedThick.coords.keys()
# sedThickX, sedThickY = np.meshgrid(sedThick.coords[coord_keys[0]].data[::sampling_factor],
#                                    sedThick.coords[coord_keys[1]].data[::sampling_factor])

sedThickX, sedThickY = np.meshgrid(sedThick.lon.data[::sampling_factor], sedThick.lat.data[::sampling_factor])

d, l = sampleOnSphere(sedThickY.flatten(),
                      sedThickX.flatten(),
                      sedThick.data.flatten(),
                      subsidence_points_lat,
                      subsidence_points_lon,
                      n=4)

sedThick_points = sedThick.data.flatten().ravel()[l]
# sedThick_points = sedThick_points * 1000.   # convert Sediment Thickness points from km to m

# Multiply bathymetry by -1 to get positive depths
subsidence_points_z = subsidence_points_z * -1

# --------------------------------------------------------
# non-linear optimisation to determine the beta value that best explains the present combination of bathymetry
# and sediment thickness, given that we know the rift end time
max_iterations = no_iterations

# --------------------------------------------------------
out_lat, out_lon, out_opt, out_xopt, out_minf, out_pres, out_pree, out_psedThicks, out_depths = \
    run_optimisation_for_dataset(subsidence_points_lat,
                                 subsidence_points_lon,
                                 interp_rs,
                                 interp_re,
                                 sedThick_points.tolist(),
                                 subsidence_points_z.tolist(),
                                 max_iterations)

# --------------------- Write output file --------------------------------
tmp = np.vstack((out_lon, out_lat, out_depths, out_psedThicks, out_pres, out_pree, out_xopt, out_minf))
header_add = "Longitude Latitude Bathymetry_m SedimentThickness_m RiftStart_Ma RiftEnd_Ma OptimisedBetaFactor DepthMismatch_m"
format='%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.6f\t%0.6f'
print('... saving output file as subsidenceinfo_' + str(today) + '.txt')
np.savetxt('subsidenceinfo_' + str(today) + '.txt', tmp.T, fmt=format, header=header_add)
print("Done!!")
