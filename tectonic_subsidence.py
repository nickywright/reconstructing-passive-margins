# --- Paleodepths of continental margins and submerged things



# -------------------------------------------------------- 
# --- Import dependencies
#import sys

import pygplates
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import nlopt
# import itertools
import time

from gprm.utils import points_in_polygons
from gprm.utils.create_gpml import create_gpml_regular_long_lat_mesh, create_gpml_healpix_mesh
from gprm.utils.sphere import sampleOnSphere,healpix_mesh
from gprm.utils.spatial import force_polygon_geometries

from Package_ReconstructScalarCoverages import group_points_by_plate_id, reconstruct_point_groups
from Package_ReconstructScalarCoverages import sample_grid_using_scipy, generate_points_grid, reconstruct_scalar_coverage, write_xyz_file
from ptt.utils.call_system_command import call_system_command

import pybacktrack as pbt
from predict_sediment_thickness import decompact_sediment_thickness
from predict_sedimentation_rate import compact_sediment_thickness


# -------------------------------------------------------- 
# Define functions for backstripping. Note that this approach requires that the unstretched crustal thickness must be assumed

# Define initial parameters
#y_c     = 35000;    # Initial crustal thickness in m               [m]
y_l     = 125000;   # Initial lithospheric thickness in m          [m]
#rho_m0  = 3330;     # Density of the mantle at 0 degrees celcius   [kg/m^3]
#rho_c0  = 2800;     # Density of the crust at 0 degrees celcius    [kg/m^3]
#rho_s   = 2066;     # Density of sediments                         [kg/m^3]
alpha_v = 3.28e-5;  # volumetric coefficient of thermal expansion  [1/K]
Tm      = 1333;     # Temperature of the mantle                    [C]
kappa   = 1e-6;     # Thermal diffusivity                          [m^2/s]
rhoW = 1030
rhoSGrains  = 2700
# densities of mantle, water and crust
rhoM = 3330
rhoC = 2800
# More parameters for test - including the ones we might vary...
# parameters that determine sediment density as a function of sediment thickness
phi = 0.56
c = 4.5
tc = 35000   # initial crustal thickness (here assumed uniform globally!?)

DEFAULT_SURFACE_POROSITY = 0.63
DEFAULT_POROSITY_EXP_DECAY = 5.71e-4


# -------------------------------------------------------- 
# Definitions
def run_grid_pip(time, points, polygons, rotation_model):
    reconstructed_polygons = []
    pygplates.reconstruct(polygons, rotation_model, reconstructed_polygons, time)
    rpolygons = []
    for polygon in reconstructed_polygons:
        if polygon.get_reconstructed_geometry():
            rpolygons.append(polygon.get_reconstructed_geometry())
    polygons_containing_points = points_in_polygons.find_polygons(points, rpolygons)
    lat = []
    lon = []
    zval = []
    for pcp,point in zip(polygons_containing_points,points):
        lat.append(point.get_latitude())
        lon.append(point.get_longitude())
        if pcp is not None:
            zval.append(1)
        else:
            zval.append(0)
    #bi = np.array(zval).reshape(181,361)
    return zval


def GetRiftStartTime(rotation_model,pid,cpid,rift_end):
# function to determine onset of motion between two plates that subsequently break apart (such that the rift end is known from the COB age assignment)
    rifting_is_going_on = True
    time = rift_end        # set starting time to be rift_end
    
    # iterate in 1Myr increments, to see if the stage pole is zero (identity rotation)
    # or not. If it is not, there is relative motion between the plates, so we keep going.
    # If the stage pole is identity, there is no relative motion --> we take the
    # rifting as not yet started, break out of the loop, and return this time
    # as the time of rift onset
    # NB only gives rift start to nearest 1 Ma, and will not be an integer (at least not the 
    # way it is coded here) 
    while rifting_is_going_on:
        # Get stage pole for previous 1 Myr stage
        stage_rotation = rotation_model.get_rotation(time,pid,time+1,cpid)
        if stage_rotation.represents_identity_rotation():
            rifting_is_going_on = False
        else:
            time+=1.          
    return time


def grd2multipoint(etopo_file,cobter,rotation_model,lonmin,lonmax,latmin,latmax,sampling_factor=1):
# -------------------------------------------------------- 
# ----  Read in a bathymetry grid

    print("... Importing bathymetry grid")

    # Specify the path to a bathymetry grid
    ds_disk = xr.open_dataset(etopo_file)

    bathy = ds_disk['z']
    bathy = bathy.sel(lon=slice(lonmin,lonmax)).sel(lat=slice(latmin,latmax))[::sampling_factor,::sampling_factor]

    coord_keys = [key for key in bathy.coords.keys()]  # updated for python3 compatibility
    tmpX, tmpY = np.meshgrid(bathy.coords[coord_keys[0]].data,
                             bathy.coords[coord_keys[1]].data)
    tmpZ = bathy.data

    # index defining where the bathymetry is less than 0 (so we can exclude points about present-day sea-level)
    index = np.where(tmpZ < 0)

    # convert the points into a multipoint (since the quick pip function takes this as input)
    # note the index is used here to only include points below sea level
    points = pygplates.MultiPointOnSphere(zip(tmpY[index].flatten(),tmpX[index].flatten())).to_lat_lon_point_list()

    # --- get points within COB and below sea level
    print("... Getting points within COB and below sea level")
    # we are doing 'present day', but still the time is used since the 'run_grid_pip' function uses reconstructed features
    time = 0
    bi = run_grid_pip(time, points, cobter, rotation_model)

    # this index will exclude points not within the COB terranes
    index2 = np.where(np.array(bi)==1)

    # after using this second index, we will have isolated only those points that are
    # 1. below present-day sea-level
    # 2. within the COB terranes at present-day
    subsidence_points_lon = tmpX[index].flatten()[index2]
    subsidence_points_lat = tmpY[index].flatten()[index2]
    subsidence_points_z = tmpZ[index].flatten()[index2]

    return subsidence_points_lon,subsidence_points_lat,subsidence_points_z



def isocob_rift_times(cob_lines_present,rotation_model):
# -------------------------------------------------------- 
# --- get rift times from isocobs

    # create empty lists of the things we want to export
    pts_lon = []
    pts_lat = []
    pts_re = []  # rift end
    pts_rs = [] # rift start

    # iterate over each COB line
    for COB in cob_lines_present:
        # Call function to get the rift start time (based on the rotation sequence for the plate pair
        # associated with this COB line)
        #print(COB.get_reconstruction_plate_id(), COB.get_conjugate_plate_id(), COB.get_valid_time())
        if not np.isfinite(COB.get_valid_time()[0]):
            print('Skipping geometry without finite valid time')
            continue
        rs = GetRiftStartTime(rotation_model,
                              COB.get_reconstruction_plate_id(),
                              COB.get_conjugate_plate_id(),
                              COB.get_valid_time()[0])
        # tesselate the COB line to a sampling fine enough to ensure the interpolation
        # of ages onto the grid points is nice and continuous
        for geometry in COB.get_geometries():
            tes_line = geometry.to_tessellated(np.degrees(0.1))
            tes_line_pts = tes_line.to_lat_lon_array()
            pts_lat.append(tes_line_pts[:,0])
            pts_lon.append(tes_line_pts[:,1])
            pts_re.append(COB.get_valid_time()[0] * np.ones(tes_line_pts[:,0].shape))
            pts_rs.append(rs * np.ones(tes_line_pts[:,0].shape))

    return pts_lon,pts_lat,pts_re,pts_rs


def AverageSedimentDensity(SedimentThickness, phi=phi, c=c):
    """  Calculate average sediment density
    Inputs: - rhoSGrains: density of sediments (2700 kg/m^3)
            - rhoW: density of water (1030 kg/m^3)
            - phi: ?
            - c: 

     Sediment Thickness should be in metres
    """
    rhoSbar = rhoSGrains + ( ((rhoSGrains - rhoW) * phi) / (c * SedimentThickness) ) * (np.exp(-c * SedimentThickness) - 1)
    return rhoSbar


def betafactor(Y, bathy, tc):
    """  The betafactor is the stretching factor.

    This is based on equation 5 in Stewart et al., 2000; GJI
    This equation assumes the lithosphere has zero stregnth (i.e. D = 0).
    This also assumes no vertical movement of horizontal surfaces during rifting

    Inputs: - TTS: total tectonic subsidence (or uplift). 
              This is obtained by adding the present-day water depth to the cumulative backstrip
            - rhoM: density of the mantle (3330 kg/m^3)
            - rhoW: density of water (1030 kg/m^3)
            - rhoC: density of continental crust (2800 kg/m^3)
            - tc: thickness of crust prior to rifting (needs to be assumed)
    """
    TTS = Y + bathy
    beta = 1 / (1 - ( (TTS * (rhoM - rhoW))/ (tc * (rhoM - rhoC)) ) )
    return TTS, beta


def AverageDensityAboveBasement(rhoSbar,psedThick,DEPTH):
    rhoColumnbar = ((rhoSbar*psedThick) + (DEPTH*rhoW)) / (psedThick+DEPTH)
    return rhoColumnbar


# SIMILAR TO PYBACKTRACK.SYN_RIFT_SUBSIDENCE
def syn_rift_subsidence(beta,tc, column_density):
	# Step 1: Calculate synrift subsidence
	# WATER LOADED???
    ys = y_l*((rhoM-rhoC)*tc/y_l*(1-alpha_v*Tm*tc/y_l)-alpha_v*Tm*rhoM/2)*(1-1/beta)/(rhoM*(1-alpha_v*Tm)-column_density)
    #print "... Syn rift subsidence is: " + str(ys)
    return ys


# SIMILAR TO PYBACKTRACK.TOTAL_SUBSIDENCE
def subsidence_curve(time_my,beta,tc,rhoS):
	# McKenzie model for basin subsidence
	# Adapted from matlab code of Sonia Scarselli ETH-Zurich
	
	# WATER LOADED???
    time_s  = time_my*365*24*3600*1e6;        # Time in seconds
    ys = syn_rift_subsidence(beta,tc,rhoS)
    # Step 2: Calculate thermal subsidence with time
    E0 = 4*y_l*rhoM*alpha_v*Tm/((np.pi**2)*(rhoM-rhoS))
    tau = (y_l**2)/((np.pi**2)*kappa)
    # Thermal Subsidence as a function of time
    S = E0*beta/np.pi*np.sin(np.pi/beta)*(1-np.exp(-time_s/tau))
    
    # Return Total Subsidence --> Tectonic + Thermal
    return S + ys



# define a function that will determine subsidence + paleowater depth at a point,
# given the time of rifting (start and end), stretching factor, and final sediment thickness

def evaluate_subsidence_at_time(prs,pre,pbeta,psedThick,pBathy,time):

    # if this point is in the post-rift subsidence phase, 'time' will be less than 'rift end' for this point 
    if (pre-time) >= 0:
    #if pre >= 0:
        sed_Density = AverageSedimentDensity(psedThick,phi,c)
        # print "... sed density is: " + str(sed_Density)
        column_Density = AverageDensityAboveBasement(sed_Density,psedThick,pBathy)
        # print "... column density is: " + str(column_Density)
        #print pre,time,pre-time
        TS = subsidence_curve(pre-time,pbeta,tc,column_Density)
        # print "... TS is:: " + str(TS)
        TS_out = float(TS)
        
    # if 'time' is between 'rift-end' and 'rift-start', assume linear thinning based on beta factor
    else:
        # just need to get syn-rift subsidence - then, assume that the amount of subsidence
        # within this phase can be linearly interpolated based on how far through the phase
        # we are
        sed_Density = AverageSedimentDensity(psedThick,phi,c)
        column_Density = AverageDensityAboveBasement(sed_Density,psedThick,pBathy)
        ys = syn_rift_subsidence(pbeta,tc,column_Density)
        synrift_time_fraction = 1-((time-pre)/(prs-pre))
        #print time, pre, prs, synrift_time_fraction
        TS_out = ys * synrift_time_fraction

    return TS_out


# -------------------------------------------------------- 

# SIMILAR TO PYBACKTRACK.ESTIMATE_BETA
def run_optimisation_for_point(prs, pre, psedThick, DEPTH, max_iterations, verbose=True):
    """ Depth needs to be positive """
    def obj_f(x, grad):
        pbeta = x[0]
        # when running optimisation, assume that subsidence is being computed for present-day
        time=0.
        TS = evaluate_subsidence_at_time(prs, pre, pbeta, psedThick, DEPTH, time)
        opt_eval =  np.abs(TS - (DEPTH + psedThick))
        return opt_eval
    
    #opt_n = [time,prs,pre,psedThick]
    opt_n = 1
    lb = 0.
    ub = 10.
    
    opt = nlopt.opt(nlopt.LN_COBYLA, opt_n)
    opt.set_min_objective(obj_f)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    
    # Select model stop condition
    #if model_stop_condition != 'threshold':
    
    opt.set_maxeval(max_iterations)
    
    x = [1.0]
    xopt = opt.optimize(x)[0]
    minf = opt.last_optimum_value() 
    
    if verbose:
        print('')
        print('Final Optimised Beta = %0.4f' % xopt)
        print('Difference between actual and modelled Depth is %0.2f m' % minf)
        #print ''
    return opt, xopt, minf


# -------------------------------------------------------- 

def run_optimisation_for_point_using_scipy(prs, pre, psedThick, DEPTH, max_iterations):

    def obj_f(x):
        pbeta = x
        # when running optimisation, assume that subsidence is being computed for present-day
        time=0.
        TS = evaluate_subsidence_at_time(prs, pre, pbeta, psedThick, DEPTH, time)
        opt_eval =  np.abs(TS - (DEPTH + psedThick))
        return opt_eval

    # run scipt optimisation using bounds of 0 and 10, user-defined max iterations.
    # initial guess for beta is one.
    res = minimize(obj_f,1.0,method='COBYLA',bounds=(0,10),options={'maxiter':max_iterations})

    return res['x'],res['fun']


# --------------------------------------------------------
# --- Iterate over arrays to determine optimised beta factors
def run_optimisation_for_dataset(point_lats, 
                                 point_lons, 
                                 RiftStart, 
                                 RiftEnd, 
                                 SedThick, 
                                 Depth,
                                 max_iterations=20,
                                 verbose=True):

    out_opt = []
    out_xopt = []
    out_minf = []
    out_lat = []
    out_lon = []
    out_prs = []
    out_pre = []
    out_psedThicks = []
    out_depths = []

    #fileout_combined = []
    for i, (y, x, pRiftStart, pRiftEnd, pSedThick, pDepth) in enumerate(zip(point_lats, point_lons, RiftStart, RiftEnd, SedThick, Depth)):
        #print("xy: %f %f" % (x,y))
        #print 'Sediment Thickness = %0.6f' % pSedThick
        coordinates = (x,y)
        opta, xopta, minfa = run_optimisation_for_point(pRiftEnd, pRiftStart, pSedThick+0.01, pDepth, max_iterations, verbose)
        out_lat.append(y)
        out_lon.append(x)
        out_opt.append(opta)
        out_xopt.append(xopta)
        out_minf.append(minfa)
        out_prs.append(pRiftStart)
        out_pre.append(pRiftEnd)
        out_psedThicks.append(pSedThick)
        out_depths.append(pDepth)
        #print 'done %0.2f percent' % (100*float(i)/float(len(point_lats)))
    
    return out_lat, out_lon, out_opt, out_xopt, out_minf, out_prs, out_pre, out_psedThicks, out_depths


# main function for stage 1
def generate_rifting_model(COBterrane_file, rotation_file, isocob_features,
                           etopo_file, sedimentthickness_file, 
                           lonmin=-180., lonmax=180., latmin=-90., latmax=90., sampling_factor=1, verbose=True):

    # get today's date in YYYYMMDD format - for creating the output file
    today = time.strftime("%Y%m%d")

    # ---------------------- Read in files ---------------------------------- 
    # --- Read in relevant files to pygplates
    print("... Importing input files")
    cobter = pygplates.FeatureCollection(COBterrane_file)
    cobter = force_polygon_geometries(cobter)
    rotation_model = pygplates.RotationModel(rotation_file)
    cob_lines = pygplates.FeatureCollection(isocob_features)

    # -------------------------------------------------------- 
    # ---- Get valid time from isocob file
    print("... Getting rift times from isocobs")

    cob_lines_present = []   # create an empty array to add points to
    for cob in cob_lines:
        if cob.get_valid_time()[1]<=0:
            cob_lines_present.append(cob)

    # -------------------------------------------------------- 
    # Get bathymetry within bounding box as point data
    subsidence_points_lon,subsidence_points_lat,subsidence_points_z = grd2multipoint(etopo_file,cobter,rotation_model,
                                                                                     lonmin,lonmax,latmin,latmax,sampling_factor)

    # -------------------------------------------------------- 
    # Get Rift Start/End times at points by interpolation from IsoCOB properties
    pts_lon,pts_lat,pts_re,pts_rs = isocob_rift_times(cob_lines_present,rotation_model)

    # interpolate the 'rift end' ages onto chosen grid points
    d,l = sampleOnSphere(np.hstack(pts_lon),
                         np.hstack(pts_lat), 
                         np.hstack(pts_re),
                         subsidence_points_lon,
                         subsidence_points_lat,
                         n=4)
    interp_re = np.hstack(pts_re).ravel()[l]

    # interpolate the 'rift start' ages onto chosen grid points
    d,l = sampleOnSphere(np.hstack(pts_lon),
                         np.hstack(pts_lat), 
                         np.hstack(pts_rs),
                         subsidence_points_lon,
                         subsidence_points_lat,
                         n=4)
    interp_rs = np.hstack(pts_rs).ravel()[l]

    # -------------------------------------------------------- 
    # --- To do backstripping, need to load in a sediment thickness grid
    print("... Getting sediment thickness grid")

    # Then, sample the sediment thickness onto the same points that have been isolated by the previous steps
    ds_disk2 = xr.open_dataset(sedimentthickness_file)

    sedThick = ds_disk2['z']
    sedThick = sedThick.sel(lon=slice(lonmin,lonmax)).sel(lat=slice(latmin,latmax))[::sampling_factor,::sampling_factor]
    #coord_keys = sedThick.coords.keys()
    coord_keys = [key for key in sedThick.coords.keys()]
    sedThickX, sedThickY = np.meshgrid(sedThick.coords[coord_keys[0]].data, 
                                       sedThick.coords[coord_keys[1]].data)

    d,l = sampleOnSphere(sedThickX.flatten(),
                         sedThickY.flatten(),
                         sedThick.data.flatten(),
                         subsidence_points_lon,
                         subsidence_points_lat,
                         n=4)

    sedThick_points = sedThick.data.flatten().ravel()[l]
    sedThick_points = sedThick_points * 1000.   # convert Sediment Thickness points from km to m

    # Multiply bathymetry by -1 to get positive depths
    subsidence_points_z = subsidence_points_z * -1

    # --------------------------------------------------------
    # non-linear optimisation to determine the beta value that best explains the present combination of bathymetry
    # and sediment thickness, given that we know the rift end time
    max_iterations = 20

    # --------------------------------------------------------
    out_lat, out_lon, out_opt, out_xopt, out_minf, out_pres, out_pree, out_psedThicks, out_depths = \
        run_optimisation_for_dataset(subsidence_points_lat, 
                                     subsidence_points_lon, 
                                     interp_rs, 
                                     interp_re, 
                                     sedThick_points.tolist(), 
                                     subsidence_points_z.tolist(),
                                     max_iterations,
                                     verbose)

    # --------------------- Write output file --------------------------------
    tmp = np.vstack((out_lon, out_lat, out_depths, out_psedThicks, out_pres, out_pree, out_xopt, out_minf))
    header_add="Longitude Latitude Bathymetry_m SedimentThickness_m RiftStart_Ma RiftEnd_Ma OptimisedBetaFactor DepthMismatch_m"
    format='%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.6f\t%0.6f'
    
    out_file_name = 'subsidenceinfo_' + str(today) + '.txt'
    np.savetxt(out_file_name, tmp.T, fmt=format, header=header_add)
    print("Done!!")
    
    return out_file_name



###################################################################
# Functions for stage 2
###################################################################

def passive_margin_point_selection(static_polygon_file, passivemarginfile, rotation_file,
                                   longitude, latitude, depth, sedthickness,
                                   rift_start, rift_end, beta):

    rotation_model = pygplates.RotationModel(rotation_file)
    static_polygon_features = pygplates.FeatureCollection(static_polygon_file)
    passive_margin_polygons = pygplates.FeatureCollection(passivemarginfile)

    # because we need to reconstruct the points, cookie cut them to the static polygons for the selected model

    # create a multipoint from the loaded lat,lons
    input_points =  pygplates.MultiPointOnSphere(zip(latitude.flatten(),longitude.flatten())).to_lat_lon_point_list()

    # Polygon test part 1, make a list of plate ids for each point in the originally loaded txt file

    # first iterate over each polygon, to make a list of plate ids that are passed as 
    # the appropriate mapping (proxy) value to the 'find_polygons' function
    static_polygons = []
    static_polygon_plate_ids = []
    for static_polygon_feature in static_polygon_features:
        plate_id = static_polygon_feature.get_reconstruction_plate_id()
        spolygon = static_polygon_feature.get_geometry()

        static_polygons.append(spolygon)
        static_polygon_plate_ids.append(plate_id)

    # This array lists plate_ids for all input points
    input_point_plate_ids = points_in_polygons.find_polygons(input_points,static_polygons,static_polygon_plate_ids)

    # Polygon test part 2

    # Load a file that contains polygons defining the extent of passive margin regions of interest
    # (put another way, it will exclude regions that we are not interested in and so do not want
    # to calculate rift-related subsidence for)

    # do a point in polygon test to isolate only those points which are within the passive margin polygons extent
    reconstructed_passive_margin_polygons = []
    pygplates.reconstruct(passive_margin_polygons, rotation_model, reconstructed_passive_margin_polygons, 0)
    rpolygons = []
    for polygon in reconstructed_passive_margin_polygons:
        if polygon.get_reconstructed_geometry():
            rpolygons.append(polygon.get_reconstructed_geometry())

    # polygons_containing_points is a list of polygons that each point is in, or 'None' if it is 
    # not within any polygon, so can be used to determine whether to keep or exclude points
    polygons_containing_points = points_in_polygons.find_polygons(input_points, rpolygons)


    # make empty arrays in which to store points that are within the passive margin polygons
    lat2 = []
    lon2 = []
    depth2 = []
    sedthickness2 = []
    rift_start2 = []
    rift_end2 = []
    beta2 = []
    plate_id2 = []

    # iterate over all points in input file, 
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

    return lat2,lon2,depth2,sedthickness,rift_start2,rift_end2,beta2,plate_id2,clip_points



# Cell defining functions for number-crunching loop
def get_paleobathymetry_snapshot(latitude, longitude, 
                                 rift_start, rift_end, 
                                 beta, depth, sedthickness, 
                                 output_directory, recon_time, 
                                 rotation_file, anchor_plate_id, 
                                 clip_points_list, points_grouped_by_plate_id, 
                                 sampling, sedimentation_mode):
    
    rotation_model = pygplates.RotationModel(rotation_file)
    clip_points = pygplates.MultiPointOnSphere(clip_points_list)
    
    print('Working on time %0.2fMa...' % recon_time)
    # evaluate subsidence
    paleobathymetry = []
    bsmt = []
    riftend = []
    riftstart = []
    sedthick = []
    equal = []
    beta_out = []
    #count = 0

    for plat, plon, prs, pre, pbeta, pBathy, psedThick in zip(latitude, longitude, rift_start, rift_end, beta, depth, sedthickness):
    
        # if the reconstruction time is greater than the rift start, no subsidence
        if recon_time>=prs:
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
            
            if sedimentation_mode is 'Constant':
                
                # determine amount of sediment that would have accumulated by this time, based on constant rate
                # of accumulation since rift start time
                time_fraction = (float(prs)-float(recon_time))/float(prs)
                
                total_uncompacted_sediment_thickness = decompact_sediment_thickness(psedThick,
                                                                                    DEFAULT_SURFACE_POROSITY,
                                                                                    DEFAULT_POROSITY_EXP_DECAY)
                
                uncompacted_sediment_thickness = total_uncompacted_sediment_thickness * time_fraction
                
                psedThick_at_time = compact_sediment_thickness(uncompacted_sediment_thickness,
                                                               DEFAULT_SURFACE_POROSITY, 
                                                               DEFAULT_POROSITY_EXP_DECAY) + 0.001
                  
            #elif sedimentation_mode is 'Keep_Pace':
            else:
                psedThick_at_time = psedThick
                
            
            # if beta is nan, then the subsequent functions won't work. Assume this
            # is because TTS is zero
            if np.isnan(pbeta):
                bsmt_depth = 0.
            else:
                bsmt_depth = evaluate_subsidence_at_time(prs,pre,pbeta,psedThick_at_time,pBathy,recon_time)
                #bsmt_depth = evaluate_subsidence_at_time(prs,pre,pbeta,psedThick_at_time,0.,recon_time)
                #print bsmt_depth
            
            # also handle cases where pre and prs are the same (which shouldn't 
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
            
            # calculate paleobathymetry, set to zero if negative (e.g. where sediment thickness overestimated)
            if np.less(float(bsmt_depth),float(psedThick_at_time)):
                paleobathymetry.append(0.)
            else:
                paleobathymetry.append(bsmt_depth - psedThick_at_time)  


    recon_point_lons,recon_point_lats = reconstruct_point_groups(clip_points,
                                                                 points_grouped_by_plate_id,
                                                                 rotation_model,
                                                                 recon_time,
                                                                 anchor_plate_id)

    # write out data into multi-column ascii file
    write_xyz_file('out_tmp/tmp_%0.2f.xyz' % recon_time, zip(recon_point_lons, 
                                                             recon_point_lats, 
                                                             paleobathymetry, 
                                                             bsmt, 
                                                             riftstart, 
                                                             riftend, 
                                                             sedthick, 
                                                             equal, 
                                                             beta_out))
    
    # pre-processing step (block median) - test if it makes output grid better??
    #call_system_command(['gmt', 'blockmedian', 'out_tmp/tmp_%0.2f.xyz' % recon_time,
    #                     '-Rg', '-I%0.8fd' % sampling, '-i0,1,2', 
    #                     '>', 'out_tmp/tmp2_%0.2f.xyz' % recon_time])    
    
    call_system_command(['gmt', 'nearneighbor', 
                         'out_tmp/tmp_%0.2f.xyz' % recon_time, 
                         '-G%s/paleobathy_%0.2f.nc' % (output_directory,recon_time), 
                         '-Rg', '-I%0.8fd' % sampling, 
                         '-N4/1','-S%0.8fd' % sampling, '-i0,1,2'])
    call_system_command(['gmt', 'nearneighbor', 
                         'out_tmp/tmp_%0.2f.xyz' % recon_time, 
                         '-G%s/bsmt_%0.2f.nc' % (output_directory,recon_time), 
                         '-Rg', '-I%0.8f' % sampling, 
                         '-N4/1','-S%0.8fd' % sampling, '-i0,1,3'])  
    call_system_command(['gmt', 'nearneighbor', 
                         'out_tmp/tmp_%0.2f.xyz' % recon_time, 
                         '-G%s/sedthick_%0.2f.nc' % (output_directory,recon_time), 
                         '-Rg', '-I%0.8f' % sampling, 
                         '-N4/1','-S%0.8fd' % sampling, '-i0,1,6'])
    
    # clean up
    call_system_command(['rm', 'out_tmp/tmp_%0.2f.xyz' % recon_time])
    #call_system_command(['rm', 'out_tmp/tmp2_%0.2f.xyz' % recon_time])


    
