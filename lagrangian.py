# generate trainging data from netCDF data
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, ErrorCode
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset
from progressbar import ProgressBar
import cmocean
import sys
from operator import attrgetter
import random
import numba
import functools
from scipy.spatial import cKDTree
import math
import pickle

from data_processes import *
from parcel_plots import *

####################
# Custom Particles #
####################
# add particle age
class oceancc_particle(JITParticle):
    # add age of particle
    age = Variable('age', dtype=np.int32, initial=0.)
    stuck = Variable('stuck', dtype=np.int32, initial=0.)
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.

####################
# Particle Kernels #
####################
# Delete particles out of bounds
def deleteParticle(particle, fieldset, time):
    particle.delete()

# Particle ageing kernel
def ageingParticle(particle, fieldset, time):
    particle.age += 1

# Stuck paerticle kernel
def stuckParticle(particle, fieldset, time):
    if (particle.prev_lon == particle.lon) and (particle.prev_lat == particle.lat):
        particle.stuck += 1
    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

# Delete old particles
def killSwitch(particle, fieldset, time):
    if particle.age >= 365:
        particle.delete()
    elif particle.stuck >= 20:
        particle.delete()

######################################################################
# Create particle generation region but filter by southward velocity #
######################################################################
def particle_generator_region_filterV(region, ROMS_dir, V_threshold, time_name='ocean_time', maxParticlesStep=None, returnSteps=True):
    """
    Will need to update this code to use dimensions and variable dictionary to make
    it more universal
    """
    print('Filtering generation zone for V less than or equal to '+str(V_threshold)+' ms-1')
    file_ls = [f for f in listdir(ROMS_dir) if isfile(join(ROMS_dir, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)
    # empty arrays for storage
    lons = []
    lats = []
    time = []
    # progress
    # pbar = ProgressBar(max_value=len(file_ls))
    # now cycle through the ROMS data and get lat and lon
    # values based on the southward velocity threshold
    steps = 0
    for i in range(0, len(file_ls)):
        fh = Dataset(ROMS_dir + file_ls[i], mode='r')
        # extract the time lats and lons and V
        nc_time = fh.variables['ocean_time'][:]
        nc_lats = fh.variables['lat_v'][:]
        nc_lons = fh.variables['lon_v'][:]
        nc_v = fh.variables['vbar'][:]

        # lists to hold the particle seeds so we don't need to 
        # call np.concatenate on each iteration
        lons_capture = [np.nan]*len(nc_time)
        lats_capture = [np.nan]*len(nc_time)
        time_capture = [np.nan]*len(nc_time)

        # cycle through each time instance and extract lats and lons and ravel
        # into 1 dimensional arrays after filtering
        for j in range(0, nc_v.shape[0]):
            array_lons = nc_lons[nc_v[j] <= V_threshold].ravel()
            # check if zero and if so skip
            if len(array_lons) == 0:
                steps += 1
                continue
            # now lats
            array_lats = nc_lats[nc_v[j] <= V_threshold].ravel()
            # filter out lats and lons outside our region
            bools = (region[0] <= array_lons) & (array_lons <= region[1]) & (region[2] <= array_lats) & (array_lats <= region[3]) 
            array_lons = array_lons[bools]
            array_lats = array_lats[bools]
            array_time = np.repeat(nc_time[j], len(array_lats))

            # downsample particles if needed
            if maxParticlesStep is not None:
             if len(array_time) > maxParticlesStep:
                # select random particles
                filter_idx = random.sample(range(0, len(array_time)), maxParticlesStep)
                filter_idx.sort()
                # filter particle arrays
                array_lons = array_lons[filter_idx]
                array_lats = array_lats[filter_idx]
                array_time = array_time[filter_idx]

            # add to capture lists
            lons_capture[j] = array_lons
            lats_capture[j] = array_lats
            time_capture[j] = array_time
            # get timesteps
            steps += 1
                

        # concat collected arrays (Much faster when not done in loop)
        lons = np.concatenate((lons, np.concatenate(lons_capture)))
        lats = np.concatenate((lats, np.concatenate(lats_capture)))
        time = np.concatenate((time, np.concatenate(time_capture)))

    # check if any particles
    if len(time) == 0:
        sys.exit('Error: No V values less than or equal to '+str(V_threshold)+' ms-1 found within the generation region')
    # convert from seconds from 1990 to seconds since origin
    time = (time - time[0])

    if returnSteps:
        return lons, lats, time, steps
    else:
        return lons, lats, time

#########################################################
# Create particle generation region from lon-lat arrays #
#########################################################
def particle_generator_region(region, lon_array, lat_array):
    lons = lon_array.ravel()
    lats = lat_array.ravel()
    bools = (region[0] <= lons) & (lons <= region[1]) & (region[2] <= lats) & (lats <= region[3]) 
    lons = lons[bools]
    lats = lats[bools]

    return lons, lats

##################################################################
# Legacy - Create particle generation region from lon-lat arrays #
##################################################################
def legacy_particle_generator_time(region, lon_array, lat_array, iterations, delta):
    """
    Legacy function - deprecated due to presence of the `repeatdt`
    argument in the `ParticleSet()` function. 
    """
    lons = lon_array.ravel()
    lats = lat_array.ravel()
    bools = (region[0] <= lons) & (lons <= region[1]) & (region[2] <= lats) & (lats <= region[3]) 
    lons = lons[bools]
    lats = lats[bools]

    time = np.arange(0, iterations) * delta.total_seconds()
    time = np.repeat(time, len(lons))

    lons = np.tile(lons, iterations)
    lats = np.tile(lats, iterations)

    return lons, lats, time

#######################################################
# Get trainign data through lagrangian particle model #
#######################################################
def particle_positions(filenames, variables, dimensions, indicies, generation_region, repeatdt, runtime, sampledt, outputfn):
    """
    Generate a netCDF file of particle psoitions
    """
    # generate feildset from netCDF files
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies)
    # make particle generation zone
    lons, lats = particle_generator_region(generation_region, fieldset.V.lon, fieldset.V.lat)
    pset = ParticleSet(fieldset=fieldset, pclass=oceancc_particle, lon=lons, lat=lats, repeatdt=sampledt)
    # set ageing kernel
    kernels = pset.Kernel(ageingParticle) + pset.Kernel(stuckParticle) + pset.Kernel(killSwitch) + pset.Kernel(AdvectionRK4)
    # collect data
    pset.execute(kernels,
                 runtime=runtime,  # runtime controls the interval of the plots
                 dt=sampledt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                 output_file=pset.ParticleFile(name=outputfn, outputdt=sampledt))

#######################################################
# Get trainign data through lagrangian particle model #
#######################################################
def particle_positions_filterV(filenames, variables, dimensions, indicies, generation_region, ROMS_dir, runtime, sampledt, 
    outputfn, V_threshold, maxParticlesStep=None):
    """
    Generate a netCDF file of particle psoitions
    """
    # generate feildset from netCDF files
    print('Loading the fieldset..')
    if indicies is not None:
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies, deferred_load=False)
    else: 
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, deferred_load=False)
    # make particle generation zone
    lons, lats, time, steps = particle_generator_region_filterV(generation_region, ROMS_dir, V_threshold, maxParticlesStep=maxParticlesStep)
    print(str(len(time))+' paticle seeds generated')
    print(time)

    # full 20 year run
    if runtime == 'full':
        runtime = steps - 1

    # convert runtime to days 
    runlength = runtime 
    runtime = timedelta(days=runtime)

    # prior to making the particle set mught be worth filtering based on the runlength
    lons = lons[np.isin(time, np.unique(time)[0:runlength+1])]
    lats = lats[np.isin(time, np.unique(time)[0:runlength+1])]
    time = time[np.isin(time, np.unique(time)[0:runlength+1])]
    print('Filtered to '+str(len(time))+' seeds based on runlength of '+str(runlength))
    print(time)

    # make particle set
    print('\nGenerating particle set...')
    pset = ParticleSet(fieldset=fieldset, pclass=oceancc_particle, lon=lons, lat=lats, time=time)
    # set ageing kernel
    kernels = pset.Kernel(ageingParticle) + pset.Kernel(stuckParticle) + pset.Kernel(killSwitch) + pset.Kernel(AdvectionRK4)

    # collect data
    output_file = pset.ParticleFile(name=outputfn, outputdt=sampledt)
    pset.execute(kernels,
                 runtime=runtime,
                 dt=sampledt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                 output_file=output_file)
    output_file.export()

#####################
# Particle training #
#####################
def particle_training(particle_trajectories_fn, ROMS_file, outputfn, timeorigin_traj=datetime(1994, 1, 1, 12), 
    timeorigin_ROMS=datetime(1990, 1, 1), animate=False, plot_type='grid_scatter', anim_out='anim/grid_selection/', 
    spinup=timedelta(days=364), coord_rounding=2, method='kdtree', cell_radius=15):
    """
    Generate training data from particle position netCDF file
    This currently does not support hydrodynamic models with split netCDF
    file sources 
    """
    # read in particle trajectory data
    fh = Dataset(particle_trajectories_fn, mode='r')
    # build particle arrays
    # 'p' stands for 'particle'
    print('Loading particle trajectories..')
    p_time = fh.variables['time'][:]
    p_lats = fh.variables['lat'][:]
    p_lons = fh.variables['lon'][:]
    # for each time cycle through and get all lats and lons
    # calculate timestep series (1d array)
    print('Constructing particle timeseries..')
    p_time_unique = np.unique(p_time) # too slow to repeat this step
    p_t_start = p_time_unique.min()
    p_t_step = p_time_unique[1] - p_time_unique[0]
    p_t_end = p_time_unique.max()
    p_t_series = np.arange(p_t_start, p_t_end, p_t_step)

    # load in the ROMS file
    fh = Dataset(ROMS_file, mode='r')
    ROMS_lats = fh.variables['lat_rho'][:]
    ROMS_lons = fh.variables['lon_rho'][:]
    ROMS_time = fh.variables['ocean_time'][:]

    # if making map then setup basemap and the grids in advance so we don't have to 
    # reconstruct the map on each iteration
    if (animate and (plot_type == 'density_map')):
        domain = {'N':ROMS_lats.max(), 'S':ROMS_lats.min()-0.2,
                  'W':ROMS_lons.min()-0.2, 'E':ROMS_lons.max()+0.2}
        m = make_map(domain)

    # convert the particle time series into the same time format as the ROMS model
    p_t_series = p_t_series + (timeorigin_traj - timeorigin_ROMS).total_seconds()
    p_time = p_time + (timeorigin_traj - timeorigin_ROMS).total_seconds()

    # convert spinup to seconds and add to pset time
    spinup_days = spinup.days
    spinup = spinup.total_seconds()
    spinup = p_t_series[0] + spinup

    # extract landmask to filter sampling
    land_mask = fh.variables['temp'][0, 0].mask
    # get coordinates of True values
    land_mask = np.where(land_mask)
    # make into tuples
    land_mask = tuple(zip(land_mask[0], land_mask[1]))

    # arrays to hold data
    temp_capture = [np.nan]*len(p_t_series)
    salt_capture = [np.nan]*len(p_t_series)
    lats_capture = [np.nan]*len(p_t_series)
    time_capture = [np.nan]*len(p_t_series)
    class_capture = [np.nan]*len(p_t_series)

    # construct k-dimensional tree for quick lookups
    if method == 'kdtree':
        # Read latitude and longitude from file into numpy arrays
        latvals = ROMS_lats[:] * math.pi/180.0
        lonvals = ROMS_lons[:] * math.pi/180.0
        clat,clon = np.cos(latvals), np.cos(lonvals)
        slat,slon = np.sin(latvals), np.sin(lonvals)
        # Build kd-tree from big arrays of 3D coordinates
        triples = list(zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat)))
        kdt = cKDTree(triples)

    # make progress bar
    pbar = ProgressBar(max_value=len(p_t_series)-spinup_days)
    print('Extracting data from hydrodynamic model..\n')

    # iterate through timestep series and use particle coordinates to create a
    # density contour.
    i = 0
    for pset_idx in numba.prange(0, len(p_t_series)):
        # first check the spin up
        if p_t_series[pset_idx] < spinup:
            continue
        # select the particle set at the right timestamp
        pset_slice = np.where(p_time == p_t_series[pset_idx])
        pset_lons = p_lons[pset_slice]
        pset_lats = p_lats[pset_slice]
        # filter out duplicate particle positions (round based on grid size)
        pset_lons = [round(x, coord_rounding) for x in pset_lons]
        pset_lats = [round(x, coord_rounding) for x in pset_lats]
        pset_coords = tuple(zip(pset_lons, pset_lats))
        pset_coords = list(set(pset_coords))
        # now find the grid cell positions for the particle positions (eta_rho, xi_rho)
        if method == 'kdtree':
            grid_dim_array = [kdtree_process(kdt, pcoord[0], pcoord[1], ROMS_lats.shape) for pcoord in pset_coords]
        elif method == 'tunnel_distance_parallel':
            grid_dim_array = tunnel_parallelise(pset_coords, ROMS_lats, ROMS_lons)
        elif method == 'tunnel_distance':
            grid_dim_array = [tunnel_fast(ROMS_lons, ROMS_lats, pcoord[0], pcoord[1]) for pcoord in pset_coords]
        else:
            print('Error: unknown method')
            #return
        # remove duplicates
        grid_dim_array = list(set(grid_dim_array))
        # make integers
        grid_dim_array = [(int(x[0]), int(x[1])) for x in grid_dim_array]
        # now sample an equal number of random points with no particles
        # first set up cells that are not allowed within N cells
        forbidden_cells = Nneighbours(grid_dim_array, ROMS_lats.shape, cell_radius)
        # make random indicies
        rand_set_done = False
        n = 2
        while not rand_set_done:
            # NOTE: parallelise these removals
            rand_eta = np.random.randint(ROMS_lats.shape[0], size=n*len(grid_dim_array))
            rand_xi = np.random.randint(ROMS_lats.shape[1], size=n*len(grid_dim_array))
            # remove duplicates and create tuple set
            rand_set = list(set(tuple(zip(rand_eta, rand_xi))))
            # remove any that contain particles are on land or near particle cells
            rand_set = [coord for coord in rand_set if coord not in land_mask]
            rand_set = [coord for coord in rand_set if coord not in forbidden_cells]
            # check if set is long enough
            if len(rand_set) >= len(grid_dim_array):
                rand_set_done = True
            else:
                n += 1
        # now take the first N tuples
        rand_set = rand_set[:len(grid_dim_array)]
        # extract the data from the model
        # split the tuples
        grid_dim_array = list(zip(*grid_dim_array))
        rand_set = list(zip(*rand_set))
        # find the ROMS ocean_time slice index
        ROMS_idx = int(np.where(ROMS_time == p_t_series[pset_idx])[0])
        # make dimensions
        eta = grid_dim_array[0]+rand_set[0]
        xi = grid_dim_array[1]+rand_set[1]
        # retrive the data
        temp_capture[i] = fh.variables['temp'][ROMS_idx, 0][eta, xi].data
        salt_capture[i] = fh.variables['salt'][ROMS_idx, 0][eta, xi].data
        # Lats are float64, so get lats and convert from float64 to float32 (64 is too slow when indexing)
        lats = fh.variables['lat_rho'][:]
        lats = np.float32(lats)
        lats_capture[i] = lats[eta, xi].data
        # make datetimes
        time_capture[i] = [timeorigin_ROMS + timedelta(seconds=p_t_series[pset_idx])]*(2*len(grid_dim_array[0]))
        # make class array
        class_capture[i] = ['A']*len(grid_dim_array[0]) + ['B']*len(rand_set[0])

        # animate
        if animate:
            fig = plot_grid_selection(eta, xi, ['orange']*len(grid_dim_array[0])+['blue']*len(rand_set[0]), ylim=(0, ROMS_lats.shape[0]), xlim=(0, ROMS_lats.shape[1]))
            fig.savefig(anim_out+'grid_selection'+str(i).zfill(4))

        # permutate
        i += 1
        # update progress
        pbar.update(i)
        
    # save the datasets
    print('\nMerging the captured data...')
    # merge lists and remove nans
    temp_capture = [x for x in np.concatenate(temp_capture, axis=None) if ~np.isnan(x)]
    salt_capture = [x for x in np.concatenate(salt_capture, axis=None) if ~np.isnan(x)]
    lats_capture = [x for x in np.concatenate(lats_capture, axis=None) if ~np.isnan(x)]
    time_capture = [x for x in np.concatenate(time_capture, axis=None) if isinstance(x, datetime)]
    class_capture = [x for x in np.concatenate(class_capture, axis=None) if x != 'nan']
    # 

    # make pandas dataframe
    print('Producing dataframe...')
    df = pd.DataFrame({
        'time':time_capture,
        'lat':lats_capture,
        'temp':temp_capture,
        'salt':salt_capture,
        'class':class_capture
        })

    # save the dataframe as pickle and csv
    print('Saving training data as pickle..')
    df.to_pickle(outputfn)
    print('Done!')

    """
    Need to remove below code
    """
    # for i in range(0, len(t_series)):
    #     # get iteration
    #     t = t_series[i]
    #     lon = lon_series[i]
    #     lat = lat_series[i]
        
    #     # plot
    #     if animate:
    #         if plot_type == 'density':
    #             fig = particle_density_plot(lon.ravel(), lat.ravel(), xlim=[lon_min, lon_max], ylim=[lat_min, lat_max])
    #         if plot_type == 'density_map':
    #             fig = particle_density_map(m, lon.ravel(), lat.ravel(), xlim=[lon_min, lon_max], ylim=[lat_min, lat_max])
    #         # save figure
    #         fig.savefig(anim_out+'density'+str(i).zfill(4))

########################################
# Animate particles with Ocean Parcels #
########################################
def particle_animation_OP(filenames, variables, dimensions, indicies, generation_region, 
    repeatdt, sampledt, out_dir, runlength, vmin, vmax):
    """
    Generate animation using Ocean Parcel plotting functions
    """
    # generate feildset from netCDF files
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies)
    # make particle generation zone
    lons, lats = particle_generator_region(generation_region, fieldset.V.lon, fieldset.V.lat)
    # make particle set
    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, repeatdt=repeatdt)

    # try vector animation
    for i in range(runlength):
        # First plot the particles
        pset.show(savefile=out_dir + 'particles'+str(i).zfill(3), field=fieldset.V, land=True, vmin=vmin, vmax=vmax)

        if i == runlength-1:
            break

        # Then advect the particles for 1 day
        pset.execute(AdvectionRK4,
                     runtime=sampledt,  # runtime controls the interval of the plots
                     dt=sampledt,
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle})

##############################
# Animate particles (custom) #
##############################
def particle_animation(filenames, variables, dimensions, indicies, generation_region, repeatdt, 
    sampledt, out_dir, runlength, domain, vmin, vmax, timeorigin, cmap=cmocean.cm.speed, plot_type='field-only'):
    """
    Generate animation using custom functions
    """
    # generate feildset from netCDF files
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies,  deferred_load=False)
    # make particle generation zone
    lons, lats = particle_generator_region(generation_region, fieldset.V.lon, fieldset.V.lat)
    # make particle set
    pset = ParticleSet(fieldset=fieldset, pclass=oceancc_particle, lon=lons, lat=lats, repeatdt=repeatdt)

    # create full domain
    if domain == 'full':
        domain = {'N':fieldset.V.lat.max(), 'S':fieldset.V.lat.min()-0.2,
                  'W':fieldset.V.lon.min()-0.2, 'E':fieldset.V.lon.max()+0.2}

    # set ageing kernel
    kernels = pset.Kernel(ageingParticle) + pset.Kernel(stuckParticle) + pset.Kernel(killSwitch) + pset.Kernel(AdvectionRK4)

    # make map
    m = make_map(domain)

    # try vector animation
    for i in range(runlength):
        # get title
        title = str(timeorigin + timedelta(days=1)*i)
        # make plot
        if plot_type == 'field-only':
            fig = plot_field(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat, 
                vmin=vmin, vmax=vmax, cmap=cmap, title=title)
        
        elif plot_type == 'particles-field':
            fig = plot_field_particles(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat,
                vmin=vmin, vmax=vmax, cmap=cmap, pset=pset, title=title)

        elif plot_type == 'particles-field-filter-t0':
            fig = plot_field_particles(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat,
                vmin=vmin, vmax=vmax, cmap=cmap, pset=pset, title=title, filter_t0=True)
        
        fig.savefig(out_dir+'particles'+str(i).zfill(4))

        if i == runlength-1:
            break

        # Then advect the particles for 1 day
        pset.execute(kernels,
                     runtime=sampledt,  # runtime controls the interval of the plots
                     dt=sampledt,
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle})

    print('PNGs can be joined with Image Magick.')
    print('convert *.png particle_animation.gif')

##############################
# Animate particles filterV #
##############################
def particle_animation_filterV(filenames, variables, dimensions, indicies, generation_region, repeatdt, 
    sampledt, out_dir, runlength, domain, vmin, vmax, timeorigin, ROMS_dir, V_threshold, cmap=cmocean.cm.speed, 
    plot_type='field-only', maxParticlesStep=None):
    """
    Generate animation using custom functions
    """
    # generate feildset from netCDF files
    if indicies is not None:
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies,  deferred_load=False)
    else: 
        fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, deferred_load=False)
    # make particle generation zone
    lons, lats, time, steps = particle_generator_region_filterV(generation_region, ROMS_dir, V_threshold, maxParticlesStep=maxParticlesStep)
    print(str(len(time))+' paticle seeds generated')
    print(time)

    # full 20 year runs
    if runlength == 'full':
        runlength = steps

    # prior to making the particle set mught be worth filtering based on the runlength
    lons = lons[np.isin(time, np.unique(time)[0:runlength+1])]
    lats = lats[np.isin(time, np.unique(time)[0:runlength+1])]
    time = time[np.isin(time, np.unique(time)[0:runlength+1])]
    print('Filtered to '+str(len(time))+' seeds based on runlength of '+str(runlength))
    print(time)

    # make particle set
    pset = ParticleSet(fieldset=fieldset, pclass=oceancc_particle, lon=lons, lat=lats, time=time)

    # create full domain
    if domain == 'full':
        domain = {'N':fieldset.V.lat.max(), 'S':fieldset.V.lat.min()-0.2,
                  'W':fieldset.V.lon.min()-0.2, 'E':fieldset.V.lon.max()+0.2}

    # set ageing kernel
    kernels = pset.Kernel(ageingParticle) + pset.Kernel(stuckParticle) + pset.Kernel(killSwitch) + pset.Kernel(AdvectionRK4)
    
    # make map
    m = make_map(domain)

    # try vector animation
    for i in range(runlength):
        # get title
        title = str(timeorigin + timedelta(days=1)*i)
        # make plot
        if plot_type == 'field-only':
            fig = plot_field(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat, 
                vmin=vmin, vmax=vmax, cmap=cmap, title=title)
        
        elif plot_type == 'particles-field':
            fig = plot_field_particles(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat,
                vmin=vmin, vmax=vmax, cmap=cmap, pset=pset, title=title)

        elif plot_type == 'particles-field-filter-t0':
            fig = plot_field_particles(m, field=fieldset.V.data[i], lons=fieldset.V.lon, lats=fieldset.V.lat,
                vmin=vmin, vmax=vmax, cmap=cmap, pset=pset, title=title, filter_t0=True, plot_region=True, region=generation_region)
        
        fig.savefig(out_dir+'particles'+str(i).zfill(4))

        # stop if at limit (no more advection)
        if i == runlength-1:
            break
        # Then advect the particles for 1 day
        pset.execute(kernels,
                     runtime=sampledt,  # runtime controls the interval of the plots
                     dt=sampledt,
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle})

    print('PNGs can be joined with Image Magick.')
    print('convert *.png particle_animation.gif')













