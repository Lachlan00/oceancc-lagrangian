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
def particle_positions_filterV(filenames, variables, dimensions, indicies, generation_region, ROMS_dir, runtime, sampledt, outputfn, V_threshold, maxParticlesStep=None):
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
    # output_file = pset.ParticleFile(name=outputfn, outputdt=sampledt)
    pset.execute(kernels,
                 runtime=runtime,
                 dt=sampledt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                 output_file=pset.ParticleFile(name=outputfn, outputdt=sampledt))
    # output_file.export()

#####################
# Particle training #
#####################
def particle_training(particlefn, ROMS_dir, ouputfn, timeorigin_traj=datetime(1994, 1, 1, 12), timeorigin_ROMS=datetime(1990, 1, 1), 
    animate=False, plot_type='density', anim_out='anim/density/'):
    """
    Generate training data from particle position netCDF file
    """
    # read in particle trajectory data
    fh = Dataset(particlefn, mode='r')
    # build particle arrays
    # 'p' stands for 'particle'
    p_time = fh.variables['time'][:]
    p_lats = fh.variables['lat'][:]
    p_lons = fh.variables['lon'][:]
    # for each time cycle through and get all lats and lons
    # calculate timestep series (1d array)
    p_t_start = np.unique(p_time).min()
    p_t_step = np.unique(p_time)[1]
    p_t_end = np.unique(p_time).max()
    p_t_series = np.arange(p_t_start, p_t_end, p_t_step)
    # coordinate series (list of 2d arrays)
    p_lon_series = [p_lons[p_time == t].data for t in p_t_series]
    p_lat_series = [p_lats[p_time == t].data for t in p_t_series]
    # get particle extent (for plotting)
    p_lon_max = np.array([x.max() for x in p_lon_series]).max()
    p_lon_min = np.array([x.min() for x in p_lon_series]).min()
    p_lat_max = np.array([x.max() for x in p_lat_series]).max()
    p_lat_min = np.array([x.min() for x in p_lat_series]).min()

    # load in the ROMS files
    # get file list
    file_ls = [f for f in listdir(ROMS_dir) if isfile(join(ROMS_dir, f))]
    file_ls = list(filter(lambda x:'.nc' in x, file_ls))
    file_ls = sorted(file_ls)
    # obtain ROMS dimension information
    # load a file
    nc_file = ROMS_dir + file_ls[0]
    fh = Dataset(nc_file, mode='r')
    ROMS_lats = fh.variables['lat_rho'][:]
    ROMS_lons = fh.variables['lon_rho'][:]

    # if making map then setup basemap and the grids in advance so we don't have to 
    # reconstruct the map on each iteration
    if (animate and (plot_type == 'density_map')):
        domain = {'N':ROMS_lat.max(), 'S':lROMS_lat.min()-0.2,
                  'W':ROMS_lon.min()-0.2, 'E':ROMS_lon.max()+0.2}
        m = make_map(domain)

    # iterate through timestep series and use particle coordinates to create a
    # density contour.
    """
    Actually I should iterate through the ROMS model here else there'll be no way
    to know which ncfile to open. I can cross reference which lat/lon arrays using 
    the t_sereis value. 
    """
    # Iterate through the netCDF files and the timesteps
    for i in range(0, len(file_ls)):
        fh = Dataset(ROMS_dir + file_ls[i], mode='r')
        # extract the time
        ROMS_time = fh.variables['ocean_time'][:]

        # cycle through each time instance and extract lats and lons and ravel
        # into 1 dimensional arrays after filtering
        for j in range(0, nc_v.shape[0]):
            test = 1

    """
    Need to remove below code
    """
    for i in range(0, len(t_series)):
        # get iteration
        t = t_series[i]
        lon = lon_series[i]
        lat = lat_series[i]
        
        # plot
        if animate:
            if plot_type == 'density':
                fig = particle_density_plot(lon.ravel(), lat.ravel(), xlim=[lon_min, lon_max], ylim=[lat_min, lat_max])
            if plot_type == 'density_map':
                fig = particle_density_map(m, lon.ravel(), lat.ravel(), xlim=[lon_min, lon_max], ylim=[lat_min, lat_max])
            
            # save figure
            fig.savefig(anim_out+'density'+str(i).zfill(4))

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
        
        fig.savefig(out_dir+'particles'+str(i).zfill(3))

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















