# generate trainging data from netCDF data
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, ErrorCode
import numpy as np
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset
from progressbar import ProgressBar
import cmocean
import sys

from data_processes import *
from parcel_plots import *

####################
# Custom Particles #
####################
# add particle age
class ageing_particle(JITParticle):
    # add age of particle
    age = Variable('age', dtype=np.int32, initial=0.)

####################
# Particle Kernels #
####################
# kernle to delete particles out of range
def deleteParticle(particle, fieldset, time):
    particle.delete()

# particle ageing kernel
def ageingParticle(particle, fieldset, time):
    particle.age += 1

# kill old particle
def killOld(particle, fieldset, time):
    if particle.age >= 365:
        particle.delete()

########

########
def particle_generator_region_filterV(region, ROMS_dir, V_threshold, time_name='ocean_time'):
    """
    Will need to update this code to use dimensions and variable dictionary
    """
    print('Filtering generation zone based with V less than or equal to '+str(V_threshold))
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
    for i in range(0, len(file_ls)):
        fh = Dataset(ROMS_dir + file_ls[i], mode='r')
        # extract the time lats and lons and V
        nc_time = fh.variables['ocean_time'][:]
        nc_lats = fh.variables['lat_rho'][:]
        nc_lons = fh.variables['lon_rho'][:]
        nc_v = fh.variables['vbar'][:]

        # cycle through each time instance and extract lats and lons and ravel
        # into 1 dimensional arrays after filtering
        for j in range(0, nc_v.shape[0]):
            array_lons = nc_lons[nc_v[j] <= V_threshold].ravel()
            # check if zero and if so skip
            if len(array_lons) == 0:
                continue
            # now lats
            array_lats = nc_lats[nc_v[j] <= V_threshold].ravel()
            # filter out lats and lons outside our region
            bools = (region[0] <= array_lons) & (array_lons <= region[1]) & (region[2] <= array_lats) & (array_lats <= region[3]) 
            array_lons = array_lons[bools]
            array_lats = array_lats[bools]
            array_time = np.repeat(nc_time[j], len(array_lats))
            # concat
            lons = np.concatenate((lons, array_lons))
            lats = np.concatenate((lats, array_lats))
            time = np.concatenate((time, array_time))

    # check if any particles
    if len(time) == 0:
        sys.exit('Error: No V values less than or equal to '+str(V_threshold)+' ms-1 found within the generation region..')
    # convert from days from 1990 to days from origin
    time = (time - time[0])

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
    pset = ParticleSet(fieldset=fieldset, pclass=ageing_particle, lon=lons, lat=lats, repeatdt=sampledt)
    # set ageing kernel
    kernels = ageingParticle + pset.Kernel(AdvectionRK4)
    # collect data
    pset.execute(kernels,
                 runtime=runtime,  # runtime controls the interval of the plots
                 dt=sampledt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                 output_file=pset.ParticleFile(name=outputfn, outputdt=sampledt))

#######################################################
# Get trainign data through lagrangian particle model #
#######################################################
def particle_positions_filterV(filenames, variables, dimensions, indicies, generation_region, ROMS_dir, runtime, sampledt, outputfn, V_threshold):
    """
    Generate a netCDF file of particle psoitions
    """
    # generate feildset from netCDF files
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies)
    # make particle generation zone
    lons, lats, time = particle_generator_region_filterV(generation_region, ROMS_dir, V_threshold)
    # make particle set
    pset = ParticleSet(fieldset=fieldset, pclass=ageing_particle, lon=lons, lat=lats, time=time)
    # particle seed check
    print(str(len(time))+' paticles seeds generated')
    print(time)
    # set ageing kernel
    kernels = ageingParticle + pset.Kernel(AdvectionRK4)
    # collect data
    pset.execute(kernels,
                 runtime=runtime,  # runtime controls the interval of the plots
                 dt=sampledt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteParticle},
                 output_file=pset.ParticleFile(name=outputfn, outputdt=sampledt))

#####################
# Particle training #
#####################
def particle_training(particlefn, ROMS_dir, ouputfn):
    """
    Generate training data from particle position netCDF file
    """
    # read in data
    fh = Dataset(particlefn, mode='r')
    # build particle arrays
    time = fh.variables['time'][:]
    lats = fh.variables['lat'][:]
    lons = fh.variables['lon'][:]
    # for each time cycle through and get all lats and lons
    # calculate timesteps
    t_start = np.unique(time).min()
    t_step = np.unique(time)[1]
    t_end = np.unique(time).max()
    t_series = np.arange(t_start, t_end, t_step)
    # coordinate series
    lon_series = [lons[time == t].data for t in t_series]
    lat_series = [lats[time == t].data for t in t_series]

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
    for cnt in range(runlength):
        # First plot the particles
        pset.show(savefile=out_dir + 'particles'+str(cnt).zfill(3), field=fieldset.V, land=True, vmin=vmin, vmax=vmax)

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
    pset = ParticleSet(fieldset=fieldset, pclass=ageing_particle, lon=lons, lat=lats, repeatdt=repeatdt)

    # create full domain
    if domain == 'full':
        domain = {'N':fieldset.V.lat.max(), 'S':fieldset.V.lat.min()-0.2,
                  'W':fieldset.V.lon.min()-0.2, 'E':fieldset.V.lon.max()+0.2}

    # set ageing kernel
    kernels = ageingParticle + pset.Kernel(AdvectionRK4)

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
    sampledt, out_dir, runlength, domain, vmin, vmax, timeorigin, ROMS_dir, V_threshold, cmap=cmocean.cm.speed, plot_type='field-only'):
    """
    Generate animation using custom functions
    """
    # generate feildset from netCDF files
    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies,  deferred_load=False)
    # make particle generation zone
    lons, lats, time = particle_generator_region_filterV(generation_region, ROMS_dir, V_threshold)
    # make particle set
    pset = ParticleSet(fieldset=fieldset, pclass=ageing_particle, lon=lons, lat=lats, time=time)

    print(str(len(time))+' paticles seeds generated')
    print(time)

    # create full domain
    if domain == 'full':
        domain = {'N':fieldset.V.lat.max(), 'S':fieldset.V.lat.min()-0.2,
                  'W':fieldset.V.lon.min()-0.2, 'E':fieldset.V.lon.max()+0.2}

    # set ageing kernel
    kernels = ageingParticle + pset.Kernel(AdvectionRK4)

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
        
        fig.savefig(out_dir+'particles'+str(i).zfill(3))

        # stop if at limit (no more advection)
        if i == runlength:
            break
        # Then advect the particles for 1 day
        pset.execute(kernels,
                     runtime=sampledt,  # runtime controls the interval of the plots
                     dt=sampledt,
                     recovery={ErrorCode.ErrorOutOfBounds: deleteParticle})

    print('PNGs can be joined with Image Magick.')
    print('convert *.png particle_animation.gif')













