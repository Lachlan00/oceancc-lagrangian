from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, ErrorCode
import numpy as np
from datetime import datetime, timedelta

from data_processes import *
from lagrangian import *

# generate feildset from netCdf files
filenames = "/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/naroom*.nc"
variables = {'U': 'u',
             'V': 'v'}
dimensions = {'lon': 'lon_rho', 'lat': 'lat_rho', 'depth':'s_rho', 'time': 'ocean_time'}
indicies = {'depth':[29]} # surface only
fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, indicies, deferred_load=False)

# show feild
fieldset.V.show(savefile='Another-test')

# experiment with animating fieldset
# particle generation
region = boxmaker(151.5, -35.5, 25)   
lons, lats = particle_generator_region(region, fieldset.V.lon, fieldset.V.lat)
repeatdt = timedelta(days=1)


# run animation
# make particle set
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, repeatdt=repeatdt)
# kernle to delete particles out of range
def DeleteParticle(particle, fieldset, time):
    particle.delete()

# try vector animation
for cnt in range(20):
    # First plot the particles
    pset.show(savefile='anim/particles'+str(cnt).zfill(3), field=fieldset.V, land=True, vmin=-2.5, vmax=1.5)

    # capture positions and timestamps of all particles

    # Then advect the particles for 1 day
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=1),  # runtime controls the interval of the plots
                 dt=timedelta(days=1),
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}) 

# turn animations into movie using image magick
# convert *.png out.gif

# generate a netcdf file with all particle positions
# make particle set
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, repeatdt=repeatdt)
# collect data
pset.execute(AdvectionRK4,
             runtime=timedelta(days=50),  # runtime controls the interval of the plots
             dt=timedelta(days=1),
             recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
             output_file=pset.ParticleFile(name="particle_positions", outputdt=timedelta(days=1)))
