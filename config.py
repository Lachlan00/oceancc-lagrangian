from datetime import datetime, timedelta
import cmocean

############
# Analysis #
############
particle_positions = False
particle_animations = True
produce_training_data = False

#################################
# Hydrodynamic model data setup #
#################################
ROMS_dir = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/'
filenames = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/EAC_20y_merge.nc'
variables = {'U': 'ubar',
             'V': 'vbar'}
dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_u', 'time': 'ocean_time'},
              'V': {'lon': 'lon_v', 'lat': 'lat_v', 'time': 'ocean_time'}}
indicies = None

#######################
# Particle generation #
#######################
generation_region = [152.5, 154.8, -30, -26]
V_threshold=-0.4
sampledt=timedelta(days=1)
p_timeorigin=datetime(1994, 1, 1, 12)
runlength = 'full'
maxParticlesStep = None # max particles generated per timestep

#############################
# Particle positions output #
#############################
particlefn = 'data/particle_positions.nc'

#######################
# Plotting parameters #
#######################
plot_type='particles-field-filter-t0'
domain = 'full'
# domain = {'N':-25.5, 'S':-32, 'W':151, 'E':156}
vmin=-0.8 
vmax=0.3
cmap = cmocean.cm.speed
animation_output = 'anim/'