from datetime import datetime, timedelta
import cmocean

############
# Analysis #
############
# generate a particle trajectory file
particle_positions = False
# generate a particle animation
particle_animation = False
# produce oceancc training dataset
produce_training_data = True

#################################
# Hydrodynamic model data setup #
#################################
# this is the directory to the ROMS model (useful if the ROMS model is split into multiple files)
ROMS_dir = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/'
# This is to direct to a single ROMS model file (used trauning data extraction)
ROMS_file = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/EAC_20y_merge.nc'
# This is used in the particle simulations. Different to the above as also can be a dictionary for split files
filenames = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/EAC_20y_merge.nc'
# Variables passing for Ocean Parcels fieldsets
variables = {'U': 'ubar',
             'V': 'vbar'}
# Dimension passing for Ocean Parcels fieldsets
dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_u', 'time': 'ocean_time'},
              'V': {'lon': 'lon_v', 'lat': 'lat_v', 'time': 'ocean_time'}}
# Inidcies slicews if required (my file is only surface so do not need to specify depth)
indicies = None
# indicies = {'depth':[29]} # this is what you would to specify the surface in a file with depth
# timeorigin of ROMS model (used in trainging set)
timeorigin_ROMS = datetime(1990, 1, 1)

#######################
# Particle generation #
#######################
# Where particles are generated
generation_region = [152.5, 154.8, -30, -26]
# Southward velocity threshold for particle seeding
V_threshold=-0.4
# the sampling timestep of the particle generation and advections
sampledt=timedelta(days=1)
# time origin of the particle generation
p_timeorigin=datetime(1994, 1, 1, 12)
# how long we want to run the model for. 'full' gives all timesteps in the fieldset
runlength = 'full'
# max particles generated per timestep
maxParticlesStep = 100
# Particle positions output
particlefn = 'data/particle_trajectories.nc'

#######################
# Plotting parameters #
#######################
# type of plot to produce
plot_type='particles-field-filter-t0'
# Domain of the plot
domain = 'full'
# domain = {'N':-25.5, 'S':-32, 'W':151, 'E':156}
# max and min of the colour scale
vmin=-0.8 
vmax=0.3
# the colout map to use for north/south velocity
cmap = cmocean.cm.speed
# Where output PNGs should be writrten to
animation_output = 'anim/'

############################
# Training data parameters #
############################
# the particle trajectory file that will be used to extract the training data
particle_trajectories_fn = 'data/particle_trajectories.nc'
# trainging dataset output file (can be a csv or a pickle)
training_outputfn = 'data/training_data.pkl'
# spinup time
spinup = timedelta(days=364)
# time origin of trajectory file
timeorigin_traj = datetime(1994, 1, 1, 12)






