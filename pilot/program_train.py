# program_train
from lagrangian import *
from datetime import datetime, timedelta

##########
# Config #
##########
# net cdf data for fieldset
#ROMS_dir = '/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/'
ROMS_dir = '/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/'
# filenames = "/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/naroom*.nc"
filenames = "/Users/lachlanphillips/PhD_Large_Data/ROMS/surface_subset/merged/EAC_20y_merge.nc"
variables = {'U': 'ubar',
             'V': 'vbar'}
# dimensions = {'lon': 'lon_rho', 'lat': 'lat_rho', 'depth':'s_rho', 'time': 'ocean_time'}
dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_u', 'time': 'ocean_time'},
              'V': {'lon': 'lon_v', 'lat': 'lat_v', 'time': 'ocean_time'}}
# indicies = {'depth':[29]} # surface only
# particle generation region
# generation_region = boxmaker(152, -35.5, 75)
generation_region = [152.5, 154.2, -29, -26]
# plotting
# domain = {'N':-34.453367, 'S':-38.050653,'W':147.996456, 'E':152.457344}

# particle position file # REMOVE THIS
particlefn = 'data/particle_positions.nc'

#################################
# Get particle position dataset #
#################################
# particle_positions(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
#     generation_region=generation_region, repeatdt=timedelta(days=1), runtime=timedelta(days=20), 
#     sampledt=timedelta(days=1), outputfn="data/particle_positions")

# particle_positions_filterV(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
#     generation_region=generation_region, ROMS_dir=ROMS_dir, runtime=timedelta(days=20), 
#     sampledt=timedelta(days=1), outputfn="data/particle_positions", V_threshold=-0.05)

###################
# Make animations #
###################
# particle_animation(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
#     generation_region=generation_region, repeatdt=timedelta(days=1), sampledt=timedelta(days=1), out_dir='anim/',
#     runlength=10, domain='full', vmin=-3, vmax=3, cmap=cmocean.cm.speed, plot_type='particles-field-filter-t0',
#     timeorigin=datetime(1994, 1, 1, 12))

particle_animation_filterV(filenames=filenames, variables=variables, dimensions=dimensions, indicies=None,
    generation_region=generation_region, repeatdt=timedelta(days=1), sampledt=timedelta(days=1), out_dir='anim/',
    runlength=120, domain='full', vmin=-0.8, vmax=0.3, cmap=cmocean.cm.speed, ROMS_dir=ROMS_dir, V_threshold=-0.2, 
    plot_type='particles-field-filter-t0', timeorigin=datetime(1994, 1, 1, 12))

"""
Notes:
Good colourmap for binomial thresholding:
cmocean.tools.crop_by_percent(cmocean.cm.balance, 50, which='both', N=None)
"""