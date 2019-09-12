# program_train
from lagrangian import *
from datetime import datetime, timedelta

##########
# Config #
##########
# net cdf data for fieldset
ROMS_dir = '/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/'
filenames = "/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/naroom*.nc"
variables = {'U': 'ubar',
             'V': 'vbar'}
dimensions = {'lon': 'lon_rho', 'lat': 'lat_rho', 'depth':'s_rho', 'time': 'ocean_time'}
indicies = {'depth':[29]} # surface only
# particle generation region
generation_region = boxmaker(152, -35.5, 75)

# plotting
domain = {'N':-34.453367, 'S':-38.050653,'W':147.996456, 'E':152.457344}

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

particle_animation_filterV(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
    generation_region=generation_region, repeatdt=timedelta(days=1), sampledt=timedelta(days=1), out_dir='anim/',
    runlength=20, domain='full', vmin=-0.050000001, vmax=-0.049999999, cmap=cmocean.tools.crop_by_percent(cmocean.cm.balance, 50, which='both', N=None), ROMS_dir=ROMS_dir, V_threshold=-0.05, plot_type='particles-field-filter-t0',
    timeorigin=datetime(1994, 1, 1, 12))

"""
Notes:
Good colourmap for binomial thresholding:
cmocean.tools.crop_by_percent(cmocean.cm.balance, 50, which='both', N=None)
"""