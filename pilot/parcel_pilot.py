# code to generate and track particles with a netCDF file
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ErrorCode
import numpy as np
from os import listdir
from os.path import isfile, join
from netCDF4 import Dataset

# local modules
from data_processes import *

##########
# Config #
##########
ROMS_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/testing/"

#######################
# Read in NetCDF data #
#######################
# get file list
file_ls = [f for f in listdir(ROMS_directory) if isfile(join(ROMS_directory, f))]
file_ls = list(filter(lambda x:'.nc' in x, file_ls))
file_ls = sorted(file_ls)

# obtain dimension information
# load a file
nc_file = ROMS_directory + file_ls[0]
fh = Dataset(nc_file, mode='r')

"""
I had a problem when I recived my orginal ROMS nc files where 
the U and V xi dimension were cut off by 1 for V (165 vs 166). In 
case this occurs again I will reshape the dimensions to the minimum
value (i.e. trim the array from 166 to 165 along xi).
"""
# get field set array dimensions
lon_rho = fh.variables['lon_rho'][:,:]
lat_rho = fh.variables['lat_rho'][:,:]
# check dimensions agree for field set and if they don't use the
# the smallest dimensions (not actually required for rho)
dims = [lon_rho.shape, lat_rho.shape]
dims_min = (min([x[0] for x in dims]) , min([x[1] for x in dims]))

"""
# NOTE
I have used lon and lat rho values but V and U actually map to eta_u
and xi_u. But I'm getting odd results pulling those corrdinates. 
Near enough is good enough I think and rho coordinated should be fine
to use for now unless I hear otherwise.
Note that eta is Y (kinda lat) and xi is x (kinda lon).. kinda
"""

# Loop through files and construct 3 dimension array from surface 
# layer spatial dimensions and time.
# empty arrays to store files
uarray = [np.nan]*len(file_ls)
varray = [np.nan]*len(file_ls)
timearray = [np.nan]*len(file_ls)
# the main loop
i = 0
for file in file_ls:
    fh = Dataset(ROMS_directory + file, mode='r')
    uarray[i] = fh.variables['u'][:,29,0:dims_min[0],0:dims_min[1]]
    varray[i] = fh.variables['u'][:,29,0:dims_min[0],0:dims_min[1]]
    timearray[i] = fh.variables['ocean_time'][:]
    i += 1
# now we need to stack the 3 dimensional arrays together.
u = np.vstack(uarray)
v = np.vstack(uarray)
time = np.concatenate(timearray)
# convert time values to datetimes
time = [oceantime_2_dt(t) for t in time]
time = [int(t.timestamp()) for t in time]

##################################
# Transform to Parcels feild-set #
##################################
# make field-set
data = {'U':u, 'V':v}
dimensions = {'lon':lon_rho, 'lat':lat_rho, 'time':time}
fieldset = FieldSet.from_data(data, dimensions)

# show feild
fieldset.U.show(savefile='test')

######################
# Generate particles #
######################
# set region where particles will be generated
generation_zone = []

# experiment with animating fieldset
# particl generation
lons = np.arange(151, 152, .1)
lats = np.arange(-35, -34, .1)

# make particle set
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats)
pset.show(field=fieldset.U, savefile='test-pset')

# kernle to delete particles out of range
def DeleteParticle(particle, fieldset, time):
    particle.delete()

# try vector animation
for cnt in range(30):
    # First plot the particles
    pset.show(savefile='anim/particles'+str(cnt).zfill(2), field='vector', land=True, vmax=2.0)

    # Then advect the particles for 6 hours
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=1),  # runtime controls the interval of the plots
                 dt=timedelta(days=1),
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}) 

# turn animations into movie using image magick
# convert *.png out.gif







