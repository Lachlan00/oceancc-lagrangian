import numpy as np
import math
import time
from scipy.spatial import cKDTree

# load the numpy lat/lon grids
lat_array = np.load('../data/lat_array.npy')
lon_array = np.load('../data/lon_array.npy')
# get the array shape dimensions for later conversion
grid_shape = lat_array.shape

# test for lat/lon coordinate 
lat = -32
lon = 154
N = 3e8/7 # how many lat/lon paris I need to do

#####################
# Harversine method #
#####################
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # harversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2.)**2. + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2.)**2.
    c = 2. * math.asin(math.sqrt(a))
    km = 6371. * c # radius of earth
    return km

# test harversine method
print('\nHarversine results')
tic = time.perf_counter()
# create a list of distance from the point for all grid cells
dist_array = np.asarray([haversine(lon, lat, grid_lon, grid_lat) for grid_lon, grid_lat in zip(lon_array.flatten(), lat_array.flatten())])
# get the index of the minium value
min_idx = np.argmin(dist_array)
# transform the index back into grid cell dimensions
grid_dims = np.unravel_index(min_idx, grid_shape)
toc = time.perf_counter()

# report results
print('Single iteration time in seconds:', round(toc - tic, 2))
print('N iterations time in days:', round(((toc - tic)*N)/60/60/24, 2))
print('Grid coordinate:', grid_dims)                                                                                                                                                                                              
if (lon_array.flatten()[min_idx] == lon_array[grid_dims[0], grid_dims[1]]) and (lat_array.flatten()[min_idx] == lat_array[grid_dims[0], grid_dims[1]]):
   print('Results pass checks! :)')
else:
    print('Results FAIL checks :(')                                                                                                                                                                                                             

##########################
# Tunnel distance method #
##########################
def tunnel_fast(latvar, lonvar, lat0, lon0):
    """
    Find closest point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
    is minimum.
    """
    rad_factor = math.pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny,nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    # Compute numpy arrays for all values, no loops
    clat,clon = np.cos(latvals), np.cos(lonvals)
    slat,slon = np.sin(latvals), np.sin(lonvals)
    delX = np.cos(lat0_rad)*np.cos(lon0_rad) - clat*clon
    delY = np.cos(lat0_rad)*np.sin(lon0_rad) - clat*slon
    delZ = np.sin(lat0_rad) - slat;
    dist_sq = delX**2 + delY**2 + delZ**2
    minindex_1d = dist_sq.argmin()  # 1D index of minimum element
    iy_min,ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return (iy_min,ix_min)

# test tunnel distance method
print('\nTunnel distance results')
tic = time.perf_counter()
# create a list of distance from the point for all grid cells
grid_dims = tunnel_fast(lat_array, lon_array, lat, lon)
toc = time.perf_counter()

# report results
print('Single iteration time in seconds:', round(toc - tic, 5))
print('N iterations time in days:', round(((toc - tic)*N)/60/60/24, 2))
print('Grid coordinate:', grid_dims)
if (lon_array.flatten()[min_idx] == lon_array[grid_dims[0], grid_dims[1]]) and (lat_array.flatten()[min_idx] == lat_array[grid_dims[0], grid_dims[1]]):
   print('Results pass checks! :)')
else:
    print('Results FAIL checks! :(')


###############################
# Alt Harversine numba method #
###############################
def haversine_numba(s_lat, s_lng, e_lat, e_lng):
    """
    https://towardsdatascience.com/better-parallelization-with-numba-3a41ca69452e
    """
    # approximate radius of earth in km
    R = 6371.0

    s_lat = np.deg2rad(s_lat)                    
    s_lng = np.deg2rad(s_lng)     
    e_lat = np.deg2rad(e_lat)                       
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + \
        np.cos(s_lat)*np.cos(e_lat) * \
        np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d)) 

# test harversine numba method
print('\nAlt Numba Harversine results')
tic = time.perf_counter()
# create a list of distance from the point for all grid cells
dist_array = np.asarray([haversine_numba(lon, lat, grid_lon, grid_lat) for grid_lon, grid_lat in zip(lon_array.flatten(), lat_array.flatten())])
# get the index of the minium value
min_idx = np.argmin(dist_array)
# transform the index back into grid cell dimensions
grid_dims = np.unravel_index(min_idx, grid_shape)
toc = time.perf_counter()

# report results
print('Single iteration time in seconds:', round(toc - tic, 2))
print('N iterations time in days:', round(((toc - tic)*N)/60/60/24, 2))
print('Grid coordinate:', grid_dims)                                                                                                                                                                                              
if (lon_array.flatten()[min_idx] == lon_array[grid_dims[0], grid_dims[1]]) and (lat_array.flatten()[min_idx] == lat_array[grid_dims[0], grid_dims[1]]):
   print('Results pass checks! :)')
else:
    print('Results FAIL checks :(') 

##################
# KD tree method #
##################
def kdtree_fast(latvar,lonvar,lat0,lon0):
    """
    Adapted from:
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    rad_factor = math.pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny,nx = latvals.shape
    clat,clon = np.cos(latvals), np.cos(lonvals)
    slat,slon = np.sin(latvals), np.sin(lonvals)
    # Build kd-tree from big arrays of 3D coordinates
    triples = list(zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat)))
    kdt = cKDTree(triples)
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0,clon0 = np.cos(lat0_rad), np.cos(lon0_rad)
    slat0,slon0 = np.sin(lat0_rad), np.sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    
    return (iy_min, ix_min)

# test harversine numba method
print('\nKD Tree method results')
tic = time.perf_counter()
# create a list of distance from the point for all grid cells
grid_dims = kdtree_fast(lat_array, lon_array, lat, lon)
toc = time.perf_counter()

# report results
print('Single iteration time in seconds:', round(toc - tic, 2))
print('N iterations time in days:', round(((toc - tic)*N)/60/60/24, 2))
print('Grid coordinate:', grid_dims)                                                                                                                                                                                              
if (lon_array.flatten()[min_idx] == lon_array[grid_dims[0], grid_dims[1]]) and (lat_array.flatten()[min_idx] == lat_array[grid_dims[0], grid_dims[1]]):
   print('Results pass checks! :)')
else:
    print('Results FAIL checks :(') 

#############################
# KD tree method alt method #
#############################
def kdtree_process(kdt,lat0,lon0):
    """
    Adapted from:
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0,clon0 = np.cos(lat0_rad), np.cos(lon0_rad)
    slat0,slon0 = np.sin(lat0_rad), np.sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    
    return (iy_min, ix_min)

# produce kd_tree outside of the function
rad_factor = math.pi/180.0 # for trignometry, need angles in radians
# Read latitude and longitude from file into numpy arrays
latvals = lat_array[:] * rad_factor
lonvals = lon_array[:] * rad_factor
ny,nx = latvals.shape
clat,clon = np.cos(latvals), np.cos(lonvals)
slat,slon = np.sin(latvals), np.sin(lonvals)
# Build kd-tree from big arrays of 3D coordinates
triples = list(zip(np.ravel(clat*clon), np.ravel(clat*slon), np.ravel(slat)))
kdt = cKDTree(triples)

print('\nKD Tree alternative method results')
tic = time.perf_counter()
# create a list of distance from the point for all grid cells
grid_dims = kdtree_process(kdt, lat, lon)
toc = time.perf_counter()

# report results
print('Single iteration time in seconds:', round(toc - tic, 5))
print('N iterations time in days:', round(((toc - tic)*N)/60/60/24, 2))
print('Grid coordinate:', grid_dims)                                                                                                                                                                                              
if (lon_array.flatten()[min_idx] == lon_array[grid_dims[0], grid_dims[1]]) and (lat_array.flatten()[min_idx] == lat_array[grid_dims[0], grid_dims[1]]):
   print('Results pass checks! :)')
else:
    print('Results FAIL checks :(') 

