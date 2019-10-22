# data processes
import math
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.neighbors import KernelDensity 
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from numba import jit


#######################
# Add to list quickly #
#######################
def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

###################
# Query yes or no #
###################
def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply == 'y':
        return True
    if reply == 'n':
        return False
    else:
        return yes_or_no("Not a vaild response..")

#############################################
# Find min and max values in area in netCDF #
#############################################
def minmax_in_region(data, lons, lats, region):
    """
    Calculate minimum and maximum values in georeferenced array
    Give range as `range = [xmin, xmax, ymin, ymax]`
    """
    # constrain data to region
    bools = (region[0] <= lons) & (lons <= region[1]) & (region[2] <= lats) & (lats <= region[3])  
    data = data[bools]
    # get min and max
    local_min = data.min()
    local_max = data.max()

    return (local_min, local_max)

####################################################
# Convert ocean time (days since 1990) to datetime #
####################################################
def oceantime_2_dt(frame_time, dtcon_start=datetime(1990,1,1)):
    """
    Datetime is in local timezone (but not timezone aware)
    """
    dtcon_days = frame_time
    dtcon_delta = timedelta(dtcon_days/24/60/60)
    dtcon_offset = dtcon_start + dtcon_delta

    return dtcon_offset

######################
# Harversine formula #
######################
def haversine(lon1, lat1, lon2, lat2, unit='km'):
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
    if unit == 'km':
        return km
    elif unit == 'm':
        return km*1000
    
##################
# Make study box #
##################
def boxmaker(lon_orig, lat_orig, km):
    """
    Calculate points directly north, south, east and 
    west a certain distance from given coordinates
    """
    # convert decimal degrees to radians
    lon_orig, lat_orig = map(math.radians, [lon_orig, lat_orig])
    # reverse harversine formula
    c = km / 6371.
    a = math.sin(c/2.)**2.
    dlat = 2. * math.asin(math.sqrt(a))
    dlon = 2. * math.asin(math.sqrt(a/(math.cos(lat_orig)**2.)))
    # convert back to decimal degrees 
    lon_orig, lat_orig, dlat, dlon = map(math.degrees, [lon_orig, lat_orig, dlat, dlon])
    # find coordinates
    north = lat_orig + dlat
    south = lat_orig - dlat
    east = lon_orig + dlon
    west = lon_orig - dlon
    # correct over the 0-360 degree line
    if west > 360:
        west = west - 360
    if east > 360:
        east = east - 360
    # round to 6 decimal places
    region = [west, east, south, north]
    region = [round(x,6) for x in region]

    # export region
    return region

##############################
# Count points in study zone #
##############################
def count_points(lons, lats, region, bath=None, depthmax=1e10):
    """
    Passing bathymetry and depthmax is optional
    """
    # if no bathymetry data, make bath = lons so depthmax logic is always True
    if bath is None:
        bath = lons
    # make generator of all points
    point_tuple = zip(lats.ravel(), lons.ravel(), bath.ravel())
    # iterate over tuple points and keep every point that is in box
    point_list = []
    j = 0
    for i in point_tuple:
        if region[2] <= i[0] <= region[3] and region[0] <= i[1] <= region[1] and i[2] < depthmax:
            point_list.append(j)
        j = j + 1

    # return number of points
    return len(point_list)

###############################
# Extract digits from strings #
###############################
def atoi(text):
    return int(text) if text.isdigit() else text

###############################
# Extract digits from strings #
###############################
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

##############
# RatioA mean #
##############
def mean_ratioA(df):
    # calculate ratios
    df['ratioA'] = [A/(A+B) for A, B in zip(df['countA'], df['countB'])]
    # add date components
    df['day'] = [x.day for x in df['dt']]
    df['month'] = [x.month for x in df['dt']]
    df['year'] = [x.year for x in df['dt']]

    # Seasonal data
    df = df[df.year != 2016] # drop 2016 (incomplete)
    # calc yearly means
    df_std = df.groupby(['month', 'day']).std().reset_index()
    df_mean = df.groupby(['month', 'day']).mean().reset_index()

    # build index
    index = df_mean.index
    base = datetime(2000, 1, 1, 0, 0, 0)
    index = [base + timedelta(int(x)) for x in index]
    df_mean['index'] = index

    return df_mean

##################
# Kernel density #
##################
def kde_estimate(lon, lat):
    """
    Calculate kernel density estimation from lon/lat arrays
    https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """
    latlon = np.vstack([lat, lon]).T
    kde = KernelDensity(bandwidth=0.03, metric='haversine')
    kde.fit(np.radians(latlon))

    return kde   

##############################################
# Get cell values from latlon gridded arrays #
##############################################
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

######################
# kdtree pass method #
######################
def kdtree_process(kdt, lon0, lat0, array_shape):
    """
    Adapted from:
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    lat0_rad = lat0 * math.pi/180.0
    lon0_rad = lon0 * math.pi/180.0
    clat0,clon0 = np.cos(lat0_rad), np.cos(lon0_rad)
    slat0,slon0 = np.sin(lat0_rad), np.sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, array_shape)
    
    return (iy_min, ix_min)

##############################
# kdtree parallelised method #
##############################
"""
Version of kdtree_process that is compatible with numba
parallelisation methods
Does not work as kdtrees are npt supported by numba
"""
@jit(nopython=True, parallel=True)
def kdtree_parallel(kdt, coords, latvals):
    out_dims = [(np.nan, np.nan)]*len(coords)
    for i in numba.prange(0, len(coords)):
        coord = coords[i]
        lat0_rad = coord[1] * math.pi/180.0
        lon0_rad = coord[0] * math.pi/180.0
        clat0,clon0 = np.cos(lat0_rad), np.cos(lon0_rad)
        slat0,slon0 = np.sin(lat0_rad), np.sin(lon0_rad)
        dist_sq_min, minindex_1d = kdt.query([clat0*clon0, clat0*slon0, slat0])
        # replacement of nucmba unspported np.unravel_index()
        # iy_min,ix_min = np.unravel_index(minindex_1d, latvals.shape)
        iy_min = int(minindex_1d / latvals.shape[1])
        ix_min = minindex_1d % latvals.shape[1]
        out_dims[i] = (iy_min, ix_min)
    return out_dims

##########################
# Tunnel distance method #
##########################
def tunnel_fast(latvar, lonvar, lon0, lat0):
    """
    Adapted from:
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
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

###########################################
# Tunnel distance method with coordinates #
###########################################
def tunnel_fast_coord(coord, latvar, lonvar):
    """
    Adapted from:
    https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    Find closest point in a set of (lat,lon) points to specified point
    latvar - 2D latitude variable from an open netCDF dataset
    lonvar - 2D longitude variable from an open netCDF dataset
    lat0,lon0 - query point
    Returns iy,ix such that the square of the tunnel distance
    between (latval[it,ix],lonval[iy,ix]) and (lat0,lon0)
    is minimum.
    """
    lat0 = coord[0]
    lon0 = coord[1]
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

#########################################################
# Parallel processing version of the tunnel fast method #
#########################################################
"""
Version of tunnel_fast_coords that is compatible with numba
parallelisation methods
"""
@jit(nopython=True, parallel=True)
def tunnel_parallelise(coords, latvar, lonvar):
    out_dims = [(np.nan, np.nan)]*len(coords)
    for i in numba.prange(0, len(coords)):
        coord = coords[i]
        rad_factor = math.pi/180.0 # for trignometry, need angles in radians
        # Read latitude and longitude from file into numpy arrays
        latvals = latvar[:] * rad_factor
        lonvals = lonvar[:] * rad_factor
        lat0_rad = coord[1] * rad_factor
        lon0_rad = coord[0] * rad_factor
        # Compute numpy arrays for all values, no loops
        clat, clon = np.cos(latvals), np.cos(lonvals)
        slat, slon = np.sin(latvals), np.sin(lonvals)
        delX = np.cos(lat0_rad)*np.cos(lon0_rad) - clat*clon
        delY = np.cos(lat0_rad)*np.sin(lon0_rad) - clat*slon
        delZ = np.sin(lat0_rad) - slat;
        dist_sq = delX**2 + delY**2 + delZ**2
        minindex_1d = dist_sq.argmin()  # 1D index of minimum element
        # replacement of nucmba unspported np.unravel_index()
        # iy_min,ix_min = np.unravel_index(minindex_1d, latvals.shape)
        iy_min = int(minindex_1d / latvals.shape[1])
        ix_min = minindex_1d % latvals.shape[1]
        out_dims[i] = (iy_min, ix_min)
    return out_dims

############################
# Find cells in N distance #
############################
def Nneighbours(cells, array_shape, N, check=False, out_dir='anim/tests/', diagonal=False):
    """
    From tuple list of cells in array select all N neighbours
    """
    cells = list(zip(*cells))
    # make empty array
    A = np.zeros(array_shape, dtype=bool)
    # True passed cells
    A[cells[0], cells[1]] = True
    # check
    if check:
        plt.imsave(out_dir+'neighbour_'+str(0).zfill(2)+'.png', np.flip(np.flip(A),1))

    # true neighbours N times
    for i in range(0, N):
        # get all True indicies
        idx = np.where(A)
        # get surrounding cell positions
        a = [idx[0], idx[1]+1]
        b = [idx[0], idx[1]-1]
        c = [idx[0]-1, idx[1]]
        d = [idx[0]+1, idx[1]]
        if diagonal:
            e = [idx[0]+1, idx[1]+1]
            f = [idx[0]-1, idx[1]+1]
            g = [idx[0]+1, idx[1]-1]
            h = [idx[0]-1, idx[1]-1]
        # make sure no illegal values
        a[1][a[1] > array_shape[1]-1] = array_shape[1]-1
        b[1][b[1] < 0] = 0
        c[0][c[0] < 0] = 0
        d[0][d[0] > array_shape[0]-1] = array_shape[0]-1
        if diagonal:
            e[0][e[0] > array_shape[0]-1] = array_shape[0]-1
            e[1][e[1] > array_shape[1]-1] = array_shape[1]-1
            f[0][f[0] < 0] = 0
            f[1][f[1] > array_shape[1]-1] = array_shape[1]-1
            g[0][g[0] > array_shape[0]-1] = array_shape[0]-1
            g[1][g[1] < 0] = 0
            h[0][h[0] < 0] = 0
            h[1][h[1] < 0] = 0
        # replace values
        A[tuple(a)] = True
        A[tuple(b)] = True
        A[tuple(c)] = True
        A[tuple(d)] = True
        if diagonal:
            A[tuple(e)] = True
            A[tuple(f)] = True
            A[tuple(g)] = True
            A[tuple(h)] = True
        # visual check
        if check:
            plt.imsave(out_dir+'neighbour_'+str(i+1).zfill(2)+'.png', np.flip(np.flip(A),1))

    # turn back into tuples
    idx = np.where(A)

    return tuple(zip(idx[0], idx[1]))

#####v###################################################
# remove elements from list in another list in parallel #
#####v###################################################
"""
Does not yet work, tricky to parallise
"""
@jit(nopython=True, parallel=True)
def remove_from_list_parallel(thelist, remove_items):
    for i in numba.prange(0, len(remove_items)):
        if remove_items[i] in thelist:
            thelist = thelist[thelist != remove_items[i]]
    return thelist
















