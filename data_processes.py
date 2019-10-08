# data processes
import math
import numpy as np
from datetime import datetime, timedelta
import re
from sklearn.neighbors import KernelDensity 

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
def oceantime_2_dt(frame_time):
    """
    Datetime is in local timezone (but not timezone aware)
    """
    dtcon_days = frame_time
    dtcon_start = datetime(1990,1,1)
    dtcon_delta = timedelta(dtcon_days/24/60/60)
    dtcon_offset = dtcon_start + dtcon_delta

    return dtcon_offset

######################
# Harversine formula #
######################
def harversine(lon1, lat1, lon2, lat2):
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









