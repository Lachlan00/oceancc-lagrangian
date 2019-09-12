# Custom plotting functions for ocean parcels
import numpy as np
import cmocean
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from itertools import compress

def make_map(domain):
    m = Basemap(projection='merc', llcrnrlat=domain['S'], urcrnrlat=domain['N'],
        llcrnrlon=domain['W'], urcrnrlon=domain['E'], lat_ts=20, resolution='h')

    return m

def plot_field(m, field, lons, lats, vmin, vmax, cmap, title):
    plt.close("all")
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add region zone
    # plot color
    m.pcolor(lons, lats, np.squeeze(field), latlon=True ,vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    # add grid
    parallels = np.arange(-81.,0,2.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # datetime title
    plt.title(title)
    plt.tight_layout()

    return fig

def plot_field_particles(m, field, lons, lats, vmin, vmax, cmap, pset, title, filter_t0=False, plot_region=False, region=None):
    """
    Plot particles over field
    """
    plt.close("all")
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add region zone
    # plot color
    m.pcolor(lons, lats, np.squeeze(field), latlon=True ,vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    # plot region
    if plot_region:
            x1,y1 = m(region[0],region[2]) 
            x2,y2 = m(region[0],region[3]) 
            x3,y3 = m(region[1],region[3]) 
            x4,y4 = m(region[1],region[2])
            p = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none',edgecolor='black',linewidth=2.5,zorder=10,ls='dashed')
            plt.gca().add_patch(p)
    # plot pset
    plons, plats = m([p.lon for p in pset], [p.lat for p in pset])
    # filter ages
    if filter_t0:
        page = np.asarray([p.age for p in pset])
        plons = list(compress(plons, page != 0.))   
        plats = list(compress(plats, page != 0.))   
    # plot points
    plt.scatter(plons, plats, c='#707070', s=1.7, edgecolors='k', linewidths=0.5)
    # add grid
    parallels = np.arange(-81.,0,2.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # datetime title
    plt.title(title)
    plt.tight_layout()

    return fig

