# Custom plotting functions for ocean parcels
import numpy as np
import cmocean
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from itertools import compress
import seaborn as sns

############
# Make map #
############
def make_map(domain):
    """
    Make basemap for plotting
    """
    m = Basemap(projection='merc', llcrnrlat=domain['S'], urcrnrlat=domain['N'],
        llcrnrlon=domain['W'], urcrnrlon=domain['E'], lat_ts=20, resolution='h')

    return m

#####################
# Plot the fieldset #
#####################
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
    cbar = plt.colorbar()
    cbar.set_label('North/South Velocity (ms-1)', rotation=270, labelpad=-3)
    # add grid
    parallels = np.arange(-81.,0,2.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,2.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # datetime title
    plt.title(title)
    plt.tight_layout()

    return fig

#############################################
# Plot the fieldset with particles overlaid #
#############################################
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
    cbar = plt.colorbar()
    cbar.set_label('North/South Velocity (ms-1)', rotation=270, labelpad=15)
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
    parallels = np.arange(-80.,0,2.)
    m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    meridians = np.arange(10.,351.,4.)
    m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1, dashes=[3,3], color='#707070')
    # datetime title
    plt.title(title)
    plt.tight_layout()

    return fig

#########################
# Particle density plot #
#########################
def particle_density_plot(x, y, xlim, ylim):
    plt.close("all")
    sns_plot = sns.jointplot(x=x, y=y, kind="kde", xlim=xlim, ylim=ylim, joint_kws={'shade_lowest':False})

    return sns_plot

###############################################
# Particle density plot with overlaid scatter #
###############################################
def particle_scatter_density_plot(x, y, xlim, ylim):
    plt.close("all")
    sns_plot = sns.jointplot(x=x ,y=y, xlim=xlim, ylim=ylim, color="k", joint_kws={'alpha':0.4}).plot_joint(sns.kdeplot, zorder=0, n_levels=6)

    return sns_plot

######################################
# Particle density plot on a basemap #
######################################
def particle_density_map(m, field, lons, lats, vmin, vmax, cmap, pset, title):
    """
    Plot particles
    """
    # calculate kde from particle coordinates

    plt.close("all")
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    # draw stuff
    m.drawcoastlines(color='black', linewidth=0.7)
    m.fillcontinents(color='#A0A0A0')
    # add region zone
    # plot contour
    m.contourf(lons, lats, np.squeeze(field), list(frange(vmin, vmax, 0.2)), cmap=cmap, latlon=True, vmin=vmin, vmax=vmax, extend='both')
    cbar = plt.colorbar()
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

####################################
# Plot hydrodynamic grid as points #
####################################
def plot_grid(lat_array, lon_array, downsample=1):
    plt.close("all")
    fig = plt.figure()
    fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    plt.scatter(lon_array[::downsample], lat_array[::downsample], c='k', s=1, linewidths=0.5)
    plt.ylabel('Lat')
    plt.xlabel('Lon')
    plt.show()

    return fig

############################################################
# Plot current sampling process (EAC vs nonEAC selections) #
############################################################
def plot_grid_selection(eta, xi, col, xlim, ylim):
    plt.close("all")
    fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0., right=1., bottom=0., top=0.9)
    ax.scatter(xi, eta, c=col, s=1, linewidths=0.5)
    plt.ylabel('eta')
    plt.xlabel('xi')
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])

    return fig
