# program contro script
from lagrangian import *
from config import *

#################################
# Get particle position dataset #
#################################
if particle_positions:
    print('\nBUILDING TRAJECTORIES\n')
    particle_positions_filterV(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
        generation_region=generation_region, ROMS_dir=ROMS_dir, runtime=runlength, 
        sampledt=sampledt, outputfn=particlefn, V_threshold=V_threshold, maxParticlesStep=maxParticlesStep)

###################
# Make animations #
###################
if particle_animation:
    print('\nBUILDING ANIMATION\n')
    particle_animation_filterV(filenames=filenames, variables=variables, dimensions=dimensions, indicies=indicies,
        generation_region=generation_region, repeatdt=sampledt, sampledt=sampledt, out_dir=animation_output,
        runlength=runlength, domain=domain, vmin=vmin, vmax=vmax, cmap=cmocean.cm.speed, ROMS_dir=ROMS_dir, V_threshold=V_threshold, 
        plot_type=plot_type, timeorigin=p_timeorigin, maxParticlesStep=maxParticlesStep)
"""
Notes:
Good colourmap for binomial thresholding:
cmocean.tools.crop_by_percent(cmocean.cm.balance, 50, which='both', N=None)
"""

######################
# Make training set #
######################
if produce_training_data:
    print('\nPRODUCING OCEANCC TRAINING DATASET\n')
    particle_training(particle_trajectories_fn=particle_trajectories_fn, ROMS_file=ROMS_file, outputfn=training_outputfn, timeorigin_traj=timeorigin_traj, 
    	timeorigin_ROMS=timeorigin_ROMS, animate=True, plot_type='grid_selection', anim_out='anim/grid_selection/', 
    	spinup=spinup)