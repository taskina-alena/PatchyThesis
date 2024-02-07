import sys
import hoomd
import numpy as np
sys.path.append('/net/theorie/auto/scratch1/alena.taskina01/md/hoomd4/progs')
import initialisation
import actions

# System configuration parameters.
n_patches = 3
num_units = 10000  # Number of units in the lattice (it's equal to number of particles when lattice is default sc).
spacing = 1.2  # Spacing between particles in the lattice in units of particles diameter.
kT = 0.06 # target kinetic temperature
rigid = False # whether rigid constrain should be introduced
angles_inter= True # whether angles interactions between patches are introduced
uniform_patches = True # whethe interactions between patches are uniformed. If not -- patches from the same core repulse each other
k = 1 # angle stiffness
seed = 0 # seed for initialising simulation
gpu_ids = [0] # which gpus to use
r_core = 0.5 # radius of core
r_patch = 0.12 # radius of patch
n_steps = 100 # length of running
n_save = 1 # period with which to save into the file
dt = 0.001 #timestep for integration
in_file=None
out_file='/share/scratch1/alena.taskina01/md/hoomd4/3patches/test0.gsd' #filename to save simulation
save=['property'] # if momentum is added, than images are saved to compute msd
elongation = False

patch_angles = {
    4: 109.5,  # Tetrahedral angle
    3: 120    # triangle configuration
}

#initialise the system
system = initialisation.ParticleInitializer(n_patches=n_patches, r_core=r_core, r_patch=r_patch, is_rigid=rigid, angles_inter=angles_inter, uniform_patches = uniform_patches)

# Set up the simulation with GPU acceleration
# start from preequilibrated system 
sim = hoomd.Simulation(device=hoomd.device.GPU(gpu_ids=gpu_ids), seed=seed)
if in_file:
    sim.create_state_from_gsd(in_file)
else:
    system.init_lattice(num_units=num_units, spacing=spacing, uniform=True)
    sim.create_state_from_snapshot(system.snapshot)
    # add patches to the core in case of rigid bodies
    if rigid: 
        system.rigid.create_bodies(sim.state)

# add patches to the core
if rigid: 
    system.rigid.create_bodies(sim.state)

# box elongation to study phase separation
if elongation:
    t1 = sim.initial_timestep
    if not rigid: 
        actions.elongate_box(sim, t1, 100, coeff=1.1) #if bonds are present, first elongate the box for a little to avoid problems with bonds due to pbc
        t1+=100
        n_steps+=100
    actions.elongate_box(sim, t1, n_steps-100, coeff=3)

# Initialize the integrator, setting integrate_rotational_dof based on whether the system is rigid.
integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=rigid)
# If the system has rigid bodies, assign the rigid body constraint to the integrator.
if rigid:
    integrator.rigid = system.rigid

# Initialize and describe forces between different types of particles.
force = initialisation.ForceInitializer()
force.describe_force('patch', 'patch', n=2, m=1, eps=9, r=r_patch, r_cut=r_patch, shift_x=True) # attractive interaction between patches
force.describe_force('core', 'patch', n=2, m=1, eps=0, r=0, r_cut=0) #no interactions between core and patch
force.describe_force('core', 'core', n=50, m=49, eps=1, r=r_core, r_cut=(50/49)*r_core, shift_x=False) # excluded volume interactions between cores
integrator.forces.append(force.force)

# describe the bond between patch and core, if rigid constrain is absent
if not rigid:
    force.describe_bond(r0=system.r_core, k=1000)
    integrator.forces.append(force.bond)

#describe angles
if angles_inter:
    angle = np.pi*(patch_angles[n_patches]/180) 
    force.describe_angle(t0=angle, k=k)
    integrator.forces.append(force.angle)

# Configure and add the NPT barostat with 0 target pressure to the simulation.
actions.add_npt(integrator, sim, kT, 0.1, 1, 'none')

# Run the simulation for a specified number of steps, saving the state periodically
actions.run_sim(sim=sim, n_steps=n_steps, filename=out_file, n_save=n_save, n_switch=0)

