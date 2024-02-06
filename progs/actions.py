"""
This module provides functionality to set up and run molecular dynamics simulations using HOOMD-blue.
It includes functions to add NVT and NPT ensembles, run simulations, and perform box elongation during a simulation.
"""

import hoomd

def add_nvt(integrator, sim, kT, tau):
    """
    Adds an NVT ensemble to the simulation.

    Parameters:
    - integrator: The integrator to use for the simulation.
    - sim: The simulation object.
    - kT: The temperature for the thermostat.
    - tau: The time constant for the thermostat.
    """
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=kT, tau=tau)
    filter = hoomd.filter.Rigid(("center", "free"))
    nvt = hoomd.md.methods.ConstantVolume(filter=filter, thermostat=thermostat)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(kT=kT, filter=filter)
    sim.run(0)
    thermostat.thermalize_dof()

def add_npt(integrator, sim, kT, tau, tauS, couple):
    """
    Adds an NPT ensemble to the simulation.

    Parameters:
    - integrator: The integrator to use for the simulation.
    - sim: The simulation object.
    - kT: The temperature for the thermostat.
    - tau: The time constant for the thermostat.
    - tauS: The time constant for the barostat.
    - couple: Specifies the coupling between the x, y, and z dimensions.
    """
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=kT, tau=tau)
    filter = hoomd.filter.Rigid(("center", "free"))
    npt = hoomd.md.methods.ConstantPressure(filter=filter, thermostat=thermostat, S=0, tauS=tauS, couple=couple)
    integrator.methods.append(npt)
    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(kT=kT, filter=filter)
    sim.run(0)
    thermostat.thermalize_dof()

def run_sim(sim, n_timesteps, name, n_save=10000, save=['property'], n_switch=1000, mode='xb'):
    """
    Runs the simulation and periodically saves the state to a GSD file.

    Parameters:
    - sim: The simulation object.
    - n_timesteps: The number of timesteps to run the simulation for.
    - name: The filename for the output GSD file.
    - n_save: The interval at which to save the simulation state.
    - save: Specifies which properties to save dynamically.
    - n_switch: Timestep before which to save more frequently.
    - mode: The file writing mode ('xb' for exclusive binary, 'wb' for write binary).
    """
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)
    logger = hoomd.logging.Logger()
    logger.add(thermodynamic_properties, quantities=['temperature', 'pressure', 'energy'])
    logger.add(sim, quantities=['timestep'])
    gsd_writer = hoomd.write.GSD(filename=name, trigger=hoomd.trigger.Or([hoomd.trigger.Periodic(n_save), hoomd.trigger.Before(n_switch)]), mode=mode, filter=hoomd.filter.All(), dynamic=save)
    sim.operations.writers.append(gsd_writer)
    gsd_writer.logger = logger
    sim.run(n_timesteps)
    gsd_writer.flush()
    sim.operations.writers.remove(gsd_writer)

def elongate_box(sim, t1, n_timesteps, coeff=3):
    """
    Elongates the simulation box over a specified number of timesteps.

    Parameters:
    - sim: The simulation object.
    - t1: The start time for the box elongation.
    - n_timesteps: The number of timesteps over which to elongate the box.
    - coeff: The factor by which to elongate the box's length.
    """
    ramp = hoomd.variant.Ramp(A=0, B=1, t_start=t1, t_ramp=n_timesteps)
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box)
    final_box.Lx *= coeff
    box_resize = hoomd.update.BoxResize(trigger=1, box1=initial_box, box2=final_box, variant=ramp, filter=hoomd.filter.Null())
    sim.operations.updaters.append(box_resize)




