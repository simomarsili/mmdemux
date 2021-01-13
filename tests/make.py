# -*- coding: utf-8 -*-
"""Prepare tests files."""
# pylint: disable=no-member
import mdtraj
import simtk.openmm as mm
from openmmtools import mcmc, states, testsystems
from openmmtools.multistate import MultiStateReporter, ParallelTemperingSampler
from simtk import unit


def energy_function(test, topology=None, system=None, positions=None):
    """Potential energy for TestSystem object."""
    topology = topology or test.topology
    system = system or test.system
    positions = positions or test.positions
    platform = mm.Platform.getPlatformByName('CPU')
    properties = {}
    integrator = mm.VerletIntegrator(0.002 * unit.picoseconds)
    simulation = mm.app.Simulation(top, system, integrator, platform,
                                   properties)
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True)
    ene = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return ene


stem = 'text_repex'

testsystem = testsystems.AlanineDipeptideImplicit()
# testsystem = testsystems.SrcImplicit()
# testsystem = testsystems.HostGuestVacuum()

# save topology as .pdb
top = mdtraj.Topology.from_openmm(testsystem.topology)
trj = mdtraj.Trajectory([testsystem.positions / unit.nanometers], top)
trj.save(stem + '.pdb')

# save system as .xml
serialized_system = mm.openmm.XmlSerializer.serialize(testsystem.system)
with open(stem + '.xml', 'w') as fp:
    print(serialized_system, file=fp)

n_replicas = 3  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 600.0 * unit.kelvin  # Maximum temperature.
reference_state = states.ThermodynamicState(system=testsystem.system,
                                            temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0 * unit.femtoseconds, n_steps=50)
sampler = ParallelTemperingSampler(mcmc_moves=move,
                                   number_of_iterations=float('inf'),
                                   online_analysis_interval=None)

storage_path = stem + '.nc'
reporter = MultiStateReporter(storage_path, checkpoint_interval=1)
sampler.create(reference_state,
               states.SamplerState(testsystem.positions),
               reporter,
               min_temperature=T_min,
               max_temperature=T_max,
               n_temperatures=n_replicas)

sampler.run(n_iterations=10)
