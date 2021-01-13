# -*- coding: utf-8 -*-
"""Prepare tests files."""
# pylint: disable=no-member
import mdtraj
import simtk.openmm as mm
from openmmtools import mcmc, states, testsystems
from openmmtools.multistate import MultiStateReporter, ParallelTemperingSampler
from simtk import unit

stem = 'test_repex'

testsystem = testsystems.AlanineDipeptideImplicit()

topology = testsystem.topology
system = testsystem.system
positions = testsystem.positions

# save topology as .pdb
top = mdtraj.Topology.from_openmm(topology)
trj = mdtraj.Trajectory([positions / unit.nanometers], top)
trj.save(stem + '.pdb')

# save system as .xml
serialized_system = mm.openmm.XmlSerializer.serialize(system)
with open(stem + '.xml', 'w') as fp:
    print(serialized_system, file=fp)

n_replicas = 3  # Number of temperature replicas.
T_min = 298.0 * unit.kelvin  # Minimum temperature.
T_max = 600.0 * unit.kelvin  # Maximum temperature.
reference_state = states.ThermodynamicState(system=system, temperature=T_min)

move = mcmc.GHMCMove(timestep=2.0 * unit.femtoseconds, n_steps=50)
simulation = ParallelTemperingSampler(mcmc_moves=move,
                                      number_of_iterations=float('inf'),
                                      online_analysis_interval=None)

storage_path = stem + '.nc'
reporter = MultiStateReporter(storage_path, checkpoint_interval=1)
simulation.create(reference_state,
                  states.SamplerState(positions),
                  reporter,
                  min_temperature=T_min,
                  max_temperature=T_max,
                  n_temperatures=n_replicas)

simulation.run(n_iterations=10)

trj = mdtraj.load(stem + '.pdb')
top = trj.topology.to_openmm()
print(top == topology)
