# -*- coding: utf-8 -*-
"""
Drop-in replacement for yank.analyze.extract_trajectory

"""
# pylint: disable=no-member,protected-access

import logging
import os

import mdtraj
import numpy as np
import openmmtools as mmtools
import simtk.openmm as mm
import yank
from pymbar import timeseries

logger = logging.getLogger(__name__)


def extract_trajectory(  # pylint: disable=R0912,R0913,R0914,R0915
        ref_system,
        top,
        nc_path,
        nc_checkpoint_file=None,
        state_index=None,
        replica_index=None,
        start_frame=0,
        end_frame=-1,
        skip_frame=1,
        keep_solvent=True,
        discard_equilibration=False,
        image_molecules=False,
        ligand_atoms=None,
        solvent_atoms='auto'):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    ref_system : System object
        Reference state System object.
    top : Topography or Topology object
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one
        chosen by the nc_path file. Default: None
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None
        (default is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).
    ligand_atoms : iterable or int or str, optional
        The atomic indices of the ligand. A string is interpreted as a mdtraj
        DSL specification. Needed to recenter with pbc around the receptor if
        image_molecules=True.
    solvent_atoms : iterable of int or str, optional
        The atom indices of the solvent. A string is interpreted as an mdtraj
        DSL specification of the solvent atoms. Needed to recenter with pbc
        around solute molecules if image_molecules=True. If 'auto', a list of
        common solvent residue names will be used to automatically detect
        solvent atoms (default is 'auto').

    Returns
    -------
    trajectory: mdtraj.Trajectory
        The trajectory extracted from the netcdf file.

    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    reference_system = ref_system
    if isinstance(top, mm.app.topology.Topology):
        topography = yank.Topography(top,
                                     ligand_atoms=ligand_atoms,
                                     solvent_atoms=solvent_atoms)
    else:
        topography = top
    topology = topography.topology

    # Import simulation data
    reporter = None
    try:
        reporter = mmtools.multistate.MultiStateReporter(
            nc_path, open_mode='r', checkpoint_storage=nc_checkpoint_file)

        # Determine if system is periodic
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: %s', is_periodic)

        # Get dimensions
        # Assume full iteration until proven otherwise
        last_checkpoint = True
        trajectory_storage = reporter._storage_checkpoint
        if not keep_solvent:
            # If tracked solute particles, use any last iteration, set with
            # this logic test
            full_iteration = len(reporter.analysis_particle_indices) == 0
            if not full_iteration:
                trajectory_storage = reporter._storage_analysis
                topology = topology.subset(reporter.analysis_particle_indices)

        n_iterations = reporter.read_last_iteration(
            last_checkpoint=last_checkpoint)
        n_frames = trajectory_storage.variables['positions'].shape[0]
        n_atoms = trajectory_storage.variables['positions'].shape[2]
        logger.info('Number of frames: %s, atoms: %s', n_frames, n_atoms)

        # Determine frames to extract.
        # Convert negative indices to last indices.
        if start_frame < 0:
            start_frame = n_frames + start_frame
        if end_frame < 0:
            end_frame = n_frames + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from %s to %s every %s', start_frame,
                    end_frame, skip_frame)

        # Discard equilibration samples
        if discard_equilibration:
            u_n = yank.analyze.extract_u_n(reporter._storage_analysis)
            # Discard frame 0 with minimized energy which throws off automatic
            # equilibration detection.
            n_equil_iterations, _, n_eff = timeseries.detectEquilibration(
                u_n[1:])
            n_equil_iterations += 1
            logger.info(
                ('Discarding initial %s equilibration samples (leaving %s '
                 'effectively uncorrelated samples)...', n_equil_iterations,
                 n_eff))
            # Find first frame post-equilibration.
            if not full_iteration:
                for iteration in range(n_equil_iterations, n_iterations):
                    n_equil_frames = reporter._calculate_checkpoint_iteration(
                        iteration)
                    if n_equil_frames is not None:
                        break
            else:
                n_equil_frames = n_equil_iterations
            frame_indices = frame_indices[n_equil_frames:-1]

        # Determine the number of frames that the trajectory will have.
        if state_index is None:
            n_trajectory_frames = len(frame_indices)
        else:
            # With SAMS, an iteration can have 0 or more replicas in a given
            # state. Deconvolute state indices.
            state_indices = [None for _ in frame_indices]
            for i, iteration in enumerate(frame_indices):
                replica_indices = reporter._storage_analysis.variables[
                    'states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0]
            n_trajectory_frames = sum(len(x) for x in state_indices)

        # Initialize positions and box vectors arrays.
        # MDTraj Cython code expects float32 positions.
        positions = np.zeros((n_trajectory_frames, n_atoms, 3),
                             dtype=np.float32)
        if is_periodic:
            box_vectors = np.zeros((n_trajectory_frames, 3, 3),
                                   dtype=np.float32)

        # Extract state positions and box vectors.
        if state_index is not None:
            logger.info('Extracting positions of state %s...', state_index)

            # Extract state positions and box vectors.
            frame_idx = 0
            for i, iteration in enumerate(frame_indices):
                for r in state_indices[i]:
                    positions[frame_idx, :, :] = trajectory_storage.variables[
                        'positions'][iteration, r, :, :].astype(np.float32)
                    if is_periodic:
                        box_vectors[
                            frame_idx, :, :] = trajectory_storage.variables[
                                'box_vectors'][iteration,
                                               r, :, :].astype(np.float32)
                    frame_idx += 1

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica %s...', replica_index)

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = trajectory_storage.variables['positions'][
                    iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = trajectory_storage.variables[
                        'box_vectors'][iteration,
                                       replica_index, :, :].astype(np.float32)
    finally:
        if reporter is not None:
            reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    # trajectory = mdtraj.Trajectory(positions, topology)
    if not isinstance(topology, mdtraj.core.topology.Topology):
        topology = mdtraj.Topology.from_openmm(topology)
    trajectory = mdtraj.Trajectory(positions, topology)
    if is_periodic:
        trajectory.unitcell_vectors = box_vectors

    # Force periodic boundary conditions to molecules positions
    if image_molecules and is_periodic:
        logger.info(
            'Applying periodic boundary conditions to molecules positions...')
        # Use the receptor as an anchor molecule.
        anchor_atom_indices = set(topography.receptor_atoms)
        if len(anchor_atom_indices) == 0:  # Hydration free energy.
            anchor_atom_indices = set(topography.solute_atoms)
        anchor_molecules = [{
            a
            for a in topology.atoms if a.index in anchor_atom_indices
        }]
        trajectory.image_molecules(inplace=True,
                                   anchor_molecules=anchor_molecules)
    elif image_molecules:
        logger.warning(
            'The molecules will not be imaged because the system is non-periodic.'
        )

    return trajectory


def extract_trajectory_to_file(  # pylint: disable=R0912,R0913,R0914,R0915
        filename, ref_system, top, nc_path, **kwargs):
    """
    Extract trajectory from the NetCDF4 and save to `filename`.

    The format is determined by the filename extension.

    Parameters
    ----------
    filename : str
        Path to output trajectory file.
    ref_system : System object
        Reference state System object.
    top : Topography or Topology object
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one
        chosen by the nc_path file. Default: None
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None
        (default is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).
    ligand_atoms : iterable or int or str, optional
        The atomic indices of the ligand. A string is interpreted as a mdtraj
        DSL specification. Needed for applying pbc using a receptor as anchor.
    solvent_atoms : iterable of int or str, optional
        The atom indices of the solvent. A string is interpreted as an mdtraj
        DSL specification of the solvent atoms. Needed to recenter with pbc
        around solute molecules if image_molecules=True. If 'auto', a list of
        common solvent residue names will be used to automatically detect
        solvent atoms (default is 'auto').

    """
    trj = extract_trajectory(ref_system, top, nc_path, **kwargs)
    trj.save(filename)
