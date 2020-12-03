# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Tests for the extract_trajectory function."""
from pathlib import Path

from openmmtools import testsystems

from mmdemux import extract_trajectory

TESTSYS = testsystems.AlanineDipeptideVacuum()
NC_PATH = Path('test_repex_sim.nc')
NC_CHK = None

kwargs = dict(ref_system=TESTSYS.system, top=TESTSYS.topology, nc_path=NC_PATH)


def test_state():
    trj = extract_trajectory(state_index=0, **kwargs)
    assert trj.n_frames == 51


def test_replica():
    trj = extract_trajectory(replica_index=0, **kwargs)
    assert trj.n_frames == 51


def test_index_out_of_bounds():
    trj = extract_trajectory(state_index=3, **kwargs)
    assert trj.n_frames == 0
