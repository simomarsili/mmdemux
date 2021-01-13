# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Tests for the extract_trajectory function."""
from pathlib import Path

from openmmtools import testsystems

from mmdemux import extract_trajectory


def base_dir():
    """Path to the test dir"""
    return Path(__file__).parent.absolute()


TESTSYS = testsystems.AlanineDipeptideImplicit()
NC_PATH = base_dir() / 'test_repex.nc'

kwargs = dict(ref_system=TESTSYS.system, top=TESTSYS.topology, nc_path=NC_PATH)


def test_state():
    trj = extract_trajectory(state_index=0, **kwargs)
    assert trj.n_frames == 11


def test_replica():
    trj = extract_trajectory(replica_index=0, **kwargs)
    assert trj.n_frames == 11


def test_index_out_of_bounds():
    trj = extract_trajectory(state_index=3, **kwargs)
    assert trj.n_frames == 0


def test_from_system_file():
    kw = dict(kwargs)
    kw['ref_system'] = 'test_repex.xml'
    trj = extract_trajectory(state_index=0, **kw)
    assert trj.n_frames == 11


def test_from_pdb_file():
    kw = dict(kwargs)
    kw['top'] = 'test_repex.pdb'
    trj = extract_trajectory(state_index=0, **kw)
    assert trj.n_frames == 11


if __name__ == '__main__':
    test_state()
