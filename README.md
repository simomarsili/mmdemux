# mmdemux

Motivation: the `extract_trajectory` function in the current version of
`yank` (0.25.2) is not compatible with the replica exchange trajectory files
generated by the current version of `openmmtools` (0.20.0)
(check [this issue](https://github.com/choderalab/openmmtools/issues/487)).

Install: clone the repo and install in a virtual environment using
`pip install .`

Needs `yank`. For a full list of dependencies, check the `requirements.yml`
file.

Extract frames corresponding to state index 0:

```python
from mmdemux import extract_trajectory, extract_trajectory_to_file

trj = extract_trajectory(ref_system=<reference system>, top=<topology>,
nc_path=<path to the NetCD4 file>, state_index=0)

```

`extract_trajectory` returns a mdtraj `Trajectory` object.

`extract_trajectory_to_file` will save the trajectory to file with the file
format determined by the filename extension.

Write frames corresponding to state index 0 to a NetCDF file `trj.nc`:

```python
extract_trajectory_to_file('trj.nc', ref_system=<reference system>,
top=<topology>, nc_path=<path to the NetCD4 file>, state_index=0)
```
