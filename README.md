# mmdemux

Porting the `extract_trajectory` function from `yank` to `openmmtools`.

Motivation: the `extract_trajectory` function in the current version of
`yank` (0.25.2) is not compatible with the replica exchange trajectory files
generated by the current version of `openmmtools` (0.20.0)

Install: clone the repo and install in a virtual environment using pip:

```
$ pip install .
```

Usage:

```python
from mmdemux import extract_trajectory, extract_trajectory_to_file

trj = extract_trajectory(ref_system=<system>, top=<topology>, nc_path=<path to the NetCD4 file>)

extract_trajectory_to_file(<filename>, ref_system=<system>, top=<topology>, nc_path=<path to the NetCD4 file>)

```

`extract_trajectry` will return a `mdtraj.Trajectory object`.


`extract_trajectory_to_file` will save the trajectory to `filename` with the
file format determined by the filename extension.



