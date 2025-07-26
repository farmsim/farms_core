# FARMS core

FARMS core provides the basic shared functionalities for the FARMS framework,
used commonly by other packages such as:

- [farms_sim](https://github.com/farmsim/farms_sim)
- [farms_mujoco](https://github.com/farmsim/farms_mujoco)
- [farms_network](https://github.com/farmsim/farms_network)
- [farms_muscle](https://github.com/farmsim/farms_muscle)

It mainly includes the following features:

- Data: Interfaces to animat data (see below for details)
- IO: Handling of file formats such as [SDF](http://sdformat.org/),
  [YAML](https://yaml.org/) and
  [HDF5](https://www.hdfgroup.org/solutions/hdf5/).
- Simulation: Definition of simulation options
- Model handling: Definitions and interfaces for animats and robots parameters,
  data and control
- Pylog: A utility for logging and displaying information to the terminal and to
  the disk (`from farms_core import pylog`)
- Units scaling: Provides helpful functions for scaling units related to mass,
  space and time
- Analysis: Functions for analyzing simulations

## Configuration files

FARMS provides functionality for defining settings related to the model,
controller and simulation using YAML configuration files. For information
concerning, please refer to [this README](farms_core/README.md).

## Animat data structures

For information related to the data structures (e.g. sensor array), please refer
to [this README](farms_core/sensors/README.md).
