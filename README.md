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
controller and simulation using YAML configuration files. For more information,
please refer to [this README](farms_core/README.md). All the configuration files
can be loaded, modified and saved as follows:

```python
from farms_core.experiment.options import ExperimentOptions
from model.options import AnimatOptions, ArenaOptions

# Load data from a logged HDF5 file after running a simulation
experiment_options = ExperimentOptions.load(path_to_existing_file)

# Your modifications to the experiment options here
# ...

# Save modified data
experiment_options.save(path_to_new_file)
```

## Experiment data structures

FARMS automatically extracts sensors data in a simulation and provides an
interface to access it. For information related to the data structures, please
refer to [the experiments' README](farms_core/experiment/README.md). For
additional information about the sensors (e.g. sensor array indices and
methods), please refer to [the sensors' README](farms_core/sensors/README.md).
Al the experiment data is automatically logged and can be saved to disk via
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) for later analysis. You can use
the following to load the data back:

```python
from farms_core.experiment.data import ExperimentData
data = ExperimentData.from_file(path_to_data_file)

# Code for analyzing data
# ...

# Data can also be modified and saved if necessary (e.g. reducing the size of
# the data)
data.to_file(path_to_save_file)
```
