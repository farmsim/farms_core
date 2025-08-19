<a id='ref-ExperimentData'></a>
# ExperimentData

Provides and logs the experiment data.

- `times` ([DoubleArray1D](#ref-DoubleArray1D)): The vector of logged times across all data.
- `timestep` (`float`): The simulation timestep (Must be positive).
- `simulation` ([SimulationData](#ref-SimulationData)): The simulation data, mostly related to the physics engine.
- `simulation` (list[[AnimatData](#ref-AnimatData)]): List of data for each animat

<a id='ref-SimulationData'></a>
## SimulationData

Provides and logs the simulation data.

- `ncon` (`1DArray[int, [n_iterations]]`): Number of constraints during iteration.
- `niter` (`1DArray[int, [n_iterations]]`): Number of physics engine iterations.
- `energy` (`2DArray[int, [2, n_iterations]]`): Potential (first index) and kinetic (second index) energy.

<a id='ref-AnimatData'></a>
## AnimatData

Provides and logs the animat data.

- `sensors` ([SensorsData](#ref-SensorsData)): Contains the logged sensors data.

<a id='ref-SensorsData'></a>
### SensorsData

Contains the sensors data extracted from the physics engine.

- `links` ([LinkSensorArray](#ref-LinkSensorArray)): Links data.
- `joints` ([JointSensorArray](#ref-JointSensorArray)): Joints data.
- `contacts` ([ContactsArray](#ref-ContactsArray)): Contacts data.
- `xfrc` ([XfrcArray](#ref-XfrcArray)): External forces data.
- `muscles` ([MusclesArray](#ref-MusclesArray)): Muscles data.
- `adhesions` ([AdhesionsArray](#ref-AdhesionsArray)): Adhesion forces data.
- `visuals` ([VisualsArray](#ref-VisualsArray)): Visuals data.

<a id='ref-LinkSensorArray'></a>
#### LinkSensorArray

Links positions, orientations, velocities, angular velocities, ...

- `names` (`list[str]`): List of links names, in order of indices in the array
- `array` (`ndarray`): Array containing the links data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.
- `masses` (`list[float]`): Links masses.

<a id='ref-JointSensorArray'></a>
#### JointSensorArray

Joints positions, velocities, forces, commands, ...

- `names` (`list[str]`): List of joints names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the joints data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

<a id='ref-ContactsArray'></a>
#### ContactsArray

Contacts forces, torques, contact position, ...

- `names` (`list[str]`): List of contacts names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the contacts data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

<a id='ref-XfrcArray'></a>
#### XfrcArray

External forces and torques (typically used for e.g. perturbations or for implementing custom hydrodynamics) 

- `names` (`list[str]`): List of external forces names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the external forces data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

<a id='ref-MusclesArray'></a>
#### MusclesArray

Muscle length, velocity, force, ...

- `names` (`list[str]`): List of muscles names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the muscles data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

<a id='ref-AdhesionsArray'></a>
#### AdhesionsArray

Adhesion forces

- `names` (`list[str]`): List of adhesions names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the adhesions data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

<a id='ref-VisualsArray'></a>
#### VisualsArray

Visuals colors and lights

- `names` (`list[str]`): List of visuals names, in order of indices in the array
- `array` ([DoubleArray3D](#ref-DoubleArray3D)): Array containing the visuals data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

