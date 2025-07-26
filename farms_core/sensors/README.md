# Simulation data structures

<a id="ref-SensorsData"></a>
## SensorsData

Contains the sensors data logged from the physics engine.

```
SensorsData
- links (LinkSensorArray): Links positions, orientations, velocities, angular velocities, ...
- joints (JointSensorArray): Joints positions, velocities, forces, commands, ...
- contacts (ContactsArray): Contacts forces, torques, contact position, ...
- external forces (XfrcArray): External forces and torques (typically used for e.g. perturbations or for implementing custom hydrodynamics) 
- muscles (MusclesArray): Muscle length, velocity, force, ...
- adhesions (AdhesionsArray): Adhesion forces
- visuals (VisualsArray): Visuals colors and lights
```

<a id="ref-LinkSensorArray"></a>
## LinkSensorArray

Links positions, orientations, velocities, angular velocities, ...

```
LinkSensorArray
- names (list): List of links names, in order of indices in the array
- array (ndarray): Array containing the links data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Link array size and indices:

| Key                  |   Value |
|----------------------|---------|
| `size`               |      20 |
| `com_position_x`     |       0 |
| `com_position_y`     |       1 |
| `com_position_z`     |       2 |
| `com_orientation_x`  |       3 |
| `com_orientation_y`  |       4 |
| `com_orientation_z`  |       5 |
| `com_orientation_w`  |       6 |
| `urdf_position_x`    |       7 |
| `urdf_position_y`    |       8 |
| `urdf_position_z`    |       9 |
| `urdf_orientation_x` |      10 |
| `urdf_orientation_y` |      11 |
| `urdf_orientation_z` |      12 |
| `urdf_orientation_w` |      13 |
| `com_velocity_lin_x` |      14 |
| `com_velocity_lin_y` |      15 |
| `com_velocity_lin_z` |      16 |
| `com_velocity_ang_x` |      17 |
| `com_velocity_ang_y` |      18 |
| `com_velocity_ang_z` |      19 |

<a id="ref-JointSensorArray"></a>
## JointSensorArray

Joints positions, velocities, forces, commands, ...

```
JointSensorArray
- names (list): List of joints names, in order of indices in the array
- array (DoubleArray3D): Array containing the joints data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Joint array size and indices:

| Key                |   Value |
|--------------------|---------|
| `size`             |      17 |
| `position`         |       0 |
| `velocity`         |       1 |
| `torque`           |       2 |
| `force_x`          |       3 |
| `force_y`          |       4 |
| `force_z`          |       5 |
| `torque_x`         |       6 |
| `torque_y`         |       7 |
| `torque_z`         |       8 |
| `cmd_position`     |       9 |
| `cmd_velocity`     |      10 |
| `cmd_torque`       |      11 |
| `torque_active`    |      12 |
| `torque_stiffness` |      13 |
| `torque_damping`   |      14 |
| `torque_friction`  |      15 |
| `limit_force`      |      16 |

<a id="ref-ContactsArray"></a>
## ContactsArray

Contacts forces, torques, contact position, ...

```
ContactsArray
- names (list): List of contacts names, in order of indices in the array
- array (DoubleArray3D): Array containing the contacts data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Contact array size and indices:

| Key          |   Value |
|--------------|---------|
| `size`       |      12 |
| `reaction_x` |       0 |
| `reaction_y` |       1 |
| `reaction_z` |       2 |
| `friction_x` |       3 |
| `friction_y` |       4 |
| `friction_z` |       5 |
| `total_x`    |       6 |
| `total_y`    |       7 |
| `total_z`    |       8 |
| `position_x` |       9 |
| `position_y` |      10 |
| `position_z` |      11 |

<a id="ref-XfrcArray"></a>
## XfrcArray

External forces and torques (typically used for e.g. perturbations or for implementing custom hydrodynamics) 

```
XfrcArray
- names (list): List of external forces names, in order of indices in the array
- array (DoubleArray3D): Array containing the external forces data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Xfrc array size and indices:

| Key        |   Value |
|------------|---------|
| `size`     |       6 |
| `force_x`  |       0 |
| `force_y`  |       1 |
| `force_z`  |       2 |
| `torque_x` |       3 |
| `torque_y` |       4 |
| `torque_z` |       5 |

<a id="ref-MusclesArray"></a>
## MusclesArray

Muscle length, velocity, force, ...

```
MusclesArray
- names (list): List of muscles names, in order of indices in the array
- array (DoubleArray3D): Array containing the muscles data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Muscle array size and indices:

| Key                    |   Value |
|------------------------|---------|
| `size`                 |      15 |
| `excitation`           |       0 |
| `activation`           |       1 |
| `tendon_unit_length`   |       2 |
| `tendon_unit_velocity` |       3 |
| `tendon_unit_force`    |       4 |
| `fiber_length`         |       5 |
| `fiber_velocity`       |       6 |
| `pennation_angle`      |       7 |
| `active_force`         |       8 |
| `passive_force`        |       9 |
| `tendon_length`        |      10 |
| `tendon_force`         |      11 |
| `Ia_feedback`          |      12 |
| `II_feedback`          |      13 |
| `Ib_feedback`          |      14 |

<a id="ref-AdhesionsArray"></a>
## AdhesionsArray

Adhesion forces

```
AdhesionsArray
- names (list): List of adhesions names, in order of indices in the array
- array (DoubleArray3D): Array containing the adhesions data, refer to the farms_core/sensor/sensor_convention for information about the indices.
```

Adhesion array size and indices:

| Key     |   Value |
|---------|---------|
| `size`  |       1 |
| `force` |       0 |