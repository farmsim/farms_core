# Sensors data structures

<a id="ref-SensorsData"></a>
## SensorsData

Contains the sensors data extracted from the physics engine.

**Attributes:**

- `links` (`LinkSensorArray`): Links data.
- `joints` (`JointSensorArray`): Joints data.
- `contacts` (`ContactsArray`): Contacts data.
- `xfrc` (`XfrcArray`): External forces data.
- `muscles` (`MusclesArray`): Muscles data.
- `adhesions` (`AdhesionsArray`): Adhesion forces data.
- `visuals` (`VisualsArray`): Visuals data.

<a id="ref-LinkSensorArray"></a>
## LinkSensorArray

Links positions, orientations, velocities, angular velocities, ...

**Attributes:**

- `names` (`list[str]`): List of links names, in order of indices in the array
- `array` (`ndarray`): Array containing the links data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.
- `masses` (`list[float]`): Links masses.

**Methods:**

- `com_ang_velocity`: CoM angular velocity of a link
- `com_lin_velocities`: CoM linear velocities
- `com_lin_velocity`: CoM linear velocity of a link
- `com_orientation`: CoM orientation of a link
- `com_position`: CoM position of a link
- `com_positions`: CoM position of a link
- `global_com_position`: Global CoM position
- `heading`: Heading
- `plot`: Plot
- `plot_base_position`: Plot
- `plot_base_velocity`: Plot
- `plot_heading`: Plot
- `to_dict`: Convert data to dictionary
- `urdf_orientation`: Orientation of a link's frame
- `urdf_orientations`: Orientation of multiple links' frames
- `urdf_position`: Position of a link's frame
- `urdf_positions`: Position of multiple links' frames

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

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

**Attributes:**

- `names` (`list[str]`): List of joints names, in order of indices in the array
- `array` (`DoubleArray3D`): Array containing the joints data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

**Methods:**

- `active`: Active torque
- `active_torques`: Active torques
- `cmd_position`: Joint position
- `cmd_positions`: Joint position
- `cmd_torque`: Joint torque
- `cmd_torques`: Joint torque
- `cmd_velocities`: Joint velocity
- `cmd_velocity`: Joint velocity
- `commanded_power`: Compute mechanical power
- `damping`: passive damping torque
- `damping_torques`: Damping torques
- `force`: Joint force
- `forces_all`: Joints forces
- `friction`: passive friction torque
- `friction_torques`: Friction torques
- `limit_force`: Joint limit force
- `limit_forces_all`: Joints limits forces
- `mechanical_power`: Compute mechanical power
- `mechanical_power_active`: Compute active mechanical power
- `motor_torque`: Joint torque
- `motor_torques`: Joint torques
- `motor_torques_all`: Joint torque
- `plot`: Plot
- `plot_active_torques`: Plot joints active torques
- `plot_cmd_positions`: Plot joints command positions
- `plot_cmd_torques`: Plot joints command torques
- `plot_cmd_velocities`: Plot joints command velocities
- `plot_damping_torques`: Plot joints damping torques
- `plot_data`: Plot data
- `plot_end`: plot_end
- `plot_forces`: Plot ground reaction forces
- `plot_friction_torques`: Plot joints friction torques
- `plot_generic`: Plot joint sensor
- `plot_generic_3`: Plot ground reaction forces
- `plot_mechanical_power_total`: Plot ground reaction torques
- `plot_mechanical_powers`: Plot ground reaction torques
- `plot_motor_torques`: Plot joints motor torques
- `plot_positions`: Plot ground reaction forces
- `plot_spring_torques`: Plot joints spring torques
- `plot_torques`: Plot ground reaction torques
- `plot_velocities`: Plot ground reaction forces
- `position`: Joint position
- `positions`: Joints positions
- `positions_all`: Joints positions
- `spring`: Passive spring torque
- `spring_torques`: Spring torques
- `to_dict`: Convert data to dictionary
- `torque`: Joint torque
- `torques_all`: Joints torques
- `velocities`: Joints velocities
- `velocities_all`: Joints velocities
- `velocity`: Joint velocity

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

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

**Attributes:**

- `names` (`list[str]`): List of contacts names, in order of indices in the array
- `array` (`DoubleArray3D`): Array containing the contacts data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

**Methods:**

- `friction`: Friction force
- `friction_all`: Friction forces
- `frictions`: Friction forces
- `plot`: Plot
- `plot_friction_forces`: Plot friction forces
- `plot_friction_forces_ori`: Plot friction forces
- `plot_ground_reaction_forces`: Plot ground reaction forces
- `plot_ground_reaction_forces_all`: Plot ground reaction forces
- `plot_total_forces`: Plot contact forces
- `position`: Position
- `position_all`: Positions
- `reaction`: Reaction force
- `reaction_all`: Reaction forces
- `reactions`: Reaction forces
- `to_dict`: Convert data to dictionary
- `total`: Total force
- `total_all`: Total forces
- `totals`: Total forces

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

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

**Attributes:**

- `names` (`list[str]`): List of external forces names, in order of indices in the array
- `array` (`DoubleArray3D`): Array containing the external forces data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

**Methods:**

- `force`: Force
- `forces`: Forces
- `plot`: Plot
- `plot_forces`: Plot
- `plot_torques`: Plot
- `set_force`: Set force
- `set_torque`: Set torque
- `to_dict`: Convert data to dictionary
- `torque`: Torque
- `torques`: Torques

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

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

**Attributes:**

- `names` (`list[str]`): List of muscles names, in order of indices in the array
- `array` (`DoubleArray3D`): Array containing the muscles data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

**Methods:**

- `II_feedback`:  Type II feedback  of a muscle at iteration 
- `II_feedbacks`:  Type II feedback of all muscles at iteration 
- `II_feedbacks_all`:  Type II feedback of all muscles 
- `Ia_feedback`:  Type Ia feedback  of a muscle at iteration 
- `Ia_feedbacks`:  Type Ia feedback of all muscles at iteration 
- `Ia_feedbacks_all`:  Type Ia feedback of all muscles 
- `Ib_feedback`:  Type Ib feedback  of a muscle at iteration 
- `Ib_feedbacks`:  Type Ib feedback of all muscles at iteration 
- `Ib_feedbacks_all`:  Type Ib feedback of all muscles 
- `activation`:  Muscle activation of a muscle at iteration 
- `activations`:  Muscle activations of all muscles at iteration 
- `activations_all`:  Muscle activations of all muscles 
- `active_force`:  Muscle active force of a muscle at iteration 
- `active_forces`:  Muscle active forces of all muscles at iteration 
- `active_forces_all`:  Muscle active forces of all muscles 
- `excitation`:  Muscle excitation of a muscle at iteration 
- `excitations`:  Muscle excitations of all muscles at iteration 
- `excitations_all`:  Muscle excitations of all muscles 
- `fiber_length`:  Muscle fiber length of a muscle at iteration 
- `fiber_lengths`:  Muscle fiber lengths of all muscles at iteration 
- `fiber_lengths_all`:  Muscle fiber lengths of all muscles 
- `fiber_velocities`:  Muscle fiber velocities of all muscles at iteration 
- `fiber_velocities_all`:  Muscle fiber velocities of all muscles 
- `fiber_velocity`:  Muscle fiber velocity of a muscle at iteration 
- `mtu_force`:  Muscle tendon unit force of a muscle at iteration 
- `mtu_forces`:  Muscle tendon unit forces of all muscles at iteration 
- `mtu_forces_all`:  Muscle tendon unit forces of all muscles 
- `mtu_length`:  Muscle tendon unit length of a muscle at iteration 
- `mtu_lengths`:  Muscle tendon unit lengths of all muscles at iteration 
- `mtu_lengths_all`:  Muscle tendon unit lengths of all muscles 
- `mtu_velocities`:  Muscle tendon unit velocities of all muscles at iteration 
- `mtu_velocities_all`:  Muscle tendon unit velocities of all muscles 
- `mtu_velocity`:  Muscle tendon unit velocity of a muscle at iteration 
- `passive_force`:  Muscle passive force of a muscle at iteration 
- `passive_forces`:  Muscle passive forces of all muscles at iteration 
- `passive_forces_all`:  Muscle passive forces of all muscles 
- `tendon_force`:  Tendon unit force of a muscle at iteration 
- `tendon_forces`:  Tendon unit forces of all muscles at iteration 
- `tendon_forces_all`:  Tendon unit forces of all muscles 
- `tendon_length`:  Tendon unit length of a muscle at iteration 
- `tendon_lengths`:  Tendon unit lengths of all muscles at iteration 
- `tendon_lengths_all`:  Tendon unit lengths of all muscles 
- `to_dict`: Convert data to dictionary

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

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

**Attributes:**

- `names` (`list[str]`): List of adhesions names, in order of indices in the array
- `array` (`DoubleArray3D`): Array containing the adhesions data, refer to the `farms_core/sensor/sensor_convention` for information about the indices.

**Methods:**

- `force`: Adhesion force
- `plot`: Plot
- `to_dict`: Convert data to dictionary

**Size and indices:**

Note: It is recommended to not use indices directly, but to favour accessing the data using the provided methods, or the sensor convention definitions provided in  ´farms_core/sensors/sensor_convention´.

| Key     |   Value |
|---------|---------|
| `size`  |       1 |
| `force` |       0 |