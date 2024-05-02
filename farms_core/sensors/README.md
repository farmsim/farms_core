# Simulation data structures

## SensorsData

Contains the sensors data logged from the physics engine.

```
SensorsData
- links [LinkSensorArray]: Links positions, orientations, velocities, angular velocities, ...
- joints [JointSensorArray]: Joints positions, velocities, forces, commands, ...
- contacts [ContactsArray]: Contacts forces, torques, contact position, ...
- external forces [XfrcArray]: External forces and torques (typically used for e.g. perturbations or for implementing custom hydrodynamics) 
- muscles [MusclesArray]: Muscle length, velocity, force, ...
```

## LinkSensorArray

Links positions, orientations, velocities, angular velocities, ...

```
LinkSensorArray
- names [List]: List of links names, in order of indices in the array
- array [NDArray]: Array containing the links data, refer to the farms_core/sensor/sensor_convention for information about indices.
```

## JointSensorArray

Joints positions, velocities, forces, commands, ...

```
JointSensorArray
- names [List]: List of joints names, in order of indices in the array
- array [DoubleArray3D]: Array containing the joints data, refer to the farms_core/sensor/sensor_convention for information about indices.
```

## ContactsArray

Contacts forces, torques, contact position, ...

```
ContactsArray
- names [List]: List of contacts names, in order of indices in the array
- array [DoubleArray3D]: Array containing the contacts data, refer to the farms_core/sensor/sensor_convention for information about indices.
```

## XfrcArray

External forces and torques (typically used for e.g. perturbations or for implementing custom hydrodynamics) 

```
XfrcArray
- names [List]: List of external forces names, in order of indices in the array
- array [DoubleArray3D]: Array containing the external forces data, refer to the farms_core/sensor/sensor_convention for information about indices.
```

## MusclesArray

Muscle length, velocity, force, ...

```
MusclesArray
- names [List]: List of muscles names, in order of indices in the array
- array [DoubleArray3D]: Array containing the muscles data, refer to the farms_core/sensor/sensor_convention for information about indices.
```