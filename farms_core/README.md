<a id="ref-ExperimentOptions"></a>
# ExperimentOptions

Describes the animat properties.

- `simulation` ([SimulationOptions](#ref-SimulationOptions)): The simulation options.
- `animats` (list[[AnimatOptions](#ref-AnimatOptions)]): List of animats options.
- `arenas` (list[[ArenaOptions](#ref-ArenaOptions)]): List of animats options.

<a id="ref-SimulationOptions"></a>
## SimulationOptions

Describes the simulation options.

- `units` ([SimulationUnitScaling](#ref-SimulationUnitScaling)): The simulation units used in the physics engine (All parameters defined in the config files are in SI units).
- `timestep` (`float`): The simulation timestep (Must be positive).
- `n_iterations` (`int`): The number of simulations iterations to run (Must be positive)
- `buffer_size` (`int`): The number of simulations itertions buffered, after which the data will be overwritten.
- `play` (`bool`): Whether to play the simulation, or keep it paused when started. Only relevent when running the simulation in interactive mode.
- `rtl` (`float`): Real-time limiter to limit the simulation speed. Only relevent when running the simulation in interactive mode.
- `fast` (`bool`): Whether to run the simulation as fast as possible, bypasses rtl.
- `headless` (`bool`): Whether to run the simulation headless, with no. external interaction.
- `show_progress` (`bool`): Whether to display a progress bar.
- `zoom` (`float`): Camera zoom.
- `free_camera` (`bool`): Whether the camera should be free moving instead of following the animat.
- `top_camera` (`bool`): Whether the camera should look at the animat from above.
- `rotating_camera` (`bool`): Whether the camera should turn around the model.
- `video` (`str`): Path to where the video should be saved. Empty string to disable recording.
- `video_fps` (`str`): Path to where the video should be saved. Empty string to disable recording.
- `video_speed` (`float`): Speed factor at which the video should be played.
- `video_name` (`str`): Video name.
- `video_yaw` (`float`): Video yaw angle.
- `video_pitch` (`float`): Video yaw pitch.
- `video_distance` (`float`): Video distance from animat.
- `video_offset` (`float`): Video position offset with respect to animat.
- `video_filter` (`float`): Video motion filter.
- `video_resolution` (`list[int]`): Video resolution (e.g. [1280, 720]).
- `gravity` (`list[float]`): Gravity vector (e.g. [0, 0, -9.81]).
- `num_sub_steps` (`int`): Number of physics substeps.
- `cb_sub_steps` (`int`): Number of callback substeps.
- `n_solver_iters` (`int`): Number of maximum solver iterations per step.
- `residual_threshold` (`float`): Residual threshold (e.g. 1e-6).
- `visual_scale` (`float`): Visual scale.
- `mujoco` ([MuJoCoSimulationOptions](#ref-MuJoCoSimulationOptions)): MuJoCo options.
- `pybullet` ([PybulletSimulationOptions](#ref-PybulletSimulationOptions)): Pybullet options.

<a id="ref-SimulationUnitScaling"></a>
### SimulationUnitScaling

Simulation units scaling used inside the physics engine. These can be useful for avoiding numerical computation artifacts for very small or very large models.

- `meters` (`float`): The length unit (Must be positive).
- `seconds` (`float`): The time unit (Must be positive).
- `kilograms` (`float`): The mass unit (Must be positive).

<a id="ref-MuJoCoSimulationOptions"></a>
### MuJoCoSimulationOptions

Describes the MuJoCo simulation options.  These options are for the MuJoCo physics engine, refer to [MuJoCo's documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html) for additional information.

- `cone` (`str`): Friction cone (e.g. pyramidal or elliptic).
- `solver` (`str`): Physics solver (e.g. PGS, CG or Newton).
- `integrator` (`str`): Physics integrator (e.g. Euler, RK4, implicit, implicitfast).
- `impratio` (`float`): Frictional-to-normal constraint impedance.
- `ccd_iterations` (`int`): Convex Collision Detection (CCD) iterations.
- `ccd_tolerance` (`float`): Convex Collision Detection (CCD) tolerance.
- `noslip_iterations` (`int`): No slip iterations.
- `noslip_tolerance` (`float`): No slip tolerance.
- `texture_repeat` (`int`): Repeating texture.
- `shadow_size` (`int`): Shadow size.
- `extent` (`float`): View extent.

<a id="ref-PybulletSimulationOptions"></a>
### PybulletSimulationOptions

Describes the Pybullet simulation options.  These options are for the Bullet physics engine, refer to [Pybullet's documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA) for additional information.

- `opengl2` (`bool`): Whether to use OpenGL2 instead of OpenGL3.
- `lcp` (`str`): Linear Complementarity Problem (LCP) constraint solver (e.g. dantzig).
- `cfm` (`float`): Constraint Force Mixing (CFM).
- `erp` (`float`): Error Reduction Parameter (ERP).
- `contact_erp` (`float`): Contact Error Reduction Parameter (ERP).
- `friction_erp` (`float`): Friction Error Reduction Parameter (ERP).
- `max_num_cmd_per_1ms` (`int`): Max number of commands per 1ms.
- `report_solver_analytics` (`int`): Whether to report the solver analytics.

<a id="ref-AnimatOptions"></a>
## AnimatOptions

Describes the animat properties.

- `sdf` (`str`): Path to SDF file.
- `spawn` ([SpawnOptions](#ref-SpawnOptions)): Provides options for spawning the animat.
- `morphology` ([MorphologyOptions](#ref-MorphologyOptions)): Provides animat morphology options.
- `control` ([ControlOptions](#ref-ControlOptions)): Provides control options.

<a id="ref-SpawnOptions"></a>
### SpawnOptions

Describes the spawn properties.

- `loader` ([SpawnLoader](#ref-SpawnLoader)): Choses the loader for loading the animat.
- `mode` ([SpawnMode](#ref-SpawnMode)): Mode and constraints to apply to spawn.
- `pose` (`list[float]`): Spawn pose ([X, Y, Z, Rx, Ry, Rz]).
- `pose` (`list[float]`): Spawn velocity ([Vx, Vy, Vz, Wx, Wy, Wz]).
- `extras` (`dict`): Extra options (Deprecated).

<a id="ref-SpawnLoader"></a>
#### SpawnLoader

Recommened to use the FARMS loader.

- `FARMS` (`IntEnum`): FARMS loader (Default).
- `PYBULLET` (`IntEnum`): Use Pybullet's SDF loader.

<a id="ref-SpawnMode"></a>
#### SpawnMode

Spawn mode providing the ability to apply constraints. Default is FREE. The constraints apply to the base link.

- `FREE` (`Enum`): Free-floating body
- `FIXED` (`Enum`): Fixed base link
- `ROTX` (`Enum`): Rotate along X-axis
- `ROTY` (`Enum`): Rotate along Y-axis
- `ROTZ` (`Enum`): Rotate along Z-axis
- `SAGITTAL` (`Enum`): Move along sagital plane (XZ, rotate around Y)
- `SAGITTAL0` (`Enum`): Move along sagital plane (XZ, no rotations)
- `SAGITTAL3` (`Enum`): Move along sagital plane (XZ, all rotations)
- `CORONAL` (`Enum`): Move along coronal plane (YZ, rotate around X)
- `CORONAL0` (`Enum`): Move along coronal plane (YZ, no rotations)
- `CORONAL3` (`Enum`): Move along coronal plane (YZ, all rotations)
- `TRANSVERSE` (`Enum`): Move along transversal plane (XY, rotate around Z)
- `TRANSVERSE0` (`Enum`): Move along transversal plane (XY, no rotations)
- `TRANSVERSE3` (`Enum`): Move along transversal plane (XY, all rotations)

<a id="ref-MorphologyOptions"></a>
### MorphologyOptions

Describes the morphological properties.

- `links` (list[[LinkOptions](#ref-LinkOptions)]): Provides options for each link.
- `joints` (list[[JointOptions](#ref-JointOptions)]): Provides options for each joint.

<a id="ref-LinkOptions"></a>
#### LinkOptions

Describes the link properties.

- `name` (`str`): The link name from the SDF file.
- `collisions` (`bool`): Whether the link should collide.
- `friction` (`list[float]`): A list of values describing the friction coefficients.
- `extras` (`dict`): Extra options (Deprecated).

<a id="ref-JointOptions"></a>
#### JointOptions

Describes the joint properties.

- `name` (`str`): The joint name from the SDF file.
- `initial` (`list[float]`): A list 2 float values describing the initial position (rad) and velocity (rad/s).
- `limits` (`list[float]`): Two lists of 2 float values describing the limit value of the joint (the first value must be smaller than or equal to the second).
- `stiffness` (`float`): The spring stifness value (Nm/rad for revolute joints).
- `springref` (`float`): The spring reference value (rad for revolute joints).
- `damping` (`float`): The damping value (N·m·s/rad for revolute joints).
- `extras` (`dict`): Extra options (Deprecated).

<a id="ref-ControlOptions"></a>
### ControlOptions

Describes the control options.

- `sensors` ([SensorsOptions](#ref-SensorsOptions)): Sensors options.
- `motors` (list[[MotorOptions](#ref-MotorOptions)]): List of options for each joint actuator.
- `hill_muscles` (list[[MuscleOptions](#ref-MuscleOptions)]): List of options for each Hill-type muscle.

<a id="ref-SensorsOptions"></a>
#### SensorsOptions

Describes the sensor options.

- `links` (`list[str]`): List of links to track.
- `joints` (`list[str]`): List of joints to track.
- `contacts` (`list[str] | list[list[str]]`): List of links / link pairs to track contacts.
- `xfrc` (`list[str]`): List of links to track external forces.
- `muscles` (`list[str]`): List of muscles to track.
- `adhesions` (`list[str]`): List of adhesions to track.
- `visuals` (`list[str]`): List of visuals to track.

<a id="ref-MotorOptions"></a>
#### MotorOptions

Describes the motor options.

- `joint_name` (`str`): Joint name to actuate.
- `control_types` (`list[str]`): List of control types (position/velocity/torque).
- `limits_torques` (`list[float]`): List of torques limits ([min, max])
- `gains` (`list[float]`): Proportional and Derivative gain ([Kp, Kd]) for position control. Proceed with caution when using this for velocity and torque contol.

<a id="ref-MuscleOptions"></a>
#### MuscleOptions

Describes the properties of Hill-type muscles.

- `name` (`str`): Muscle name.
- `model` (`str`): Muscle model.
- `max_force` (`float`): Maximum force the muscle can exert.
- `optimal_fiber` (`float`): Optimal fiber length.
- `tendon_slack` (`float`): Tendon slack.
- `max_velocity` (`float`): Maximum velocity.
- `pennation_angle` (`float`): Pennation angle.
- `lmtu_min` (`float`): Minimum muscle-tendon unit length.
- `lmtu_max` (`float`): Maximum muscle-tendon unit length.
- `waypoints` (`list[list[float]]`): Muscle waypoints.
- `act_tconst` (`float`): Activation time constant.
- `deact_tconst` (`float`): Deactivation time constant.
- `lmin` (`float`): Minimum muscle length.
- `lmax` (`float`): Maximum muscle length.
- `init_activation` (`float`): Initial activation.
- `init_fiber` (`float`): Initial fiber length.
- `type_I_kv` (`float`): Type I kv.
- `type_I_pv` (`float`): Type I pv.
- `type_I_k_dI` (`float`): Type I k dI.
- `type_I_k_nI` (`float`): Type I k nI.
- `type_I_const_I` (`float`): Type I constant I.
- `type_I_l_ce_th` (`float`): Type I l ce th.
- `type_Ib_kf` (`float`): Type Ib kf.
- `type_II_k_dII` (`float`): Type II k dII.
- `type_II_k_nII` (`float`): Type II k nII.
- `type_II_const_II` (`float`): Type II constant II.
- `type_II_l_ce_th` (`float`): Type II l ce th.

<a id="ref-ArenaOptions"></a>
## ArenaOptions

Describes the arena properties.

- `sdf` (`str`): Path to SDF file.
- `spawn` ([SpawnOptions](#ref-SpawnOptions)): Provides options for spawning the animat.
- `water` ([WaterOptions](#ref-WaterOptions)): Provides water options.
- `ground_height` (`float`): Height offset at which to place the arena.

<a id="ref-WaterOptions"></a>
### WaterOptions

Describes the water options.

- `sdf` (`str`): Path to SDF file.
- `drag` (`bool`): Whether to apply drag forces.
- `buoyancy` (`bool`): Whether to apply buoyancy forces.
- `height` (`float`): Height of water surface.
- `velocity` (`float`): Water velocity in 3 directions [Vx, Vy, Vz].
- `viscosity` (`float`): Viscosity of water.
- `density` (`float`): Density of water.
- `maps` (`list[str]`): Provides water maps from images.

