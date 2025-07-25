"""Parse command line arguments"""

from typing import Any, Type
from argparse import (
    ArgumentParser,
    ArgumentTypeError,
    ArgumentDefaultsHelpFormatter,
)


def positive(value: Any, value_type: Type):
    """Positive value"""
    typed_value = value_type(value)
    if typed_value <= 0:
        raise ArgumentTypeError(f'{value} is not a positive int value')
    return typed_value


def positive_int(value: int):
    """Positive int"""
    return positive(value, value_type=int)


def positive_float(value: float):
    """Positive float"""
    return positive(value, value_type=float)


def config_argument_parser() -> ArgumentParser:
    """Argument parser"""
    parser = ArgumentParser(
        description='FARMS simulation config generation',
        formatter_class=(
            lambda prog:
            ArgumentDefaultsHelpFormatter(prog, max_help_position=50)
        ),
    )

    # Simulation
    parser.add_argument(
        '--timestep',
        type=positive_float,
        default=1e-3,
        help='Simulation timestep',
    )
    parser.add_argument(
        '--duration',
        type=positive_float,
        default=10,
        help='Simulation duration',
    )
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=0,
        help='Buffer size for sensors arrays, 0 for full simulation length',
    )
    parser.add_argument(
        '--pause',
        action='store_true',
        default=False,
        help='Pause simulation at start',
    )
    parser.add_argument(
        '--rtl',
        type=positive_float,
        default=1.0,
        help='Simulation real-time limiter',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        default=False,
        help='Remove real-time limiter',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False,
        help='Headless mode instead of using GUI',
    )
    parser.add_argument(
        '--noprogress', '--npb',
        action='store_false',
        dest='show_progress',
        help='Hide progress bar',
    )

    # Units
    parser.add_argument(
        '--meters',
        type=positive_float,
        default=1,
        help='Unit scaling of meters within physics engine',
    )
    parser.add_argument(
        '--seconds',
        type=positive_float,
        default=1,
        help='Unit scaling of seconds within physics engine',
    )
    parser.add_argument(
        '--kilograms',
        type=positive_float,
        default=1,
        help='Unit scaling of kilograms within physics engine',
    )

    # Camera
    parser.add_argument(
        '--zoom',
        type=positive_float,
        default=1,
        help='Camera zoom',
    )
    parser.add_argument(
        '-f', '--free_camera',
        action='store_true',
        default=False,
        help='Allow for free camera (User controlled)',
    )
    parser.add_argument(
        '-r', '--rotating_camera',
        action='store_true',
        default=False,
        help='Enable rotating camera',
    )
    parser.add_argument(
        '-t', '--top_camera',
        action='store_true',
        default=False,
        help='Enable top view camera',
    )

    # Video recording
    parser.add_argument(
        '--video',
        type=str,
        default='',
        help='Record video path',
    )
    parser.add_argument(
        '--video_fps',
        type=int,
        default=30,
        help='Video recording frames per second',
    )
    parser.add_argument(
        '--video_speed',
        type=float,
        default=1.0,
        help='Video recording duration multiplier',
    )
    parser.add_argument(
        '--video_pitch',
        type=float,
        default=45,
        help='Camera pitch',
    )
    parser.add_argument(
        '--video_yaw',
        type=float,
        default=30,
        help='Camera yaw',
    )
    parser.add_argument(
        '--video_distance',
        type=float,
        default=1,
        help='Camera distance',
    )
    parser.add_argument(
        '--video_offset',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Camera offset',
    )
    parser.add_argument(
        '--video_resolution',
        nargs=2,
        type=int,
        metavar=('width', 'height'),
        default=(1280, 720),
        help='Camera resolution',
    )
    parser.add_argument(
        '--video_motion_filter',
        type=positive_float,
        default=None,
        help='Camera motion filter',
    )

    # Physics engine
    parser.add_argument(
        '--gravity',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, -9.81),
        help='Gravity',
    )
    parser.add_argument(
        '--num_sub_steps',
        type=positive_int,
        default=0,
        help='Number of physics sub-steps',
    )
    parser.add_argument(
        '--n_solver_iters',
        type=positive_int,
        default=100,
        help='Number of solver iterations for physics simulation',
    )
    parser.add_argument(
        '--residual_threshold',
        type=positive_float,
        default=1e-8,
        help='Solver residual threshold',
    )

    # MuJoCo
    parser.add_argument(
        '--cone',
        type=str,
        choices=('pyramidal', 'elliptic'),
        default='pyramidal',
        help='MuJoCo cone',
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=('Newton', 'PGS', 'CG'),
        default='Newton',
        help='MuJoCo solver',
    )
    parser.add_argument(
        '--integrator',
        type=str,
        choices=('Euler', 'RK4', 'implicit'),
        default='Euler',
        help='MuJoCo integrator',
    )
    parser.add_argument(
        '--impratio',
        type=positive_float,
        default=1,
        help='Ratio of frictional-to-normal constraint impedance',
    )
    parser.add_argument(
        '--ccd_iterations',
        type=positive_int,
        default=50,
        help='MuJoCo - Maximum number of iterations of the MPR algorithm',
    )
    parser.add_argument(
        '--ccd_tolerance',
        type=positive_float,
        default=1e-6,
        help='MuJoCo - Tolerance for early termination of the MPR algorithm',
    )
    parser.add_argument(
        '--noslip_iterations',
        type=int,
        default=0,
        help='MuJoCo - Maximum number of iterations of the noslip solver',
    )
    parser.add_argument(
        '--noslip_tolerance',
        type=positive_float,
        default=1e-6,
        help='MuJoCo - Tolerance for early termination of the noslip solver',
    )
    parser.add_argument(
        '--mujoco_extent',
        type=positive_float,
        default=100.0,
        help='MuJoCo - View extent',
    )

    # Pybullet
    parser.add_argument(
        '--lcp',
        type=str,
        choices=('si', 'dantzig', 'pgs'),
        default='si',
        help='Constraint solver LCP type',
    )
    parser.add_argument(
        '--opengl2',
        action='store_true',
        default=False,
        help='Run simulation with OpenGL 2 instead of 3',
    )
    parser.add_argument(
        '--cfm',
        type=positive_float,
        default=0,
        help='Pybullet CFM',
    )
    parser.add_argument(
        '--erp',
        type=positive_float,
        default=0,
        help='Pybullet ERP',
    )
    parser.add_argument(
        '--contact_erp',
        type=positive_float,
        default=0,
        help='Pybullet contact ERP',
    )
    parser.add_argument(
        '--friction_erp',
        type=positive_float,
        default=0,
        help='Pybullet friction ERP',
    )
    parser.add_argument(
        '--max_num_cmd_per_1ms',
        type=positive_int,
        default=int(1e9),
        help='Pybullet maximum number of commands per millisecond',
    )
    return parser


def config_parse_args():
    """Parse arguments"""
    parser = config_argument_parser()
    # return parser.parse_args()
    args, _ = parser.parse_known_args()
    return args
