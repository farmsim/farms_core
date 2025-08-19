"""Experiment options"""

import os

from ..options import Options
from ..model.options import AnimatOptions, ArenaOptions
from ..simulation.options import SimulationOptions
from ..doc import ClassDoc, ChildDoc


def resolve_path(path, config):
    """Resolve path given directory"""
    directory = os.path.dirname(config)
    relative_path = os.path.join(directory, path)
    # Absolute path
    if os.path.isfile(path):
        return path
    # Relative path
    if os.path.isfile(relative_path):
        return relative_path
    raise FileNotFoundError(f'Could not not resolve "{path}" from "{config}"')


class ExperimentOptions(Options):
    """Experiment options

    The experiment options contains the top-level information for running an
    experiment. It includes information about the animats being run, the
    environment and its arenas, as well as the simulation options. This is done
    by pointing to the configs in the right file format:

    - simulation: A path to a SimulationOptions configuration file
    - animats: A list of paths to AnimatOptions configuration files
    - arenas: A list of paths to ArenaOptions configuration files

    """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="animat",
            description="Describes the animat properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="simulation",
                    class_type=SimulationOptions,
                    description="The simulation options.",
                ),
                ChildDoc(
                    name="animats",
                    class_type="list[AnimatOptions]",
                    class_link=AnimatOptions,
                    description="List of animats options.",
                ),
                ChildDoc(
                    name="arenas",
                    class_type="list[ArenaOptions]",
                    class_link=ArenaOptions,
                    description="List of animats options.",
                ),
            ],
        )

    def __init__(
            self,
            simulation: SimulationOptions,
            animats: list[AnimatOptions],
            arenas: list[ArenaOptions],
            **kwargs,
    ):
        super().__init__()
        self.simulation: SimulationOptions = simulation
        self.animats = animats
        self.arenas = arenas
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def load(
            cls,
            filename: str,
            strict: bool = True,
            animat_class=AnimatOptions,
            arena_class=ArenaOptions,
    ):
        """Load from file"""
        options = super().load(filename)
        if isinstance(options.simulation, str):
            path = resolve_path(options.simulation, filename)
            options.simulation = SimulationOptions.load(
                filename=path,
                strict=strict,
            )
        for animat_i, animat in enumerate(options.animats):
            if isinstance(animat, str):
                path = resolve_path(animat, filename)
                options.animats[animat_i] = animat_class.load(
                    filename=path,
                    strict=strict,
                )
        for arena_i, arena in enumerate(options.arenas):
            if isinstance(arena, str):
                path = resolve_path(arena, filename)
                options.arenas[arena_i] = arena_class.load(
                    filename=path,
                    strict=strict,
                )
        return options

    # @classmethod
    # def from_configs(
    #         cls,
    #         simulation_path: str,
    #         animat_paths: list[str],
    #         arena_paths: list[str],
    #         animat_class=AnimatOptions,
    #         arena_class=ArenaOptions,
    # ):
    #     """Experiment options from paths"""
    #     return cls(
    #         simulation=SimulationOptions.load(simulation_path),
    #         animats=[
    #             animat_class.load(animat_path)
    #             for animat_path in animat_paths
    #         ],
    #         arenas=[
    #             arena_class.load(arena_path)
    #             for arena_path in arena_paths
    #         ],
    #     )
