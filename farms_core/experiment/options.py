"""Experiment options"""

import os

from ..options import Options
from ..model.options import AnimatOptions, ArenaOptions
from ..simulation.options import SimulationOptions
from ..extensions.extensions import import_item
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


class ExperimentLoadOptions(Options):
    """Experiment load options

    Provides the path to the classes to use for loading the options.

    """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="experiment load options",
            description="Describes the loaders.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="simulation_options",
                    class_type=str,
                    description=(
                        "SimulationOptions loading class. Default if empty."
                    ),
                ),
                ChildDoc(
                    name="animats_options",
                    class_type="list[str]",
                    description=(
                        "AnimatOptions loading classes. Default if empty."
                    ),
                ),
                ChildDoc(
                    name="arenas_options",
                    class_type="list[str]",
                    description=(
                        "ArenaOptions loading classes. Default if empty."
                    ),
                ),
                ChildDoc(
                    name="experiment_data",
                    class_type=str,
                    description=(
                        "ExperimentData loading class. Default if empty."
                    ),
                ),
                ChildDoc(
                    name="animats_data",
                    class_type="list[str]",
                    description=(
                        "AnimatData loading classes. Default if empty."
                    ),
                ),
            ],
        )

    def __init__(
            self,
            simulation_options: str,
            animats_options: list[str],
            arenas_options: list[str],
            experiment_data: str,
            animats_data: list[str],
            **kwargs,
    ):
        super().__init__()
        self.simulation_options = simulation_options
        self.animats_options = animats_options
        self.arenas_options = arenas_options
        self.experiment_data = experiment_data
        self.animats_data = animats_data
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs


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
                ChildDoc(
                    name="loaders",
                    class_type=ExperimentLoadOptions,
                    description="Loaders to use for options and data.",
                ),
            ],
        )

    def __init__(
            self,
            simulation: SimulationOptions,
            animats: list[AnimatOptions],
            arenas: list[ArenaOptions],
            loaders: ExperimentLoadOptions,
            **kwargs,
    ):
        super().__init__()
        self.simulation: SimulationOptions = simulation
        self.animats = animats
        self.arenas = arenas
        self.loaders = loaders
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def load(
            cls,
            filename: str,
            strict: bool = True,
    ):
        """Load from file"""
        options = super().load(filename)
        options.loaders = ExperimentLoadOptions(**options["loaders"])
        if isinstance(options.simulation, str):
            path = resolve_path(options.simulation, filename)
            simulation_class = import_item(options.loaders.simulation_options)
            options.simulation = simulation_class.load(
                filename=path,
                strict=strict,
            )
        assert len(options.animats) == len(options.loaders.animats_options), (
            f"In the experiment config ({filename}), there should be as many"
            f" animats ({len(options.animats)}) as there are animats_options"
            f" loaders ({len(options.loaders.animats_options)})."
        )
        for animat_i, (animat, animat_class_path) in enumerate(zip(
                options.animats,
                options.loaders.animats_options,
        )):
            if isinstance(animat, str):
                path = resolve_path(animat, filename)
                animat_class = import_item(animat_class_path)
                options.animats[animat_i] = animat_class.load(
                    filename=path,
                    strict=strict,
                )
        assert len(options.arenas) == len(options.loaders.arenas_options), (
            f"In the experiment config ({filename}), there should be as many"
            f" arenas ({len(options.arenas)}) as there are arenas_options"
            f" loaders ({len(options.loaders.arenas_options)})."
        )
        for arena_i, (arena, arena_class_path) in enumerate(zip(
                options.arenas,
                options.loaders.arenas_options,
        )):
            if isinstance(arena, str):
                path = resolve_path(arena, filename)
                arena_class = import_item(arena_class_path)
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
