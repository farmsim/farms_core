"""Automatic documentation"""

import os
import inspect
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate
from jinja2 import Environment, FileSystemLoader, select_autoescape

from farms_core.doc import ClassDoc, doc_class_markdown
from farms_core.sensors.sensor_convention import sc
from farms_core.experiment.options import ExperimentOptions
from farms_core.experiment.data import ExperimentData
from farms_core.sensors.data import (
    SensorsData,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    XfrcArray,
    MusclesArray,
    AdhesionsArray,
)

@dataclass
class ElementDoc:
    """Element to document"""
    name: str
    description: str


@dataclass
class ElementClassDoc(ElementDoc):
    """Element to document"""
    class_type: str


def sensors():
    """Main"""

    path_gen = os.path.expandvars("$FARMS_SRC/farms_core/ducks/gen/")
    path_md = os.path.expandvars("$FARMS_SRC/farms_core/farms_core/sensors/README.md")
    env = Environment(
        loader=FileSystemLoader(path_gen),
        autoescape=select_autoescape(),
    )
    template = env.get_template("sensors.md")

    # Title
    text = "# Sensors data structures\n\n"

    # SensorsData
    doc: ClassDoc = SensorsData.doc()
    children = [
        ElementClassDoc(
            name=child.name,
            class_type=(
                child.class_type
                if isinstance(child.class_type, str)
                else child.class_type.__name__
            ),
            description=child.description,
        )
        for child in doc.children
    ]
    text += template.render(doc=doc, children=children)

    # Sensor array classes
    for sensor_array_class, identifier in [
            [LinkSensorArray, "link"],
            [JointSensorArray, "joint"],
            [ContactsArray, "contact"],
            [XfrcArray, "xfrc"],
            [MusclesArray, "muscle"],
            [AdhesionsArray, "adhesion"],
    ]:
        text += "\n\n"
        doc = sensor_array_class.doc()
        sensor_children = [
             ElementClassDoc(
                name=child.name,
                class_type=(
                    child.class_type
                    if isinstance(child.class_type, str)
                    else child.class_type.__name__
                ),
                description=child.description,
            )
            for child in doc.children
        ]
        methods = [
            ElementDoc(
                name=method[1].__name__,
                description=method[1].__doc__.split("\n\n")[0],
            )
            for method in inspect.getmembers(
                sensor_array_class,
                predicate=inspect.isfunction,
            )
            if method[1].__doc__ is not None
        ]
        text += template.render(doc=doc, children=sensor_children, methods=methods)

        # Table
        text += "\n\n**Size and indices:**"
        text += (
            "\n\nNote: It is recommended to not use indices directly, but to"
            " favour accessing the data using the provided methods, or the"
            " sensor convention definitions provided in "
            " ´farms_core/sensors/sensor_convention´."
        )
        len_id = len(identifier)
        table = np.array([
            ["`"+key.replace(identifier+"_", "")+"`", value]
            for key, value in sc.__members__.items()
            if key[:len_id] == identifier
        ])
        text += "\n\n"
        text += tabulate(table, headers=["Key", "Value"], tablefmt="github")

    # Whitespace
    for _ in range(5):
        text = text.replace('\n\n\n', '\n\n')

    print(text)
    with open(path_md, "w+", encoding="utf-8") as output:
        output.write(text)


def doc_to_readme(class_to_document, output_path):
    """Main"""
    path_md = os.path.expandvars(output_path)
    text = doc_class_markdown(
        doc=class_to_document.doc(),
        level="#",
    )
    print(text)
    with open(path_md, "w+", encoding="utf-8") as output:
        output.write(text)


def main():
    """Main"""
    sensors()
    print("\n")
    doc_to_readme(
        class_to_document=ExperimentData,
        output_path="$FARMS_SRC/farms_core/farms_core/experiment/README.md",
    )
    print("\n")
    doc_to_readme(
        class_to_document=ExperimentOptions,
        output_path="$FARMS_SRC/farms_core/farms_core/README.md",
    )


if __name__ == "__main__":
    main()
