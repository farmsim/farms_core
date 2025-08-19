"""Data documentation"""

import os
import inspect
from dataclasses import dataclass

import numpy as np
from tabulate import tabulate
from jinja2 import Environment, FileSystemLoader, select_autoescape

from farms_core import pylog
from farms_core.sensors.sensor_convention import sc
from farms_core.experiment.options import ExperimentOptions
from farms_core.experiment.data import ExperimentData
from farms_core.sensors.data import (
    SensorsData,
    ClassDoc,
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


def config_doc(doc, level, classes_defined):
    """Config documentation"""
    class_name = (
        doc.class_type
        if isinstance(doc.class_type, str)
        else doc.class_type.__name__
    )
    class_ref = f"<a id='ref-{class_name}'></a>"
    text = f"{class_ref}\n{level} {class_name}\n\n{doc.description}\n"

    # Children short descriptions
    text += "\n"
    for child in doc.children:
        module = inspect.getmodule(child.class_link)
        class_link_name = (
            child.class_link.__name__
            if not isinstance(child.class_link, str)
            else child.class_link
        )
        reference = (
            f"`{class_link_name}`"
            if module is None
            or "farms_" not in module.__name__
            else child.class_type.replace(
                class_link_name,
                f"[{class_link_name}](#ref-{class_link_name})",
            )
            if isinstance(child.class_type, str)
            else f"[{class_link_name}](#ref-{class_link_name})"
        )
        text += f"- `{child.name}` ({reference}):"
        text += f" {child.description}\n"
    text += "\n"

    # Children recursive descriptions
    for child in doc.children:
        if hasattr(child.class_link, "doc"):
            if child.class_link.__name__ not in classes_defined:
                text += config_doc(
                    doc=child.class_link.doc(),
                    level=f"#{level}",
                    classes_defined=classes_defined,
                )
                classes_defined.append(child.class_link.__name__)
        else:
            module = inspect.getmodule(child.class_link)
            if module is not None and "farms_" in module.__name__:
                pylog.warning("WARNING: %s does not have doc", child.class_link)

    # Whitespace
    for _ in range(5):
        text = text.replace("\n\n\n", "\n\n")

    text += "\n"
    return text


def model():
    """Main"""

    path_md = os.path.expandvars("$FARMS_SRC/farms_core/farms_core/README.md")
    classes_defined = []
    text = config_doc(
        doc=ExperimentOptions.doc(),
        level="#",
        classes_defined=classes_defined,
    )

    # Whitespace
    for _ in range(5):
        text = text.replace("\n\n\n", "\n\n")

    print(text)
    with open(path_md, "w+", encoding="utf-8") as output:
        output.write(text)


def main():
    """Main"""
    sensors()
    print("\n")
    model()


if __name__ == "__main__":
    main()
