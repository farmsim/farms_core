"""Data documentation"""

import os
import inspect
import numpy as np
from tabulate import tabulate
from jinja2 import Environment, FileSystemLoader, select_autoescape

from farms_core import pylog
from farms_core.sensors.sensor_convention import sc
from farms_core.experiment.options import ExperimentOptions
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


def sensors():
    """Main"""

    path_gen = os.path.expandvars('$FARMS_SRC/farms_core/ducks/gen/')
    path_md = os.path.expandvars('$FARMS_SRC/farms_core/farms_core/sensors/README.md')
    env = Environment(
        loader=FileSystemLoader(path_gen),
        autoescape=select_autoescape(),
    )
    template = env.get_template('sensors.md')

    # Title
    text = '# Simulation data structures\n\n'

    # SensorsData
    doc = SensorsData.doc()
    children = [
        child.doc()
        if not isinstance(child, ClassDoc)
        else child
        for child in doc.children
    ]
    text += template.render(doc=doc, children=children)

    # Sensor array classes
    for sensor_array_class, identifier in [
            [LinkSensorArray, 'link'],
            [JointSensorArray, 'joint'],
            [ContactsArray, 'contact'],
            [XfrcArray, 'xfrc'],
            [MusclesArray, 'muscle'],
            [AdhesionsArray, 'adhesion'],
    ]:
        text += '\n\n'
        doc = sensor_array_class.doc()
        children = [
            child.doc()
            if not isinstance(child, ClassDoc)
            else child
            for child in doc.children
        ]
        text += template.render(doc=doc, children=children)

        # Table
        text += f'\n\n{identifier.capitalize()} array size and indices:'
        len_id = len(identifier)
        table = np.array([
            ['`'+key.replace(identifier+"_", "")+'`', value]
            for key, value in sc.__members__.items()
            if key[:len_id] == identifier
        ])
        text += '\n\n'
        text += tabulate(table, headers=['Key', 'Value'], tablefmt="github")

    print(text)
    with open(path_md, 'w+', encoding='utf-8') as output:
        output.write(text)


def config_doc(doc, level, classes_defined):
    """Main"""
    class_name = (
        doc.class_type
        if isinstance(doc.class_type, str)
        else doc.class_type.__name__
    )
    class_ref = f'<a id="ref-{class_name}"></a>'
    text = f'{class_ref}\n{level} {class_name}\n\n{doc.description}\n'

    # Children short descriptions
    text += '\n'
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
            or 'farms_' not in module.__name__
            else child.class_type.replace(
                class_link_name,
                f"[{class_link_name}](#ref-{class_link_name})",
            )
            if isinstance(child.class_type, str)
            else f"[{class_link_name}](#ref-{class_link_name})"
        )
        text += f'- `{child.name}` ({reference}):'
        text += f' {child.description}\n'
    text += '\n'

    # Chbildren recursive descriptions
    for child in doc.children:
        if hasattr(child.class_link, 'doc'):
            if child.class_link.__name__ not in classes_defined:
                text += config_doc(
                    doc=child.class_link.doc(),
                    level=f'#{level}',
                    classes_defined=classes_defined,
                )
                classes_defined.append(child.class_link.__name__)
        else:
            module = inspect.getmodule(child.class_link)
            if module is not None and 'farms_' in module.__name__:
                pylog.warning(f'WARNING: {child.class_link} does not have doc')

    # Whitespace
    for _ in range(5):
        text = text.replace('\n\n\n', '\n\n')

    text += '\n'
    return text


def model():
    """Main"""

    path_md = os.path.expandvars('$FARMS_SRC/farms_core/farms_core/README.md')
    classes_defined = []
    text = config_doc(
        doc=ExperimentOptions.doc(),
        level='#',
        classes_defined=classes_defined,
    )

    # Whitespace
    for _ in range(5):
        text = text.replace('\n\n\n', '\n\n')

    print(text)
    with open(path_md, 'w+', encoding='utf-8') as output:
        output.write(text)


def main():
    """Main"""
    sensors()
    print('\n')
    model()


if __name__ == '__main__':
    main()
