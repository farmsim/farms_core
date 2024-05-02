"""Sensors documentation"""

import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

from farms_core.sensors.data import (
    SensorsData,
    ClassDoc,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    XfrcArray,
    MusclesArray,
)


def main():
    """Main"""
    path_gen = os.path.expandvars('$FARMS_SRC/farms_core/ducks/gen/')
    path_md = os.path.expandvars('$FARMS_SRC/farms_core/farms_core/sensors/README.md')
    env = Environment(
        loader=FileSystemLoader(path_gen),
        autoescape=select_autoescape(),
    )
    template = env.get_template('sensors.md')
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
    for sensor_array_class in [
            LinkSensorArray,
            JointSensorArray,
            ContactsArray,
            XfrcArray,
            MusclesArray,
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
    print(text)
    with open(path_md, 'w+', encoding='utf-8') as output:
        output.write(text)


if __name__ == '__main__':
    main()
