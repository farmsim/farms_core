"""Documentation"""

import inspect
from . import pylog


class ClassDoc:
    """Class documentation"""

    def __init__(self, name, description, class_type, children, class_link=None):
        super().__init__()
        self.name: str = name
        self.class_type = class_type
        self.class_link = class_type if class_link is None else class_link
        self.description: str = description
        self.children: list[ClassDoc] = children


class ChildDoc:
    """Child documentation"""

    def __init__(self, name, description, class_type, class_link=None):
        super().__init__()
        self.name: str = name
        self.class_type = class_type
        self.class_link = class_type if class_link is None else class_link
        self.description: str = description


def get_inherited_doc_children(cls) -> list[ChildDoc]:
    """Get inherited doc children"""
    return [
        child
        for base in cls.__bases__
        if hasattr(base, 'doc')
        for child in base.doc().children
    ]


def doc_class_markdown(doc, level, classes_defined=None, clean_whitespace=True):
    """Config documentation"""

    # Avoid class documentation repetitions
    if classes_defined is None:
        classes_defined = []

    # Class to document
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
                text += doc_class_markdown(
                    doc=child.class_link.doc(),
                    level=f"#{level}",
                    classes_defined=classes_defined,
                )
                classes_defined.append(child.class_link.__name__)
        else:
            module = inspect.getmodule(child.class_link)
            if module is not None and "farms_" in module.__name__:
                pylog.warning("WARNING: %s does not have doc", child.class_link)

    text += "\n"

    # Whitespace
    if clean_whitespace:
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

    return text
