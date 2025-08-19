"""Documentation"""


from typing import Callable


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
