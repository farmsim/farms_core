"""Extensions"""

import importlib

from ..options import Options
from ..doc import ClassDoc, ChildDoc


def import_module_item(module_path: str, item_name: str):
    """from module_path import item_name"""
    module = importlib.import_module(module_path)
    return getattr(module, item_name)


def import_item(item_path: str):
    """item_name = module_path.item_name; from module_path import item_name"""
    module_path, _, item_name = item_path.rpartition('.')
    return import_module_item(module_path, item_name)


class ExtensionOptions(Options):
    """Extension options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="extension config",
            description="Describes the extension loader and options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="loader",
                    class_type=str,
                    description="Extension loader.",
                ),
                ChildDoc(
                    name="config",
                    class_type=dict,
                    description="Extension configuration",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.loader: str = kwargs.pop('loader')
        self.config: list[str] = kwargs.pop('config')
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
