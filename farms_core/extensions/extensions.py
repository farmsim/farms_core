"""Extensions"""

import importlib


def import_module_item(module_path: str, item_name: str):
    """from module_path import item_name"""
    module = importlib.import_module(module_path)
    return getattr(module, item_name)


def import_item(item_path: str):
    """item_name = module_path.item_name; from module_path import item_name"""
    module_path, _, item_name = item_path.rpartition('.')
    return import_module_item(module_path, item_name)
