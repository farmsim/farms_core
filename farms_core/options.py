""" Options """

from enum import IntEnum
from .io.yaml import pyobject2yaml, yaml2pyobject


class Options(dict):
    """Options"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getstate__(self):
        """Get state"""
        return self

    def __setstate__(self, value):
        """Get state"""
        for item in value:
            self[item] = value[item]

    def __getattr__(self, name):
        """ Get attribute by name using [] operator """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def to_dict(self) -> dict[str, dict | int | list]:
        """To dictionary"""
        return {
            key: (
                value.to_dict() if isinstance(value, Options)
                else int(value) if isinstance(value, IntEnum)
                else [
                    val.to_dict() if isinstance(val, Options) else val
                    for val in value
                ] if isinstance(value, list)
                else value
            )
            for key, value in self.items()
        }

    @classmethod
    def load(cls, filename: str, strict: bool = True):
        """Load from file"""
        kwargs = {'strict': False} if not strict else {}
        return cls(**yaml2pyobject(filename), **kwargs)

    def save(self, filename: str):
        """Save to file"""
        pyobject2yaml(filename, self.to_dict())
