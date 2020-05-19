"""Options"""

import yaml


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

    def to_dict(self):
        """To dictionary"""
        return {
            key: (
                value.to_dict() if isinstance(value, Options)
                else [
                    val.to_dict() if isinstance(val, Options) else val
                    for val in value
                ] if isinstance(value, list)
                else value
            )
            for key, value in self.items()
        }

    @classmethod
    def load(cls, filename):
        """Load from file"""
        with open(filename, 'r') as yaml_file:
            options = yaml.full_load(yaml_file)
        return cls(**options)

    def save(self, filename):
        """Save to file"""
        with open(filename, 'w+') as yaml_file:
            yaml.dump(
                self.to_dict(),
                yaml_file,
                default_flow_style=False,
                sort_keys=False,
            )
