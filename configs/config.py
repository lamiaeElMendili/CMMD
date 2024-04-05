import yaml


def get_config(name):
    stream = open(name, 'r')
    config_dict = yaml.safe_load(stream)
    return Config(config_dict)


class Config:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Config(val) if isinstance(val, dict) else val)
    def to_dict(self):
        """
        Convert the Config object to a dictionary.
        """
        config_dict = {}
        for key, val in self.__dict__.items():
            if isinstance(val, Config):
                config_dict[key] = val.to_dict()
            else:
                config_dict[key] = val
        return config_dict