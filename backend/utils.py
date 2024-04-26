import os
import yaml


def set_config(yaml_file_path):
    if os.path.isfile(yaml_file_path):
        with open(yaml_file_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

    return data_loaded
