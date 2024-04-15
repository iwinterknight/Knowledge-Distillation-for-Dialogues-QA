import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')


def get_root_dir():
    return ROOT_DIR


def get_config_path():
    return CONFIG_PATH
