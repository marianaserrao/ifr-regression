from box import Box
import yaml

def get_config(file_path):
    """Load a YAML file and return a Box object (dot notation)"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return Box(data)