from typing import Dict

import os 
import json
import yaml
from yaml import Loader


def to_json(results: Dict, dir_path: str, file_name: str) -> None: 

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, file_name)

    with open(file_path, "w") as f:
        json.dump(results, f)

    print(f"Saved results to {file_path}.")

def from_json(dir_path: str, file_name: str) -> Dict: 
    file_path = os.path.join(dir_path, file_name)

    with open(file_path, "r") as f:
        results = json.load(f)

    return results

def load_config(cfg_path: str) -> Dict:
    """Load yaml configuration file."""
    
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader)

    return cfg
