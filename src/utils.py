from typing import Optional, Dict, Callable, List

import numpy as np
import pandas as pd
from tqdm import tqdm 
import time 
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

def measure_time(func: Callable, n_repeat: int=50, return_results: bool=False, desc: Optional[str]=None, *args, **kwargs):
    """Returns the time taken to run a function n_repeat times.
    
    Args:
        func (Callable): The function to run.
        n_repeat (int): The number of times to run the function.
        return_results (bool): Whether to return the results of the function. Defaults to False.
        desc (str): A description to display in the progress bar.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Dict: A dictionary containing the minimum, maximum, mean, median, and standard deviation of the time taken to run the function.
        Any results returned by the function."""    

    times = []
    loop = tqdm(range(n_repeat))
    loop.set_description(desc)

    for _ in loop:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    
    time_result = {
        "min": min(times),
        "max": max(times),
        "mean": np.mean(times),
        "median": np.median(times), 
        "std": np.std(times)
    }

    if return_results:
        return time_result, result
    
    return time_result

def compute_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Returns the mean squared error between two arrays."""

    if len(x) != len(y):
        raise ValueError("Arrays must be of same length")

    mse = np.sum((x - y) ** 2) / len(x)
    return mse

def flatten_results(results: Dict) -> List:
    """Flattens the results of an experiment into a list of dictionaries.
    
    Args:
        results (Dict): The results of an experiment.
        
    Returns:
        List: A list of dictionaries containing the results of the experiment per method."""

    results_flat = []

    for method in ["numpy", "cython"]:
        row = {
            "method": method,
            "n_nodes": results["n_nodes"],
            "min_conn_per_node": results["min_conn_per_node"],
            "max_iter": results["max_iter"],
            "min_time": results[f"time_{method}"]["min"],
            "max_time": results[f"time_{method}"]["max"],
            "mean_time": results[f"time_{method}"]["mean"],
            "median_time": results[f"time_{method}"]["median"],
            "std_time": results[f"time_{method}"]["std"]
        }
        results_flat.append(row)  

    return results_flat

def dict_to_dataframe(results: Dict) -> pd.DataFrame:
    """Converts the results of an experiment into a pandas DataFrame.
    
    Args:
        results (Dict): The results of an experiment.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the results of the experiment per method."""
    
    df = pd.DataFrame(
        columns=["method", "n_nodes", "min_conn_per_node", "max_iter", "min_time", "max_time", "mean_time", "median_time", "std_time"]
    )

    for item in results:    
        item_flat = flatten_results(item)
        df = df.append(item_flat, ignore_index=True)
        
    return df