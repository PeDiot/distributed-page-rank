from typing import Optional, Dict, Callable, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm 
import time 


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

def compute_mse(x: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
    """Returns the mean squared error between two arrays."""

    if len(x) != len(y):
        raise ValueError("Arrays must be of same length")
    
    if isinstance(x, list):
        x = np.array(x)
    
    if isinstance(y, list):
        y = np.array(y)

    mse = np.sum((x - y) ** 2) / len(x)
    return mse

def flatten_results(results: Dict, method: str) -> Dict:
    """Flattens the nested results of an experiment into a flat dictionnary.
    
    Args:
        results (Dict): The results of an experiment.
        method (str): The method used to generate the results.
        
    Returns:
        Dict: A flat dictionary containing the results of the experiment per method."""

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
       
    return row

def make_computation_time_table(results: List, method: str) -> pd.DataFrame:
    """Returns a pandas DataFrame containing the computation times of an experiment for a given method.
    
    Args:
        results (List): The results of an experiment.
        method (str): The method used to generate the results.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the results of the experiment per method."""
    
    df = pd.DataFrame(
        columns=["method", "n_nodes", "min_conn_per_node", "max_iter", "min_time", "max_time", "mean_time", "median_time", "std_time"]
    )

    for item in results:    
        item_flat = flatten_results(item, method)
        df = df.append(item_flat, ignore_index=True)
        
    return df

def make_mse_table(results_per_method: Dict) -> pd.DataFrame:
    """Returns a pandas DataFrame containing the MSE of an experiment for a given method.
    
    Args:
        results (List): The results of an experiment.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the results of the experiment per method."""
    
    df = pd.DataFrame(
        columns=["method", "n_nodes", "min_conn_per_node", "max_iter", "mse"]
    )

    numpy_pageranks = [
        item["pagerank_numpy"] for item in results_per_method["numpy"]
    ]

    for method, values in results_per_method.items():  

        if method == "numpy":
            pass 

        else: 
            for ix, item in enumerate(values) :
                mse = compute_mse(item[f"pagerank_{method}"], numpy_pageranks[ix])
                row = {
                    "method": method,
                    "n_nodes": item["n_nodes"],
                    "min_conn_per_node": item["min_conn_per_node"],
                    "max_iter": item["max_iter"],
                    "mse": mse
                }
                df = df.append(row, ignore_index=True)     

    return df