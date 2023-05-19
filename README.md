# Distributed PageRank algorithm 

## Gettind started

1. Clone the repository 

```git
git clone 
```

## Description

The main part of the project lies in the [`pagerank`](src/pagerank/) module which implements differents PageRank computation methods. 

```
src/
├── backup.py
├── experiment.py
├── graph.py
├── plots.py
├── utils.py
└── pagerank/ 
    ├── pagerank_base.py
    ├── pagerank_numpy.py
    ├── pagerank_cython.py
    └── pagerank_gpu.py
```


| File       | Description                                             |
|-------------------|---------------------------------------------------------|
| `backup.py`         | Save files to `json`, load from `json` and `yaml`                      |
| `experiment.py`     | Prepare and run experiments to measure computation time and MSE               |
| `graph.py`          | Make graph as adjacency matrix            |
| `plots.py`          | Plot computation time & MSE                     |
| `utils.py`         | Helper functions             |
| `pagerank_base.py`  | Basic PageRank algorithm             |
| `pagerank_numpy.py` | Vectorized PageRank algorithm using `NumPy`        |
| `pagerank_cython.py`| Parallelized (CPU) PageRank algorithm using `Cython` and `OpenMP`       |
| `pagerank_gpu.py`   | Parallelized (GPU) PageRank algorithm using `pycuda`          |

See [`notebook_gpu`](https://drive.google.com/file/d/1rNhbZQfeArP8kCoHw2yuxpTgE_mHOudy/view?usp=sharing) for the PageRank GPU implementation.