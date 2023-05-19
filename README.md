# Distributed PageRank algorithm 

## Gettind started

1. Clone the repository 

```
git clone https://github.com/PeDiot/distributed-page-rank.git
```

2. Install [`requirements`](requirements.txt)

You can create a virtual environment before launching the command.

```
pip install -r requirements
```

3. Run [`setup`](setup.py) file 

It is used to compile [`pagerank_cython`](src/pagerank/pagerank_cython.pyx) into an extension module.

```
python setup.py build_ext --inplace
```

4. Download the results from the experiment [here](https://drive.google.com/drive/folders/1UABC7J5kSZ_muW9rC5dkxyvPtCX_w-0i?usp=sharing). 

5. Run [`main`](main.py) file

The `eval` mode can be used to evaluate the downloaded results. 

```
Run an experiment or evaluate the results of an experiment.

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  The type of the experiment. Either 'exp' or 'eval'.
  -hp HYPERPARAM, --hyperparam HYPERPARAM
                        The hyperparameter to evaluate. Either 'n_nodes', 'min_conn_per_node' or 'max_iter'.   
  -pm PAGERANK_METHODS, --pagerank_methods PAGERANK_METHODS
                        The pagerank methods to evaluate.
  -ly LOG_Y, --log_y LOG_Y
                        Whether to use a logarithmic y-axis or not.
```

## Description

### Architecture

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

### PageRank methods

The main part of the project lies in the [`pagerank`](src/pagerank/) module which implements differents PageRank computation methods.

| File       | Description                                             |
|-------------------|---------------------------------------------------------|
| `pagerank_base.py`  | Basic PageRank algorithm             |
| `pagerank_numpy.py` | Vectorized PageRank algorithm using `NumPy`        |
| `pagerank_cython.py`| Parallelized (CPU) PageRank algorithm using `Cython` and `OpenMP`       |
| `pagerank_gpu.py`   | Parallelized (GPU) PageRank algorithm using `pycuda`          |

See [`notebook_gpu`](https://drive.google.com/file/d/1rNhbZQfeArP8kCoHw2yuxpTgE_mHOudy/view?usp=sharing) for the PageRank GPU implementation.