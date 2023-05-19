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