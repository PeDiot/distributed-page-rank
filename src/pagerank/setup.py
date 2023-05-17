"""Setup file to compile cpu_parallel.pyx with Cython.

Run `python setup.py build_ext --inplace` to compile the Cython code."""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="pagerank_cython", 
        sources=["pagerank_cython.pyx"],
        include_dirs=[numpy.get_include()], 
        extra_compile_args=["-fopenmp"],        # /openmp for Windows     
        extra_link_args=["-fopenmp"]
    )
]

setup(
    ext_modules=cythonize(extensions)
)


