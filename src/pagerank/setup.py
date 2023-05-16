"""Setup file to compile cpu_parallel.pyx with Cython.

python setup.py build_ext --inplace"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("pagerank_cython", ["pagerank_cython.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(extensions)
)


