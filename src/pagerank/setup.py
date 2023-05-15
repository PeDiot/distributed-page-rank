"""Setup file to compile cpu_parallel.pyx with Cython.

python setup.py build_ext --inplace"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cpu_parallel.pyx")
)

