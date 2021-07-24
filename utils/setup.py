from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("fast_fstm", ["utils/fast_fstm.pyx"],
              extra_compile_args = ["-ffast-math", '-mavx', '-O3'],),
]

setup(
    ext_modules=cythonize(extensions, annotate=True)
)
