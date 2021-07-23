from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("fast_fstm", ["fast_fstm.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-ffast-math", '-mavx']),
]

setup(
    ext_modules=cythonize(extensions, annotate=True)
)
