from setuptools import setup
from distutils.extension import Extension

import Cython.Compiler.Options
import cython_gsl
import numpy
from Cython.Distutils import build_ext

# Generate compiler annotations in HTML
Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension(
        "ciles",
        ["ciles/ciles.pyx"],
        include_dirs=[
            numpy.get_include(), cython_gsl.get_cython_include_dir()],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
    )
]

setup(
    cmdclass={'build_ext': build_ext},
    include_dirs=[cython_gsl.get_include()],
    ext_modules=ext_modules,
    test_suite='tests',
    name='ciles',
    description='Langevin integrator for SDEs with constant drift and diffusion on continuous intervals with circular boundary conditions.',
    author='Alex Seeholzer',
    author_email='seeholzer@gmail.com',
    url='https://github.com/flinz/ciles',
    keywords=['science', 'langevin', 'math'],  # arbitrary keywords
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'],
    license='GPLv2',
    version='0.1.0'
)
