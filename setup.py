from setuptools import setup, find_packages
from distutils.extension import Extension
from pip.req import parse_requirements

import Cython.Compiler.Options
import cython_gsl
import numpy
from Cython.Build import cythonize

# Generate compiler annotations in HTML
Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension(
        "ciles.ciles",
        ["ciles/ciles.pyx"],
        include_dirs=[
            numpy.get_include(), cython_gsl.get_cython_include_dir()],
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir()],
    )
]

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    include_dirs=[cython_gsl.get_include()],
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
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
    version='0.1.0',
    install_requires=reqs,
    setup_requires=reqs
)
