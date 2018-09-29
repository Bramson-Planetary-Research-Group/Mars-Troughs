from __future__ import print_function
import sys, os, glob
import setuptools
from setuptools import setup, Extension
import subprocess

#Create the symlink.
try:
    os.system('ln -s ../include mars_troughs/include')
except:
    OSError

#Include GSL flags
try:
    cflags = subprocess.check_output(['gsl-config', '--cflags']).split()
    lflags = subprocess.check_output(['gsl-config', '--libs']).split()
    cflags = [cflag.decode("utf-8") for cflag in cflags]
    lflags = [lflag.decode("utf-8") for lflag in lflags]
except OSError:
    raise Exception("must have GSL installed and gsl-config working.")

#Specify the sources
sources = glob.glob(os.path.join('src','*.c'))
#and the header files.
headers = glob.glob(os.path.join('include','*.h'))

#Create the extension.
ext=Extension("mars_troughs._mars_troughs",
              sources,
              depends=headers,
              include_dirs=['include'],
              extra_compile_args=[os.path.expandvars(flag) for flag in cflags],
              extra_link_args=[os.path.expandvars(flag) for flag in lflags])

#Create the distribution
dist = setup(name="mars_troughs",
             author="Tom McClintock",
             author_email="tmcclintock89@gmail.com",
             description="Modules for modeling ice troughs on the Mars North polar ice cap.",
             license="MIT License",
             url="https://github.com/tmcclintock/Mars-Troughs",
             packages=['mars_troughs'],
             package_data={'mars_troughs' : headers },
             ext_modules=[ext],
             install_requires=['cffi','numpy']),
#             setup_requires=['pytest_runner'],
#             tests_require=['pytest'])

#setup.py doesn't put the .so file in the mars_troughss directory, 
#so this bit makes it possible to
#import mars_troughs from the root directory.  
#Not really advisable, but everyone does it at some
#point, so might as well facilitate it.
build_lib = glob.glob(os.path.join('build','*','mars_troughs','_mars_troughs*.so'))
if len(build_lib) >= 1:
    lib = os.path.join('mars_troughs','_mars_troughs.so')
    if os.path.lexists(lib): os.unlink(lib)
    os.link(build_lib[0], lib)
