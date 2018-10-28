from __future__ import print_function
import sys, os, glob
import setuptools
from setuptools import setup, Extension
import subprocess

#Create the distribution
dist = setup(name="mars_troughs",
             author="Tom McClintock",
             author_email="tmcclintock89@gmail.com",
             description="Modules for modeling ice troughs on the Mars North polar ice cap.",
             license="MIT License",
             url="https://github.com/tmcclintock/Mars-Troughs",
             include_package_data = True,
             packages=['mars_troughs'],
             install_requires=['numpy'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest'])
