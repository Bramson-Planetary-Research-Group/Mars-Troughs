import os

PATH_ROOT = os.path.dirname(__file__)
from setuptools import setup

import mars_troughs  # noqa: E402

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Create the distribution
dist = setup(
    name="mars_troughs",
    author="Tom McClintock",
    author_email="tmcclintock89@gmail.com",
    version=mars_troughs.__version__,
    description=mars_troughs.__docs__,
    long_description=long_description,
    license="MIT License",
    url="https://github.com/tmcclintock/Mars-Troughs",
    include_package_data=True,
    packages=["mars_troughs"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
