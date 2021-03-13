# Mars-Troughs

This repository contains routines for modeling trough migration patterns on the northern polar ice caps on Mars. This means solving some differential equations, with inputs for these trajectories supplied by the output files from a heat transfer simulation of the Martian crust.

## Installation

You should first install the development environment with `conda`:
```bash
conda env create -f environment.yml
```
If you do not plan on doing development you can just install the dependencies
in `requirements.txt` with `pip`:
```bash
pip install -r requirements.txt
```

To install after cloning this repository you can do:
```bash
python setup.py install
```
If you want to develop the code, there are two ways to install in an editable/develop mode so that your code changes are "seen" automatically by your environment. The first method is preferred if you are using an Anaconda environment:
```bash
conda develop .
```
Use this second method if you are using `pip` to manage your packages by using the "editable" model:
```bash
pip install -e .
```

Once installed, run the unit tests from the home directory `Mars-Troughs/` with
```bash
pytest
```
If you encounter any errors, please feel free to [open an issue](https://github.com/tmcclintock/Mars-Troughs/issues).

## Developing

Before committing any code, you are encouraged to set up [`pre-commit`](https://pre-commit.com/), which will clean up code for you before committing it to
the repository:
```bash
pre-commit install
```
If you see any "failures" while running `git commit`, it means that one of
the `pre-commit` packages either automatically made a change to your code,
or is suggesting a code. Just add/commit again and you are all set!

## Usage

Using the code is simple, however the interface is a bit annoying to work with, since it has been kept as flexible as possible to make switching between models seemless (so, easy for us, difficult for you :)).

A minimal example that would produce a trough trajectory is the following:
```python
import mars_troughs

accumulation_params = [1e-6] #example value; units of mm/yr/Insolation
lag_params = [1] #example value; units of mm
accumulation_model_number = 0 #only a single parameter; a constant model
lag_model_number = 0 #only a single parameter; a constant lag model

errorbar = 0 #necessary as input, but does not affect the migration path

trough = mars_troughs.Trough(accumulation_params, lag_params
                             accmulation_model_number, lag_model_number
                             errorbar)


times = trough.ins_times #times over which the trajectory is computed
#Note: you can query for any time between 0 - 5 Myr ago
x = trough.get_xt(times)
y = trough.get_yt(times)
```

For a better example with visualizations, see `example.py`.