# Mars-Troughs

This repository contains routines for modeling trough migration patterns on the northern polar ice caps on Mars. This amounts to solving some differential equations regarding the trough trajectory, with inputs for these trajectories supplied by the output files from a heat transfer simulation of the Martian crust.

Downloading and installing this code is fairly straight forward:
```bash
git clone https://github.com/tmcclintock/Mars-Troughs.git
cd Mars-Troughs
python setup.py install
```
The requirements are light, and come with any package manager. They are:
* numpy
* scipy
* pytest

Once installed, run the unit tests from the home directory `Mars-Troughs/` with
```bash
pytest
```
If you encounter any errors, please feel free to [open an issue](https://github.com/tmcclintock/Mars-Troughs/issues).

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