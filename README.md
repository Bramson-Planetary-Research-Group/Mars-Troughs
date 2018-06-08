# Mars-Troughs
Code for modeling the troughs on the north polar ice cap on Mars.

To use do the following:
`cd src`
`make`
`cd ..`
`python trough_test.py`

If you change the lag model, you have to change the initial parameters and priors in `trough_test.py` as well as the model itself in `src/equations_of_motion.c`. Just look for the comments and it shows where to make changes.