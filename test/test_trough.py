import numpy as np
import numpy.testing as npt
import mars_troughs as mt

def test_trough():
    #Test that it builds
    test_acc_params = [1e-6, 1e-11]
    acc_model_number = 1
    test_lag_params = [1, 1e-7]
    lag_model_number = 1
    errorbar = 100.
    tr = mt.Trough(test_acc_params, test_lag_params,
                   acc_model_number, lag_model_number,
                   errorbar)
    return
