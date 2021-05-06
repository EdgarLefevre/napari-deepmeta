import pytest
import deepmeta.deepmeta_functions as df
import numpy as np

def test_add_z():
    arr = np.array([[0, 1], [0, 1]])
    assert (df.add_z(arr, 3) == np.array([[3, 0, 1], [3, 0, 1]])).all()

