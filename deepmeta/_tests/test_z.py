import numpy as np
import deepmeta.deepmeta_functions as df
import deepmeta._dock_widget as dw


def test_add_z():
    arr = np.array([[0, 1], [0, 1]])
    assert (df.add_z(arr, 3) == np.array([[3, 0, 1], [3, 0, 1]])).all()


def test_fix_v():
    v = [1, 2, 3]
    contours = [1, 2, 3, 4, 5]
    assert dw.fix_v(v, contours) == [1, 2, 3, 0, 0]
