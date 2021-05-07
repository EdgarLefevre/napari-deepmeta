from deepmeta import napari_experimental_provide_dock_widget
import pytest


# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-deepmeta"
# the name of your widget(s)
MY_WIDGET_NAMES = ["Segment Lungs", "Segment Metas"]


@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(widget_name, make_napari_viewer):

    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1


def test_add_z():
    import deepmeta.deepmeta_functions as df
    import numpy as np
    arr = np.array([[0, 1], [0, 1]])
    assert (df.add_z(arr, 3) == np.array([[3, 0, 1], [3, 0, 1]])).all()

#
def test_fix_v():
    import deepmeta._dock_widget as dw
    v = [1, 2, 3]
    contours = [1, 2, 3, 4, 5]
    assert dw.fix_v(v, contours) == [1, 2, 3, 0, 0]

