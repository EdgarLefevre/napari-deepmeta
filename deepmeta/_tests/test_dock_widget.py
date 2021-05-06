from deepmeta import napari_experimental_provide_dock_widget
import deepmeta._dock_widget as dw
import pytest



def test_fix_v():
    v = [1, 2, 3]
    contours = [1, 2, 3, 4, 5]
    assert dw.fix_v(v, contours) == [1, 2, 3, 0, 0]

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

