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


def test_get_volumes():
    import deepmeta.deepmeta_functions as df
    import numpy as np
    arr = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    res = df.get_volumes(arr, 0.0047)
    assert res == [[0.0047], [0.0047], [0.0047]]


def test_dilate_and_erode():
    import deepmeta.deepmeta_functions as df
    import numpy as np
    img = np.zeros((128, 128))
    res = df.dilate_and_erode(img)
    assert np.shape(img) == np.shape(res)


def test_create_text():
    import deepmeta._dock_widget as dw
    vols = [0, 5, 3, 4]
    _, prop = dw.create_text(vols)
    assert {"vol": vols} == prop


def test_show_shapes(make_napari_viewer):
    import deepmeta._dock_widget as dw
    import deepmeta.deepmeta_functions as df
    import numpy as np
    mask = [
        np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
    ]
    res = df.from_mask_to_non_plottable_list(mask)
    viewer = make_napari_viewer()
    dw.show_shapes(viewer, res, [[5]], 'red')
    assert len(viewer.layers) == 1


def test_load_config():
    import deepmeta.deepmeta_functions as df
    from pathlib import Path
    from appdirs import user_config_dir
    cfg_loc = Path(user_config_dir(appname="deepmeta")) / "config.ini"
    assert df.load_config()["Deepmeta"] is not None
    assert cfg_loc.exists()
