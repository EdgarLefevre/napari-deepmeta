import numpy as np

from napari_deepmeta import DeepmetaDemoWidget, DeepmetaWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_deepmeta_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = DeepmetaWidget(viewer)

    # call our widget method
    my_widget._on_click()

    assert viewer.layout().count() == 5


def test_demo_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    my_widget = DeepmetaDemoWidget(viewer)

    # call our widget method
    my_widget._on_click()

    assert viewer.layout().count() == 8
