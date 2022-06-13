import numpy as np

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
# def test_deepmeta_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))
#
#     # create our widget, passing in the viewer
#     my_widget = DeepmetaWidget(viewer)
#
#     # call our widget method
#     my_widget._on_click()
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "done\n"


#
# def test_demo_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))
#
#     # create our widget, passing in the viewer
#     my_widget = DeepmetaDemoWidget(viewer)
#
#     # call our widget method
#     my_widget._on_click()
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "done\n"


def test_add_z():
    import napari_deepmeta.deepmeta_functions as df

    arr = np.array([[0, 1], [0, 1]])
    assert (df.add_z(arr, 3) == np.array([[3, 0, 1], [3, 0, 1]])).all()


def test_dilate_and_erode():
    import napari_deepmeta.deepmeta_functions as df

    img = np.zeros((128, 128))
    res = df.dilate_and_erode(img)
    assert np.shape(img) == np.shape(res)


def test_contrast():
    import napari_deepmeta.deepmeta_functions as df

    seg = np.ones((2, 128, 128))
    res = df.contrast_and_reshape(seg)
    assert np.shape(res) == (2, 1, 128, 128)
