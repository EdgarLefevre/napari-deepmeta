import numpy as np
import scipy.ndimage as ndimage
import skimage.transform as transform
from napari_plugin_engine import napari_hook_implementation
from qtpy import QtCore
from qtpy.QtWidgets import QWidget, QPushButton, QCheckBox, QLabel, QVBoxLayout


def create_text(vols):
    text_parameters = {
        'text': 'volume : {vol:.3f}mm3',
        'size': 12,
        'color': 'lavender',  # mintcream works too
        # colorlist: https://github.com/vispy/vispy/blob/main/vispy/color/_color_dict.py
        'anchor': 'lower_right',
        'translation': [-1, 0],
        'blending': "additive"
    }
    properties = {
        'vol': vols,
    }
    return text_parameters, properties


def fix_contours(v, contours):
    contours.sort(key=len)
    v.sort()
    while len(contours) > len(v):
        contours.pop(0)
    return contours


def reprocess_volume(obj):
    if obj.layout().count() == 6:  # check if volume label is already here
        new_vol = 0
        layers = obj.viewer.layers[1:]
        for shape in layers:
            for contour in shape.data:
                contour = np.uint8(contour.round())
                mask = np.zeros((128, 128))
                mask[contour[:, 1], contour[:, 2]] = 1
                mask = ndimage.morphology.binary_fill_holes(mask)
                new_vol += mask.sum()
        obj.layout().itemAt(5).widget().setParent(None)
        elt = QLabel("New total volume {:.3f}mm3".format(new_vol * float(obj.cfg["Deepmeta"]["volume"])))
        obj.layout().addWidget(elt)


def show_shapes(viewer, non_plottable, vols, color):
    for i, contours in enumerate(non_plottable):
        v = vols[i]
        text_p, prop = create_text(v)
        try:
            viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                              edge_color=color, face_color="#6a6a6aff",
                              opacity=0.3, name="Mask " + str(i), properties=prop,
                              text=text_p
                              )
        except ValueError:
            contours = fix_contours(v, contours)
            text_p, prop = create_text(v)
            viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                              edge_color=color, face_color="#6a6a6aff",
                              opacity=0.3, name="Mask " + str(i), properties=prop,
                              text=text_p
                              )


def show_total_vol(layout, vols):
    if layout.count() == 6:  # check if volume label is already here
        layout.itemAt(5).widget().setParent(None)
    vol_tot = np.array([np.array(l).sum() for l in vols]).sum()
    elt = QLabel("Total volume {:.3f}mm3".format(vol_tot))
    layout.addWidget(elt)


def load_img(obj):
    import skimage.io as io
    import deepmeta.deepmeta_functions as df
    img = io.imread(obj.img_path, plugin="tifffile")
    img = df.contrast_and_reshape(img).reshape(128, 128, 128)
    obj.viewer.add_image(img, name="mouse")
    return img


def clean_layers(obj, vol_id=5):
    if len(obj.viewer.layers) != 0:
        while obj.viewer.layers:
            obj.viewer.layers.pop()
        try:
            obj.layout().itemAt(vol_id).widget().setParent(None)
        except:
            print("no volume displayed")


def prepare_image(obj):
    image = None
    if len(obj.viewer.layers) == 1:
        image = obj.viewer.layers[0].data / 255
        image = transform.resize(image, (len(image), 128, 128),
                                 anti_aliasing=True)
        clean_layers(obj)
        obj.viewer.add_image(image, name="mouse")
    else:
        print("You do not have only one image opened.")
    return image

class SegmentLungs(QWidget):
    def __init__(self, napari_viewer):
        import deepmeta.deepmeta_functions as df
        super().__init__()
        self.cfg = df.load_config()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Run Lung Seg")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)

        check = QCheckBox("Contrast ?", self)
        check.stateChanged.connect(self._click_box)
        self.contrast = False
        self.layout().addWidget(check)

        btn2 = QPushButton("Reprocess volume")
        btn2.clicked.connect(self._reprocess_volume)
        self.layout().addWidget(btn2)

        btn3 = QPushButton("Clear")
        btn3.clicked.connect(self._clean)
        self.layout().addWidget(btn3)

    def _clean(self):
        clean_layers(self)

    def _reprocess_volume(self):
        reprocess_volume(self)

    def _click_box(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        image = prepare_image(self)
        if image is not None:
            if self.contrast:
                image = df.contrast_and_reshape(image)
            non_plottable, vols = df.seg_lungs(image, self.cfg)
            show_total_vol(self.layout(), vols)
            show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_lungs"])


class SegmentMetas(QWidget):
    def __init__(self, napari_viewer):
        import deepmeta.deepmeta_functions as df
        super().__init__()
        self.cfg = df.load_config()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        btn = QPushButton("Run Metastasis Seg")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)

        check = QCheckBox("Contrast ?", self)
        check.stateChanged.connect(self._click_box)
        self.contrast = False
        self.layout().addWidget(check)

        btn2 = QPushButton("Reprocess volume")
        btn2.clicked.connect(self._reprocess_volume)
        self.layout().addWidget(btn2)

        btn3 = QPushButton("Clear")
        btn3.clicked.connect(self._clean)
        self.layout().addWidget(btn3)

    def _clean(self):
        clean_layers(self)

    def _reprocess_volume(self):
        reprocess_volume(self)

    def _click_box(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        image = prepare_image(self)
        if image is not None:
            if self.contrast:
                image = df.contrast_and_reshape(image)
            non_plottable, vols = df.seg_metas(image, self.cfg)
            show_total_vol(self.layout(), vols)
            show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_metas"])



class Demo(QWidget):
    def __init__(self, napari_viewer):
        import deepmeta.deepmeta_functions as df
        import os
        super().__init__()
        self.cfg = df.load_config()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.img_path = os.path.dirname(os.path.realpath(__file__)) + "/resources/souris_8.tif"

        btn = QPushButton("Demo Lung Seg")
        btn.clicked.connect(self._on_click)
        self.layout().addWidget(btn)

        btn2 = QPushButton("Demo Meta Seg")
        btn2.clicked.connect(self._on_click2)
        self.layout().addWidget(btn2)

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        clean_layers(self, 3)
        image = load_img(self)
        non_plottable, vols = df.seg_lungs(image, self.cfg)
        show_total_vol(self.layout(), vols)
        show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_lungs"])

    def _on_click2(self):
        import deepmeta.deepmeta_functions as df
        clean_layers(self, 3)
        image = load_img(self)
        non_plottable, vols = df.seg_metas(image, self.cfg)
        show_total_vol(self.layout(), vols)
        show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_metas"])


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [SegmentLungs, SegmentMetas, Demo]
