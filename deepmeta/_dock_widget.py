from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QPushButton, QCheckBox, QLabel, QVBoxLayout
from qtpy import QtCore
import numpy as np
import scipy.ndimage as ndimage


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


def fix_v(v, contours):
    # unused
    while len(v) < len(contours):
        v.append(0)
    return v


def fix_contours(v, contours):
    contours.sort(key=len)
    v.sort()
    while len(contours) > len(v):
        contours.pop(0)
    return contours


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
    vol_tot = np.array([np.array(l).sum() for l in vols]).sum()
    elt = QLabel("Total volume {:.3f}mm3".format(vol_tot))
    layout.addWidget(elt)


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

    def _reprocess_volume(self):
        if self.layout().count() == 4:
            new_vol = 0
            layers = self.viewer.layers[1:]
            for shape in layers:
                for contour in shape.data:
                    contour = np.uint8(contour.round())
                    mask = np.zeros((128, 128))
                    mask[contour[:, 1], contour[:, 2]] = 1
                    mask = ndimage.morphology.binary_fill_holes(mask)
                    new_vol += mask.sum()
            self.layout().itemAt(3).widget().setParent(None)
            elt = QLabel("New total volume {:.3f}mm3".format(new_vol * 0.0047))
            self.layout().addWidget(elt)

    def _click_box(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            im_shape = np.shape(image)
            try:
                if im_shape[-1] == 128 and im_shape[-2] == 128:
                    if self.contrast:
                        image = df.contrast_and_reshape(image)
                    non_plottable, vols = df.seg_lungs(image, self.cfg)
                    show_total_vol(self.layout(), vols)
                    show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_lungs"])
                else:
                    print("Image shape should be (X, 128, 128)")
            except IndexError:
                print("Image should at least have two dimensions")
        else:
            print("You do not have only one image opened.")


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

    def _reprocess_volume(self):
        new_vol = 0
        layers = self.viewer.layers[1:]
        for shape in layers:
            for contour in shape.data:
                contour = np.uint8(contour.round())
                mask = np.zeros((128, 128))
                mask[contour[:, 1], contour[:, 2]] = 1
                mask = ndimage.morphology.binary_fill_holes(mask)
                new_vol += mask.sum()
        elt = QLabel("New total volume {:.3f}mm3".format(new_vol * 0.0047))
        self.layout().addWidget(elt)

    def _click_box(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            im_shape = np.shape(image)
            try:
                if im_shape[-1] == 128 and im_shape[-2] == 128:
                    if self.contrast:
                        image = df.contrast_and_reshape(image)
                    non_plottable, vols = df.seg_metas(image, self.cfg)
                    show_total_vol(self.layout(), vols)
                    show_shapes(self.viewer, non_plottable, vols, self.cfg["Deepmeta"]["color_metas"])
                else:
                    print("Image shape should be (X, 128, 128)")
            except IndexError:
                print("Image should at least have two dimensions")
        else:
            print("You do not have only one image opened.")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [SegmentLungs, SegmentMetas]
