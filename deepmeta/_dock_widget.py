from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QCheckBox, QLabel, QVBoxLayout
from qtpy import QtCore
import numpy as np


def create_text(vols):
    text_parameters = {
        'text': 'volume : {vol:.3f}',
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
    while len(v) < len(contours):
        v.append(0)
    return v


class SegmentLungs(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        btn = QPushButton("Run Lung Seg")
        btn.clicked.connect(self._on_click)
        check = QCheckBox("Contrast ?", self)
        check.stateChanged.connect(self.clickBox)
        self.contrast = False
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)
        self.layout().addWidget(check)

    def clickBox(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def show_total_vol(self, vols):
        vol_tot = np.array([np.array(l).sum() for l in vols]).sum()
        elt = QLabel("Total volume {:.3f}mm3".format(vol_tot))
        self.layout().addWidget(elt)

    def show_shapes(self, non_plottable, vols):
        for i, contours in enumerate(non_plottable):
            v = vols[i]
            text_p, prop = create_text(v)
            try:
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='red', face_color="#6a6a6aff",
                                       opacity=0.3, name="Mask " + str(i), properties=prop,
                                       text=text_p
                                       )
            except ValueError:
                v = fix_v(v, contours)
                text_p, prop = create_text(v)
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='red', face_color="#6a6a6aff",
                                       opacity=0.3, name="Mask " + str(i), properties=prop,
                                       text=text_p
                                       )

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            im_shape = np.shape(image)
            try:
                if im_shape[-1] == 128 and im_shape[-2] == 128:
                    if self.contrast:
                        image = df.contrast_and_reshape(image)
                    non_plottable, vols = df.seg_lungs(image)
                    self.show_total_vol(vols)
                    self.show_shapes(non_plottable, vols)
                else:
                    print("Image shape should be (X, 128, 128)")
            except IndexError:
                print("Image should at least have two dimensions")
        else:
            print("You do not have only one image opened.")


class SegmentMetas(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        btn = QPushButton("Run Metastasis Seg")
        btn.clicked.connect(self._on_click)

        check = QCheckBox("Contrast ?", self)
        check.stateChanged.connect(self.clickBox)
        self.contrast = False

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)
        self.layout().addWidget(check)

    def clickBox(self, state):
        if state == QtCore.Qt.Checked:
            self.contrast = True
        else:
            self.contrast = False

    def show_total_vol(self, vols):
        vol_tot = np.array([np.array(l).sum() for l in vols]).sum()
        elt = QLabel("Total volume {:.3f}mm3".format(vol_tot))
        self.layout().addWidget(elt)

    def show_shapes(self, non_plottable, vols):
        for i, contours in enumerate(non_plottable):
            v = vols[i]
            text, t_prop = create_text(v)
            try:
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='blue', face_color="#6a6a6aff",
                                       opacity=0.3, name="Mask " + str(i), text=text,
                                       properties=t_prop
                                       )
            except ValueError:
                v = fix_v(v, contours)
                text, t_prop = create_text(v)
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='blue', face_color="#6a6a6aff",
                                       opacity=0.3, name="Mask " + str(i), text=text,
                                       properties=t_prop
                                       )

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            im_shape = np.shape(image)
            try:
                if im_shape[-1] == 128 and im_shape[-2] == 128:
                    if self.contrast:
                        image = df.contrast_and_reshape(image)
                    non_plottable, vols = df.seg_metas(image)
                    self.show_total_vol(vols)
                    self.show_shapes(non_plottable, vols)
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

# todo: text property is capricious, need to bold that thing
# todo: refactor and clean code
# todo: btn reprocess volume (recalculer le volume si modif de masks)
