from typing import TYPE_CHECKING

import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


def load_img(obj):
    import os

    import skimage.io as io

    import napari_deepmeta.deepmeta_functions as df

    img = io.imread(
        os.path.dirname(os.path.realpath(__file__))
        + "/resources/souris_8.tif",
        plugin="tifffile",
    )
    img = df.contrast_and_reshape(img).reshape(128, 128, 128)
    obj.viewer.add_image(img, name="mouse")
    return img


def show_total_vol(layout, masks, organ, nb=None):
    vol_tot = masks.sum() * 0.0047
    elt = QLabel(f"Total volume {organ} {vol_tot:.3f}mm3")
    layout.addWidget(elt)
    if nb is not None:
        elt2 = QLabel(f"Metastases number {nb}")
        layout.addWidget(elt2)


def clean_labels(layout):
    if layout.count() >= 6:
        try:
            layout.takeAt(5).widget().setParent(None)
            layout.takeAt(5).widget().setParent(None)
            layout.takeAt(5).widget().setParent(None)
        except AttributeError:
            pass


def show_error(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText(text)
    msg.setWindowTitle("Error")
    msg.exec_()


def show_shapes_3D(viewer, plottable, color, text):
    try:
        viewer.add_shapes(
            plottable,
            shape_type="path",
            edge_width=0.5,
            edge_color=color,
            face_color="#6a6a6aff",
            opacity=0.6,
            name=text,
        )
    except Exception as e:
        print(e)
        print(np.shape(plottable))


class DeepmetaWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Segment Stack")
        btn.clicked.connect(self._on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)

        check = QCheckBox("Contrast ?", self)
        check.stateChanged.connect(self._click_box)
        self.contrast = False
        self.layout().addWidget(check)

        check2 = QCheckBox("Post-processing ?", self)
        check2.stateChanged.connect(self._click_box2)
        self.postprocess = False
        self.layout().addWidget(check2)

        check3 = QCheckBox("Show metastases ?", self)
        check3.stateChanged.connect(self._click_box3)
        self.metas = False
        self.layout().addWidget(check3)

    def _click_box(self, state):
        self.contrast = state == QtCore.Qt.Checked

    def _click_box2(self, state):
        self.postprocess = state == QtCore.Qt.Checked

    def _click_box3(self, state):
        self.metas = state == QtCore.Qt.Checked

    def _on_click(self):
        import napari_deepmeta.deepmeta_functions as df

        if len(self.viewer.layers) == 1:
            img = df.prepare_mouse(self.viewer.layers[0].data, self.contrast)
            output = df.segment_stack(img)
            output = output.max(1).indices.detach().numpy()
            if self.postprocess:
                output = df.postprocess(img, output)
            masks = [mask > 0.5 for mask in output]
            plottable_list = df.mask_to_plottable_3D(masks)
            clean_labels(self.layout())
            show_shapes_3D(self.viewer, plottable_list, "red", "Lung masks")
            show_total_vol(self.layout(), np.array(masks), "lungs")
            if self.metas:
                masks = [mask > 1.5 for mask in output]
                plottable_list = df.mask_to_plottable_3D(masks)
                show_shapes_3D(
                    self.viewer, plottable_list, "blue", "Metastases masks"
                )
                show_total_vol(
                    self.layout(),
                    np.array(masks),
                    "metastases",
                    nb=df.get_meta_nb(masks),
                )
            print("done")
        else:
            show_error(
                "Cannot run segmentation if you have multiple files opened."
            )


class DeepmetaDemoWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Run demo")
        btn.clicked.connect(self._on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        import napari_deepmeta.deepmeta_functions as df

        img = load_img(self)
        img = df.prepare_mouse(img, False)
        output = df.segment_stack(img)
        output = output.max(1).indices.detach().numpy()
        output = df.postprocess(img, output)
        masks = [mask > 0.5 for mask in output]
        plottable_list = df.mask_to_plottable_3D(masks)
        clean_labels(self.layout())
        show_shapes_3D(self.viewer, plottable_list, "red", "Lung masks")
        show_total_vol(self.layout(), np.array(masks), "lungs")
        masks = [mask > 1.5 for mask in output]
        plottable_list = df.mask_to_plottable_3D(masks)
        show_shapes_3D(self.viewer, plottable_list, "blue", "Metastases masks")
        show_total_vol(
            self.layout(),
            np.array(masks),
            "metastases",
            nb=df.get_meta_nb(masks),
        )
        print("done")
