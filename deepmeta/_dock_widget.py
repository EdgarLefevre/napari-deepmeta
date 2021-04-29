"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton


class SegmentLungs(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Run Lung Seg")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            non_plottable = df.seg_lungs(image)
            for contours in non_plottable:
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='red', face_color="#6a6a6aff",
                                       opacity=0.3
                                       )
        else:
            print("You do not have only one image opened.")

class SegmentMetas(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        btn = QPushButton("Run Metastasis Seg")
        btn.clicked.connect(self._on_click)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        import deepmeta.deepmeta_functions as df
        if len(self.viewer.layers) == 1:
            image = self.viewer.layers[0].data / 255
            non_plottable = df.seg_metas(image)
            for contours in non_plottable:
                self.viewer.add_shapes(contours, shape_type='polygon', edge_width=1,
                                       edge_color='blue', face_color="#6a6a6aff",
                                       opacity=0.3
                                       )
        else:
            print("You do not have only one image opened.")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [SegmentLungs, SegmentMetas]

# todo: check image shape
# todo: contrast (checkbox serait lourd, sinon btn seg lungs contrast)
