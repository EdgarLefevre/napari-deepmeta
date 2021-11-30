# napari-deepmeta

[![License](https://img.shields.io/github/license/EdgarLefevre/napari-deepmeta?label=license)](https://github.com/EdgarLefevre/napari-deepmeta/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-deepmeta.svg?color=green)](https://pypi.org/project/napari-deepmeta)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-deepmeta.svg?color=green)](https://python.org)
[![tests](https://github.com/EdgarLefevre/napari-deepmeta/workflows/tests/badge.svg)](https://github.com/EdgarLefevre/napari-deepmeta/actions)
[![codecov](https://codecov.io/gh/EdgarLefevre/napari-deepmeta/branch/main/graph/badge.svg?token=H41ZaCAg31)](https://codecov.io/gh/EdgarLefevre/napari-deepmeta)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-deepmeta)](https://napari-hub.org/plugins/napari-deepmeta)

Segment mouse lungs and metastasis on MRI images.

This plugin is a demo for the [Deepmeta project](https://github.com/EdgarLefevre/DeepMeta).

![Lungs segmentation](https://github.com/EdgarLefevre/napari-deepmeta/blob/main/docs/_static/screen_napari_lungs.png?raw=true)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-deepmeta` via [pip]:

    pip install napari-deepmeta

> We advise you to create a specific python virtual environment in order to have a clean installation.

## Usage
In a terminal, just type `napari` to open napari.

Open a (x, 128, 128) image, go in the plugin menu, add deepmeta plugin to the dock viewer and click on the button.

![Menu](https://github.com/EdgarLefevre/napari-deepmeta/blob/main/docs/_static/plugin_menu.png?raw=true)

By adding the dock widget, a menu will be created on the left of your Napari instance.

![Deepmeta panel](https://github.com/EdgarLefevre/napari-deepmeta/raw/main/docs/_static/panel.png?raw=true)

In this panel you will find two buttons and one checkbox:

+ The first button *Run lung seg* process segmentation and show the result (With the widget segment metas, the button is called *Run meta seg*).
+ The second, *Reprocess Volume*, is useful when you modify contours. It will reprocess all slices to give you a new volume.
+ The checkbox is here to enhance contrast if your image is dark.

## Demo

If you just want to see what we've done, you can try the plugin with the Demo button, this button will load an image and process it
as if you use the plugin in a classic way.

## Conf file and custom models

The first time you run the plugin a config file will be created at `~/.config/deepmeta/config.ini`.

![conf.ini](https://github.com/EdgarLefevre/napari-deepmeta/blob/main/docs/_static/confini.png?raw=true)

In this file you can find parameters for postprocessing loop and the path for the models.
Feel free to change values and colors to fit to your needs.

>If you want to try another model, you can change the path. Be careful to not having custom objects in you model, otherwise, you'll have to modify the code.

## Known issue

If you encounters this error : `libGL error: failed to load driver: swrast`

Copy your lib file into your anaconda lib folder : 

```sh
cp /usr/lib/libstdc++.so.6 (conda_path)/lib/
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-deepmeta" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/EdgarLefevre/napari-deepmeta/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

