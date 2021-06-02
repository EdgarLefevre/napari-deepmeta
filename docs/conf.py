# -*- coding: utf-8 -*-

import sys
import os
import shlex
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'sphinx.ext.ifconfig',
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'napari-deepmeta'
copyright = u'2021, Edgar Lefevre'
author = u'Edgar Lefevre'
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
htmlhelp_basename = 'napari-cookiecutterplugin_namedoc'
latex_elements = {}
latex_documents = [
  (master_doc, 'napari-cookiecutterplugin_name.tex', u'napari-\\{\\{cookiecutter.plugin\\_name\\}\\} Documentation',
   u'\\{\\{cookiecutter.full\\_name\\}\\}', 'manual'),
]
man_pages = [
    (master_doc, 'napari-cookiecutterplugin_name', u'napari-deepmeta Documentation',
     [author], 1)
]
texinfo_documents = [
  (master_doc, 'napari-cookiecutterplugin_name', u'napari-deepmeta Documentation',
   author, 'napari-cookiecutterplugin_name', 'One line description of project.',
   'Miscellaneous'),
]
