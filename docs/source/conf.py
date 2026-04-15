import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'PyTorch Geometric Signed Directed'
author = 'Yixuan He'
copyright = f'{datetime.datetime.now().year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

master_doc = 'index'
source_suffix = '.rst'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
    "navigation_depth": 2,
}

html_logo = '_static/img/text_logo.jpg'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

add_module_names = False
autodoc_mock_imports = [
    'torch',
    'torch_geometric',
    'torch_sparse',
    'torch_scatter',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = ['__init__', '__repr__', '__weakref__', '__dict__', '__module__']
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)