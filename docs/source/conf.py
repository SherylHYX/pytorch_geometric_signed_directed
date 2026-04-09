import datetime
import doctest
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

source_suffix = '.rst'
master_doc = 'index'

project = 'PyTorch Geometric Signed Directed'
author = 'Yixuan He'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 2,
}

html_logo = '_static/img/text_logo.jpg'
html_static_path = ['_static']
html_context = {'css_files': ['_static/css/custom.css']}

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}
add_module_names = False
autosummary_generate = True

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