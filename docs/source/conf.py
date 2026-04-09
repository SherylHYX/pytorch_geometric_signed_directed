import datetime
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../'))

project = 'PyTorch Geometric Signed Directed'
author = 'Yixuan He'
copyright = f'{datetime.datetime.now().year}, {author}'

extensions = [
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

master_doc = 'modules/root'
source_suffix = '.rst'

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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

autoapi_type = 'python'
autoapi_dirs = [os.path.abspath('../../torch_geometric_signed_directed')]
autoapi_file_patterns = ['*.py']
autoapi_root = 'modules/_autoapi'
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

def setup(app):
    app.add_css_file('css/custom.css')