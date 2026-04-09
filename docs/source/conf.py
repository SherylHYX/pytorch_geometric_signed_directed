import datetime
import doctest
import sphinx_rtd_theme

extensions = [
    'autoapi.extension',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

source_suffix = '.rst'
master_doc = 'index'
author = 'Yixuan He'
project = 'PyTorch Geometric Signed Directed'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_logo = '_static/img/text_logo.jpg'
html_static_path = ['_static']
html_context = {'css_files': ['_static/css/custom.css']}
add_module_names = False

autoapi_type = 'python'
autoapi_dirs = ['../../torch_geometric_signed_directed/']
autoapi_keep_files = True
autoapi_generate_api_docs = True

autodoc_mock_imports = [
    'torch',
    'torch_geometric',
    'torch_sparse',
    'torch_scatter',
]

def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = ['__init__', '__repr__', '__weakref__', '__dict__', '__module__']
        return True if name in members else skip

    app.connect('autodoc-skip-member', skip)