import os
import sys
sys.path.insert(0, os.path.abspath('..'))

try:
    import sympy_matrices_random as pkg
except ImportError:
    pkg = None


project = 'Random Matrices for SymPy'
copyright = '2026, Sonntagsgesicht'
author = 'Sonntagsgesicht'
version = '0.1.1'
release = '0.1.1'

extensions = [
    'sphinx_pytype_substitution',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    # 'm2r2'
]

# source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#html_theme = 'sphinx_rtd_theme'
html_theme = "furo"
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sympy': ('https://docs.sympy.org/latest', None),
}

# Myst configuration (for .md files). See
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
## myst_enable_extensions = ["dollarmath", "linkify", "tasklist"]
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 6
# Make - [ ] checkboxes from the tasklist extension checkable
# Requires https://github.com/executablebooks/MyST-Parser/pull/686
myst_enable_checkboxes = True
# myst_update_mathjax = False

# Don't linkify links unless they start with "https://". This is needed
# because the linkify library treats .py as a TLD.
myst_linkify_fuzzy_links = False

# -- Config for pytype_substitution extension ------------------------------

pytype_substitutions = pkg,  # package, module or class to reference to
pytype_buildins = False  # not implemented in v0.1
pytype_short_ref = True  # drop module from reference (if it does not conflict)
pytype_match_pattern = ''  # regex to filter entities to ref to
pytype_exclude_pattern = ''  # regex to exclude entities to ref to
pytype_show = False

# Configure Sphinx copybutton (see https://sphinx-copybutton.readthedocs.io/en/latest/use.html)
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

html_logo = "_static/sympylogo.png"

html_sidebars = {}
