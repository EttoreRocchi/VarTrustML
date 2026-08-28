# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))
from vartrustml._version import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "VarTrustML"
copyright = f"{datetime.now().year}, Ettore Rocchi"
author = "Ettore Rocchi"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    # Type hints in documentation
    "sphinx_autodoc_typehints",
    # CLI documentation (Typer/Click)
    "sphinx_click",
    # Enhanced features
    "sphinx_copybutton",
    "sphinx_design",
    # Markdown support (used to include CHANGELOG.md verbatim)
    "myst_parser",
]

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]

# -- Autodoc configuration ---------------------------------------------------

# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    "optuna",
    "optuna_integration",
    "xgboost",
    "catboost",
    "shap",
    "plotly",
    "torch",
    "sklearn",
    "scipy",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "joblib",
    "tqdm",
]

autodoc_default_options = {
    # Don't include members by default - add :members: explicitly where needed
    # This prevents dataclass fields from being documented twice
    "member-order": "bysource",
    "exclude-members": "__weakref__, __init__, __new__",
    "show-inheritance": True,
}

# For sphinx-autodoc-typehints: merge type info into description for cleaner output
typehints_defaults = "comma"  # Show defaults inline with type

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"

# Napoleon settings for NumPy/scikit-learn style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
# Use :ivar: for Attributes section - integrates better with autodoc for dataclasses
napoleon_use_ivar = True
# Use Parameters/Returns sections (NumPy style) instead of :param:/:rtype:
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None
# Don't create duplicate attribute entries - napoleon handles them via :ivar:
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output configuration -----------------------------------------------

html_theme = "furo"

html_title = "VarTrustML"
html_short_title = "VarTrustML"

# Furo theme options for academic appearance
html_theme_options = {
    # Light/dark mode
    "light_css_variables": {
        "color-brand-primary": "#008080",
        "color-brand-content": "#008080",
        "color-admonition-background": "rgba(0, 128, 128, 0.1)",
        # Typography for academic feel
        "font-stack": "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif",
        "font-stack--monospace": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00b3b3",
        "color-brand-content": "#00b3b3",
    },
    # Navigation
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
    # Top of page
    "top_of_page_button": "edit",
    # Footer
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/EttoreRocchi/VarTrustML",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    # Source repository
    "source_repository": "https://github.com/EttoreRocchi/VarTrustML",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Static files
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"

# -- LaTeX output for academic PDF export ------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
        \usepackage{charter}
        \usepackage[defaultsans]{lato}
        \usepackage{inconsolata}
    """,
}

latex_documents = [
    (
        master_doc,
        "VarTrustML.tex",
        "VarTrustML Documentation",
        "Ettore Rocchi",
        "manual",
    ),
]

# -- Copy button configuration -----------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Todo extension ----------------------------------------------------------

todo_include_todos = True
