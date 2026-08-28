============
Installation
============

Installing VarTrustML and its dependencies.

Requirements
------------

VarTrustML requires **Python 3.10 or newer**. All runtime dependencies are installed automatically; a fully pinned environment is available in ``requirements.lock.txt``.

Installation Methods
--------------------

From source
^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/VarTrustML.git
   cd VarTrustML
   pip install .

This installs VarTrustML and makes the ``vartrustml`` command available. Optional
extras are ``[dev]`` for the test and lint toolchain, ``[docs]`` to build the
documentation, and ``[dev,docs]`` for both.

Developer Installation
^^^^^^^^^^^^^^^^^^^^^^

Clone the repository and install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/VarTrustML.git
   cd VarTrustML
   pip install -e ".[dev]"

Verifying Installation
----------------------

Check that the CLI is on your path:

.. code-block:: bash

   vartrustml --help

You should see the help message with available commands.

Run the smoke test:

.. code-block:: bash

   vartrustml smoke-test

This verifies that all dependencies are correctly installed.

Optional: Creating a Virtual Environment
----------------------------------------

We recommend using a virtual environment:

Using venv
^^^^^^^^^^

.. code-block:: bash

   python -m venv vartrustml-env
   source vartrustml-env/bin/activate  # On Windows: vartrustml-env\Scripts\activate

Using conda
^^^^^^^^^^^

.. code-block:: bash

   conda create -n vartrustml python=3.10
   conda activate vartrustml

Then proceed with the installation steps above.

GPU Support (Optional)
----------------------

For faster training with XGBoost:

XGBoost GPU
^^^^^^^^^^^

.. code-block:: bash

   pip install xgboost[gpu]

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

ImportError: No module named 'vartrustml'
"""""""""""""""""""""""""""""""""""""""""

Make sure you installed the package:

.. code-block:: bash

   pip install -e .

And that you're in the correct environment.

XGBoost version conflict
""""""""""""""""""""""""

VarTrustML requires XGBoost < 3.0.0:

.. code-block:: bash

   pip install 'xgboost<3.0.0'

Plotly visualizations not working
"""""""""""""""""""""""""""""""""

Ensure plotly is installed:

.. code-block:: bash

   pip install plotly

Getting Help
^^^^^^^^^^^^

If you encounter issues:

1. Search `GitHub Issues <https://github.com/EttoreRocchi/VarTrustML/issues>`_
2. Open a new issue with:

   - Python version (``python --version``)
   - OS information
   - Full error message
   - Steps to reproduce

Next Steps
----------

- :doc:`quickstart` -- Learn basic usage
- :doc:`../cli-reference/index` -- Explore commands
- :doc:`../user-guide/compare-models` -- Model comparison documentation
