Installation
============

Development version
-------------------

You can install the latest development version of :py:obj:`starry_process` directly
from `GitHub <https://github.com/rodluger/starry_process>`_:

.. code-block:: bash

    git clone https://github.com/rodluger/starry_process.git
    cd starry_process
    pip install .

If you also want to install the `starry-process` web app locally, do

.. code-block:: bash

    git clone https://github.com/rodluger/starry_process.git
    cd starry_process
    pip install -e ".[app]"

To ensure you have all dependencies to run unit tests, perform
calibration runs, and/or to reproduce the results in the paper:

.. code-block:: bash

    git clone https://github.com/rodluger/starry_process.git
    cd starry_process
    pip install -e ".[tests]"