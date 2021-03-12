Installation
============

Stable version
--------------

Install the most recent stable version using :py:obj:`pip`:

.. code-block:: bash

    pip install starry-process

To ensure you have the dependencies to run the :py:obj:`starry-process`
web app locally, do

.. code-block:: bash

    pip install starry-process[app]

To run the web app, just execute :py:obj:`starry-process` on the command line.

.. note ::

   The ``starry_process`` package requires Python 3.6 or later.

Additional requirements (installed if not present) are:

- ``numpy>=1.19.2``
- ``scipy>=1.5.0``
- ``matplotlib``
- ``pymc3``
- ``pymc3-ext``
- ``tqdm``


Development version
-------------------

You can install the latest development version of :py:obj:`starry_process` directly
from `GitHub <https://github.com/rodluger/starry_process>`_:

.. code-block:: bash

    git clone https://github.com/rodluger/starry_process.git
    cd starry_process
    pip install .

To ensure you have all dependencies to run unit tests, perform
calibration runs, build the docs, and/or to reproduce the results in the paper:

.. code-block:: bash

    git clone https://github.com/rodluger/starry_process.git
    cd starry_process
    pip install -e ".[tests,docs]"
