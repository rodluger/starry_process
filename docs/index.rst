.. raw:: html

   <div align="center">
   <img src="https://github.com/rodluger/starry_process/blob/master/starry_process.gif?raw=true" width="450px">
   </img>
   <br/>
   </div>
   <br/><br/>

Documentation
=============

Welcome to the :py:mod:`starry_process` documentation.
The :py:mod:`starry_process` code is an implementation of an
**interpretable Gaussian process (GP) for stellar light curves.**
This means that the hyperparameters of the GP are actual
physical properties of the stellar surface, such as the size,
position, contrast, and number of star spots. The primary application
of :py:mod:`starry_process` is to model stellar light curves
with the goal of inferring their spot parameters.
For more information, check out the
`JOSS paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210201774L>`_, the
Mapping Stellar Surfaces paper series
(`Paper I <https://ui.adsabs.harvard.edu/abs/2021arXiv210200007L>`_,
`Paper II <https://ui.adsabs.harvard.edu/abs/2021arXiv210201697L>`_),
as well as this `interactive live demo <http://starry-process.flatironinstitute.org>`_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

    Installation <install>
    Examples <examples>
    API <api>
    Live demo <http://starry-process.flatironinstitute.org>
    GitHub <https://github.com/rodluger/starry_process>
    Submit an issue <https://github.com/rodluger/starry_process/issues>
    Read the JOSS paper <https://ui.adsabs.harvard.edu/abs/2021arXiv210201774L>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
