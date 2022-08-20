<p align="center">
  <img width="450" src="https://raw.githubusercontent.com/rodluger/starry_process/master/starry_process.gif"/>
  <br/>
  <a href="https://github.com/rodluger/starry_process/actions?query=workflow%3Atests">
    <img src="https://github.com/rodluger/starry_process/workflows/tests/badge.svg"/>
  </a>
  <a href='https://starry-process.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/starry-process/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a style="border-width:0" href="https://doi.org/10.21105/joss.03071">
    <img src="https://joss.theoj.org/papers/10.21105/joss.03071/status.svg" alt="DOI badge" >
  </a>
  <br/>
  <a href="http://starry-process.flatironinstitute.org">
    <img src="https://img.shields.io/badge/web-app-orange.svg?style=flat"/>
  </a>
  <a href="https://github.com/rodluger/mapping_stellar_surfaces/raw/paper1-pdf/ms.pdf">
    <img src="https://img.shields.io/badge/read-paper_1-blue.svg?style=flat"/>
  </a>
  <a href="https://github.com/rodluger/mapping_stellar_surfaces/raw/paper2-pdf/ms.pdf">
    <img src="https://img.shields.io/badge/read-paper_2-blue.svg?style=flat"/>
  </a>
</p>

<p align="center">
Interpretable Gaussian processes for stellar light curves using <a href="https://github.com/rodluger/starry">starry</a>.
</p>

# A Gaussian Process for Stellar Variability

The `starry_process` code implements an interpretable Gaussian process (GP)
for modeling stellar light curves. Whether your goal is to marginalize
over the stellar variability signal (if you think of it as noise)
or to understand the surface features that generated it (if you
think of it as data), this code is for you. The GP implemented here works
just like any other GP you might already use in your analysis, except that
its hyperparameters are *physically interpretable*. These are (among others)
the **radius of the spots**, the
**mean and variance of the latitude distribution**,
the **spot contrast**, and the **number of spots**. Users can also specify
things like the rotational period of the star, the limb darkening parameters,
and the inclination (or marginalize over the inclination if it is not known).

The code is written in Python and relies on the
[Theano package](https://theano-pymc.readthedocs.io/en/stable/index.html),
so a little familiarity with that is recommended. Check out the crash
course [here](https://starry-process.readthedocs.io/en/latest/notebooks/Quickstart/#Compiling-theano-functions).
If you would like to report an issue or contribute to the project, please
check out [CONTRIBUTING.md](CONTRIBUTING.md).

# Installation

The quickest way is via `pip`:

```bash
pip install starry-process
```

Note that the `starry_process` package requires Python 3.6 or later.

# Quickstart

Import the main interface:

```python
from starry_process import StarryProcess
```

Draw samples from a Gaussian process with small mid-latitude spots:

```python
import numpy as np
import matplotlib.pyplot as plt

# Instantiate the GP
sp = StarryProcess(
  r=10,               # spot radius in degrees
  mu=30,              # central spot latitude in degrees
  sigma=5,            # latitude std. dev. in degrees
  c=0.1,              # fractional spot contrast
  n=10                # number of spots
)

# Draw & visualize a spherical harmonic sample
y = sp.sample_ylm().eval()
sp.visualize(y)

# Compute & plot the flux at some inclination
t = np.linspace(0, 4, 1000)
flux = sp.flux(y, t, i=60).eval()[0]
plt.plot(t, flux)
```

<img src="https://raw.githubusercontent.com/rodluger/starry_process/master/docs/samples_0.png"/>

Same as above, but for high-latitude spots:

```python
sp = StarryProcess(r=10, mu=0, sigma=10, c=0.1, n=10)
```

<img src="https://raw.githubusercontent.com/rodluger/starry_process/master/docs/samples_1.png"/>

Large equatorial spots:

```python
sp = StarryProcess(r=30, mu=0, sigma=10, c=0.1, n=10)
```

<img src="https://raw.githubusercontent.com/rodluger/starry_process/master/docs/samples_2.png"/>

Small, approximately isotropic spots:

```python
sp = StarryProcess(r=10, mu=0, sigma=40, c=0.1, n=10)
```

<img src="https://raw.githubusercontent.com/rodluger/starry_process/master/docs/samples_3.png"/>

For more information check out the full
[Quickstart tutorial](https://starry-process.readthedocs.io/en/latest/notebooks/Quickstart) and
the complete [documentation](https://starry-process.readthedocs.io/en/latest).

# References & Attribution

The code is described in this
[JOSS paper](https://github.com/rodluger/starry_process/raw/joss-paper/joss/paper.pdf).
It is the backbone of the
[Mapping Stellar Surfaces](https://github.com/rodluger/mapping_stellar_surfaces)
paper series, including:

  - [Degeneracies in the rotational light curve problem](https://github.com/rodluger/mapping_stellar_surfaces/raw/paper1-pdf/ms.pdf)
  - [An interpretable Gaussian process model for stellar light curves](https://github.com/rodluger/mapping_stellar_surfaces/raw/paper2-pdf/ms.pdf)

If you make use of this code in your research, please cite

```
@article{Luger2021a,
  author        = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and {Hedges}, Christina and {Hogg}, David W.},
  title         = {{Mapping stellar surfaces I: Degeneracies in the rotational light curve problem}},
  journal       = {arXiv e-prints},
  keywords      = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
  year          = 2021,
  month         = jan,
  eid           = {arXiv:2102.00007},
  pages         = {arXiv:2102.00007},
  archiveprefix = {arXiv},
  eprint        = {2102.00007},
  primaryclass  = {astro-ph.SR},
  adsurl        = {https://ui.adsabs.harvard.edu/abs/2021arXiv210200007L},
  adsnote       = {Provided by the SAO/NASA Astrophysics Data System}
}
```

```
@article{Luger2021b,
  author        = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and {Hedges}, Christina},
  title         = {{Mapping stellar surfaces II: An interpretable Gaussian process model for light curves}},
  journal       = {arXiv e-prints},
  keywords      = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
  year          = 2021,
  month         = feb,
  eid           = {arXiv:2102.01697},
  pages         = {arXiv:2102.01697},
  archiveprefix = {arXiv},
  eprint        = {2102.01697},
  primaryclass  = {astro-ph.SR},
  adsurl        = {https://ui.adsabs.harvard.edu/abs/2021arXiv210201697L},
  adsnote       = {Provided by the SAO/NASA Astrophysics Data System}
}
```

```
@article{Luger2021c,
  author        = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and {Hedges}, Christina},
  title         = {{starry\_process: Interpretable Gaussian processes for stellar light curves}},
  journal       = {arXiv e-prints},
  keywords      = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
  year          = 2021,
  month         = feb,
  eid           = {arXiv:2102.01774},
  pages         = {arXiv:2102.01774},
  archiveprefix = {arXiv},
  eprint        = {2102.01774},
  primaryclass  = {astro-ph.SR},
  adsurl        = {https://ui.adsabs.harvard.edu/abs/2021arXiv210201774L},
  adsnote       = {Provided by the SAO/NASA Astrophysics Data System}
}
```
