<p align="center">
  <img width="450" src="https://raw.githubusercontent.com/rodluger/starry_process/master/starry_process.gif"/>
  <br/>
  <a href="https://github.com/rodluger/starry_process/actions?query=workflow%3Atests">
    <img src="https://github.com/rodluger/starry_process/workflows/tests/badge.svg"/>
  </a>
  <a href="https://luger.dev/starry_process">
    <img src="https://github.com/rodluger/starry_process/workflows/docs/badge.svg"/>
  </a>
  <a href="https://github.com/rodluger/starry_process/raw/joss-paper/joss/paper.pdf">
    <img src="https://github.com/rodluger/starry_process/workflows/joss%20paper/badge.svg"/>
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
course [here](https://luger.dev/starry_process/notebooks/Quickstart.html#Compiling-theano-functions).

# Quickstart

Import the main interface:

```python
from starry_process import StarryProcess
```

Draw samples from a Gaussian process with small mid-latitude spots:

```python
# Instantiate the GP
sp = StarryProcess(
  r=10,               # spot radius in degrees
  mu=30,              # central spot latitude in degrees
  sigma=5,            # latitude std. dev. in degrees
  c=0.1,              # fractional spot contrast
  n=10                # number of spots
)

# Draw & visualize a spherical harmonic sample
y = sp.draw_ylm().eval()
sp.visualize(y)

# Compute & plot the flux at some inclination
t = np.linspace(0, 4, 1000)
flux = sp.flux(y, t, i=60).eval()
plt.plot(t, flux)
```

<img src="https://github.com/rodluger/starry_process/raw/gh-pages/_images/samples_0.png"/>

Same as above, but for high-latitude spots:

```python
sp = StarryProcess(r=10, mu=0, sigma=10, c=0.1, n=10)
```

<img src="https://github.com/rodluger/starry_process/raw/gh-pages/_images/samples_1.png"/>

Large equatorial spots:

```python
sp = StarryProcess(r=30, mu=0, sigma=10, c=0.1, n=10)
```

<img src="https://github.com/rodluger/starry_process/raw/gh-pages/_images/samples_2.png"/>

Small, approximately isotropic spots:

```python
sp = StarryProcess(r=10, mu=0, sigma=40, c=0.1, n=10)
```

<img src="https://github.com/rodluger/starry_process/raw/gh-pages/_images/samples_3.png"/>

For more information check out the full
[Quickstart tutorial](https://luger.dev/starry_process/notebooks/Quickstart.html) and
the complete [documentation](https://luger.dev).

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
  author  = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and
                  {Hedges}, Christina},
  title   = {{Mapping stellar surfaces I: Degeneracies in the rotational light curve problem}},
  journal = {in preparation},
  year    = {2021},
  url     = {https://github.com/rodluger/mapping_stellar_surfaces/raw/paper1-pdf/ms.pdf}
}
```

```
@article{Luger2021b,
  author  = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and
                  {Hedges}, Christina},
  title   = {{Mapping stellar surfaces II: An interpretable Gaussian process model for light curves}},
  journal = {in preparation},
  year    = {2021},
  url     = {https://github.com/rodluger/mapping_stellar_surfaces/raw/paper2-pdf/ms.pdf}
}
```

```
@article{Luger2021c,
  author  = {{Luger}, Rodrigo and {Foreman-Mackey}, Daniel and
                  {Hedges}, Christina},
  title   = {{starry\_process: Interpretable Gaussian processes for stellar light curves}},
  journal = {in preparation},
  year    = {2021},
  month   = {Jan},
  url     = {https://github.com/rodluger/starry_process/raw/joss-paper/joss/paper.pdf}
}
```
