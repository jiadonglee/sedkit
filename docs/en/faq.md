# FAQ

[中文](../zh/faq.md) · [GitHub](https://github.com/jiadonglee/spbin)


**Why does fitting require positive parallax?** Absolute flux prediction uses parallax to set distance. Supply a physically justified positive parallax and uncertainty, or use an external joint model for a more general distance posterior.

**Why is SPHEREx absent from hot-star fits?** `StellarModel(hot=True)` has a shared fitting mask restricted to XP and J/H/Ks.

**Why does a call return None?** `StellarModel.evaluate` may encounter unsupported components; `fit_star` may encounter an unsupported region or missing giant labels. Inspect the route reason and choose a model covering that physical region.

**Why does `extinction=None` produce a map error?** Base `fit` uses EdenhoferPrior. Install the dust extra, configure map files and provide coordinates. A numeric `extinction=` fixes E.

**Why does a batch download not return SED?** `download_gaia` supplies native products. Use `download` for a calibrated single-source object.

**How can an analysis be repeated?** Save the observed `SED`; record the software commit, fit settings, priors, route and final mask; save fit results and figures.


## Examples, support and license

- [Basic SED notebook](https://github.com/jiadonglee/spbin/blob/main/examples/01_sed_fit.ipynb)
- [Gaia luminous-twin branches](https://github.com/jiadonglee/spbin/blob/main/examples/02_gaia_orbit_twin.ipynb)
- [Warm-binary experiments](https://github.com/jiadonglee/spbin/blob/main/examples/03_warm_binaries.ipynb)
- [Extinction/parallax notebook](https://github.com/jiadonglee/spbin/blob/main/examples/05_extinction_parallax.ipynb)
- [White-dwarf script](https://github.com/jiadonglee/spbin/blob/main/examples/06_whitedwarf.py)
- [GitHub Issues](https://github.com/jiadonglee/spbin/issues)
- [Source license](https://github.com/jiadonglee/spbin/blob/main/LICENSE): package metadata declares MIT AND BSD-2-Clause; retain the associated copyright and license text.

Follow the citation requirements of Gaia, 2MASS, WISE, SPHEREx and the relevant atmosphere/evolutionary models. See [data and provenance](../data.md).

