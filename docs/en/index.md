# Overview

[中文](../zh/index.md) · [GitHub](https://github.com/jiadonglee/spbin)


spbin is a Python package for downloading, modelling and fitting stellar spectral energy distributions (SEDs). It combines Gaia XP spectra, 2MASS and AllWISE photometry, and optional SPHEREx spectra on an absolute physical flux scale. Stellar evolution and spectral models describe single stars and unresolved binaries, while dedicated routes cover giants, hot subdwarfs and DA white dwarfs. The predicted light ratios can be connected to Gaia photocentre orbits.

Core fitting uses NumPy, SciPy and Matplotlib. Network weights, evolutionary tables and spectral-model assets ship with the package. Basic fitting requires no JAX, GPU or separate J-CAPS installation.

Homepage description:

> From Gaia XP to physical binary models. spbin models spectra and infrared photometry at the measured distance, compares single stars and binaries, studies white-dwarf and hot-subdwarf companions, and connects their light to Gaia photocentre orbits.

The package is intended for researchers working on stellar populations, binaries and Gaia data. Start from a Gaia source ID or supply your own flux arrays, then run the analysis in a local Python environment.


## Capabilities

| Capability | Entry points | Inputs | Outputs |
| --- | --- | --- | --- |
| Single-source XP and infrared acquisition | `download` | Source ID or sky coordinates | Standard `SED` and local cache |
| Catalogue queries and native batch downloads | `query_gaia`, `download_gaia` | ADQL or source IDs | Astropy Table, native FITS ZIP records |
| SPHEREx acquisition and import | `download_spherex`, `load_spherex` | `SED` or XphereX CSV | `SED` with SPHEREx channels |
| Single-star and coeval-binary fitting | `fit` | `SED`, age and metallicity settings | Masses, mass ratio, fluxes, fit statistics |
| HR-diagram location and route selection | `locate`, `fit_star` | Gaia photometry, Ks and parallax | Region, selected hypotheses and fits |
| Giant plus hot main-sequence companion | `fit_giant_companion` | `SED` and giant label priors | Mass profile, improvement, conditional exclusion |
| Hot subdwarf plus cool companion | `fit_subdwarf_companion` | `SED`, labels and extinction constraints | Hypotheses, radius, passband light fractions |
| DA white dwarf plus main-sequence companion | `fit_whitedwarf_companion` | `SED`, cooling relation or free radius | Component parameters and hypothesis comparison |
| Conditional WD light envelope | `whitedwarf_light_limit` | Mass/temperature or cooling-age grid | Upper G-light envelope and allowed intervals |
| Photocentre-orbit branches | `spbin.orbit` | Orbit scale, period, parallax, mass/light | Dark, faint and luminous mass solutions |
| External joint inference | `loglike_sed`, `loglike_whitedwarf_sed` | Shared physical proposal | SED log likelihood |
| Scientific plotting | `plot` | `SED` and standard `fit` output | Matplotlib Figure, PNG/PDF |

