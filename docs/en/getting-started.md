# Installation

[中文](../zh/getting-started.md) · [GitHub](https://github.com/jiadonglee/spbin)



Python 3.10 or later is required. Install from GitHub:

```bash
python -m pip install "spbin @ git+https://github.com/jiadonglee/spbin.git"
```

Include catalogue and spectrum acquisition:

```bash
python -m pip install "spbin[download] @ git+https://github.com/jiadonglee/spbin.git"
```

| Extra | Purpose | Dependencies |
| --- | --- | --- |
| `download` | Gaia and related catalogue acquisition | Astropy, Astroquery, GaiaXPy, Requests, PyVO |
| `dust` | Edenhofer 3D extinction prior | dustmaps, Astropy |
| `spherex` | Isolated SPHEREx extraction worker | uv |
| `notebook` | Interactive examples | JupyterLab, ipykernel |
| `test` | Repository tests | pytest and related data tools |

```bash
python -m pip install "spbin[download,dust,spherex,notebook] @ git+https://github.com/jiadonglee/spbin.git"
```

For local development:

```bash
git clone https://github.com/jiadonglee/spbin.git
cd spbin
python -m pip install -e ".[download,notebook]"
```

Verify the import:

```python
import spbin
print(spbin.__version__)
```

For a fixed analysis version, replace the floating branch with a commit or an existing tag: `git+https://github.com/jiadonglee/spbin.git@<commit>`.


## First fit

```python
from spbin import download, fit, plot

sed = download("858860697467058688", cache_dir="data")
result = fit(sed, age_gyr=5.0, feh=0.0)

print("Single objective:", result["single"]["objective"])
print("Binary objective:", result["binary"]["objective"])
print("Delta:", result["delta"])
print("Mass ratio:", result["binary"]["q"])

sed.save("data/example_sed.npz")
fig = plot(sed, result, path="example_sed.png")
```

This fixes age at 5 Gyr, metallicity at the solar value, extinction at zero and parallax at its catalogue value. `delta = objective_single - objective_binary`; a positive value means the binary objective is lower. It measures fit improvement under the selected models and constraints.

To fit age and metallicity:

```python
result = fit(sed, age_gyr=None, feh=None)
```

For a single-star fit:

```python
single = fit(sed, kind="single", age_gyr=5.0, feh=0.0)
print(single["m1"], single["converged"], single["at_bounds"])
```

