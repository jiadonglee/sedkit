# Extinction and parallax

[中文](../zh/extinction.md) · [GitHub](https://github.com/jiadonglee/spbin)


```python
from spbin import EdenhoferPrior

dust = EdenhoferPrior()
result = fit(sed, extinction=None, dust_prior=dust, fit_parallax=True)
```

`EdenhoferPrior(query=None, sigma=None)` wraps a 3D dust query. `moments(ra, dec, distance_pc)` returns E mean and standard deviation; `penalty(extinction, mean, sigma)` gives the truncated-Gaussian constraint.

Prepare map data beforehand; construction and fitting do not download it:

```python
from dustmaps.edenhofer2023 import fetch
fetch(fetch_samples=True)
```

The posterior-sample map is about 19 GB and needs sufficient disk space and memory. Coordinates are ICRS degrees from `sed.metadata["ra"]` and `["dec"]`. Mean-only maps require an explicit E prior width. Out-of-map behavior is route-specific; see [extinction fitting](../extinction.md) and the giant API.

