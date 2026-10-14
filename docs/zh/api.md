# 公开接口签名索引

[English](../en/api.md) · [GitHub](https://github.com/jiadonglee/spbin)


以下列出顶层导出、主要类方法、轨道接口和诊断工具的参数与默认值。`*` 后的参数必须使用关键字；类方法省略 self/cls。`SED.metadata` 的默认值是独立空字典；`wavelength_um` 是属性。`EdenhoferPrior.penalty` 为静态方法。`DATALINK` 是 Gaia 模块的默认服务 URL 常量。内部实现函数不属于这个公开接口索引。

### `spbin.SED`

```python
SED(flux, error, mask, parallax_mas, parallax_error_mas=0.0, source_id="", metadata=<dict>)
SED.wavelength_um  # property
SED.save(path)
SED.load(path)
SED.with_spherex(wavelength_um, flux, error, valid=None)
SED.fit_mask(use_wise=False)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/data.py)

### `spbin.StellarModel`

```python
StellarModel(hot=False)
StellarModel.labels(masses, age_gyr=5.0, feh=0.0)
StellarModel.in_domain(labels)
StellarModel.hot_weight(teff)
StellarModel.in_hot_domain(teff, logg, feh)
StellarModel.predict_labels(labels)
StellarModel.predict_hot(teff, logg, radius)
StellarModel.evaluate(m1, q=0.0, age_gyr=5.0, feh=0.0)
StellarModel.error_factors(prediction, scale=1.0)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/model.py)

### `spbin.fit`

```python
fit(sed, kind='both', *, model=None, age_gyr=5.0, feh=0.0, q=None, use_wise=False, fit_parallax=False, extinction=0.0, dust_prior=None, age_prior=None, logg_prior=None, teff_prior=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/fit.py)

### `spbin.GiantTemplate`

```python
GiantTemplate()
GiantTemplate.in_support(teff, logg, mh)
GiantTemplate.evaluate(teff, logg, mh)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/giant.py)

### `spbin.fit_giant_companion`

```python
fit_giant_companion(sed, teff, logg, feh, *, m2_grid=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0), companion_age_gyr=0.01, extinction_prior=None, tilt_sigma=0.15, use_wise=True, template=None, model=None, threshold=None, luminosity=None, parallax=None, dust_prior=None, coeval=False)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/giant.py)

### `spbin.SubdwarfModel`

```python
SubdwarfModel(correction='ab', calibration='bundled', wavelength_um=None)
SubdwarfModel.support(tier)
SubdwarfModel.in_support(teff, logg, tier)
SubdwarfModel.predict(teff, logg, radius, tier)
SubdwarfModel.fractional_error()
SubdwarfModel.passband(coarse, name)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/subdwarf.py)

### `spbin.fit_subdwarf_companion`

```python
fit_subdwarf_companion(sed, *, companions=('dwarf', 'subgiant'), tiers=('H', 'mid', 'He'), model=None, stellar=None, template=None, parallax=None, fit_parallax=False, extinction=None, extinction_prior=None, dust_prior=None, subdwarf_prior=None, companion_prior=None, companion_age_gyr=None, companion_feh=None, companion_mass=None, use_wise=False, use_spherex=False, blue=None, galex=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/subdwarf.py)

### `spbin.WhiteDwarfModel`

```python
WhiteDwarfModel(calibration='bundled', hydrogen_layer='thick')
WhiteDwarfModel.physical(teff, mass)
WhiteDwarfModel.predict(teff, mass=0.6, *, logg=None, radius=None)
WhiteDwarfModel.passband(coarse, name)
WhiteDwarfModel.teff_at_age(mass, age_gyr)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/whitedwarf.py)

### `spbin.fit_whitedwarf_companion`

```python
fit_whitedwarf_companion(sed, *, model=None, stellar=None, companions=True, free_radius=False, companion_age_gyr=5.0, companion_feh=0.0, extinction=0.0, extinction_prior=None, dust_prior=None, parallax=None, fit_parallax=False, blue=None, galex=None, use_spherex=False, whitedwarf_prior=None, companion_prior=None, hypotheses=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/whitedwarf.py)

### `spbin.whitedwarf_light_limit`

```python
whitedwarf_light_limit(sed, *, masses=(0.45, 0.6, 0.8, 1.0, 1.2), temperatures=None, cooling_ages_gyr=None, delta=9.0, luminous_mass=None, companion_age_gyr=5.0, companion_feh=0.0, galex_upper_limits=None, **kwargs)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/whitedwarf.py)

### `spbin.loglike_whitedwarf_sed`

```python
loglike_whitedwarf_sed(sed, *, teff, mass, primary_mass=None, age_gyr=5.0, feh=0.0, parallax_mas=None, extinction=0.0, model=None, stellar=None, blue=None, galex=None, use_spherex=False)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/whitedwarf.py)

### `spbin.loglike_sed`

```python
loglike_sed(sed, *, m1, q=0.0, age_gyr=5.0, feh=0.0, parallax_mas=None, model=None, use_wise=False, extinction=0.0)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/likelihood.py)

### `spbin.locate`

```python
locate(sed, *, extinction=0.0, dust_prior=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/hrd.py)

### `spbin.fit_star`

```python
fit_star(sed, *, region=None, labels=None, extinction=0.0, extinction_prior=None, dust_prior=None, fit_parallax=False, age_gyr=None, feh=0.0, tie=1.0, options=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/hrd.py)

### `spbin.EdenhoferPrior`

```python
EdenhoferPrior(query=None, *, sigma=None)
EdenhoferPrior.moments(ra, dec, distance_pc)
EdenhoferPrior.penalty(extinction, mean, sigma)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/extinction.py)

### `spbin.plot`

```python
plot(sed, result=None, path=None, *, axes=None, components=True, title=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/plot.py)

### `spbin.download`

```python
download(source_id=None, *, ra=None, dec=None, radius_arcsec=2.0, cache_dir='data', refresh=False)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/fetch.py)

### `spbin.query_gaia`

```python
query_gaia(query, *, cache_dir='data/gaia-query', tap='esa', maxrec=1000000, wait=600, timeout=(30, 180))
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/gaia.py)

### `spbin.download_gaia`

```python
download_gaia(source_ids, *, products=('XP_CONTINUOUS',), cache_dir='data/gaia-products', data_release='Gaia DR3', batch_size=100, workers=2, data_structure='RAW', datalink_url=DATALINK, timeout=(30, 300))
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/gaia.py)

### `spbin.download_spherex`

```python
download_spherex(sed, *, cache_dir='data', refresh=False, radius_arcmin=0.8)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/spherex.py)

### `spbin.load_spherex`

```python
load_spherex(sed, path, *, method=None)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/spherex.py)

### `spbin.orbit`

```python
amrf(q, beta)
amrf_observed(a0_mas, parallax_mas, period_day, m1)
dark_mass_ratio(a_obs)
locus(m1, age_gyr=5.0, feh=0.0, model=None)
branch_mass_ratio(kind, a_obs, m1, age_gyr=5.0, feh=0.0, model=None, near=None)
solve_amrf(a_obs, m1, age_gyr=5.0, feh=0.0, model=None)
solve_orbit(a0_mas, parallax_mas, period_day, m1, age_gyr=5.0, feh=0.0, model=None)
rank_roots(sed, roots, *, parallax_mas=None, parallax_error_mas=None, model=None, **fit_kwargs)
relative_semimajor_mas(m_tot, period_day, parallax_mas)
photocentre_a0(m1, m2, beta, period_day, parallax_mas)
thiele_innes(a0, inclination_deg, omega_deg, node_deg)
campbell(a, b, f, g)
solve_luminous_pair(a0_mas, parallax_mas, period_day, m2, beta, *, errors=None)
mass_from_rv(k2_kms, period_day, eccentricity, inclination_deg, m2)
branch_from_rv(a0_mas, parallax_mas, period_day, m2, k2_kms, eccentricity, inclination_deg, beta_sed=None, beta_sed_error=None)
solve_dark_companion(a0_mas, parallax_mas, period_day, m1, beta=0.0)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/orbit.py)

### `spbin.subdwarf`

```python
blue_xp(sed, cache_dir='data')
galex(sed, cache_dir='data', radius_arcsec=3.0)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/subdwarf.py)

### `spbin.extinction`

```python
extinction_curve(wavelength_um)
transmission(wavelength_um, extinction)
```

[Source / 源码](https://github.com/jiadonglee/spbin/blob/main/src/spbin/extinction.py)
