# Visual identity

The sedkit logo uses a blue primary, a coral-orange secondary and one
shared spectral curve. The lowercase wordmark is charcoal on white.

The plotting palette uses blue `#024397`, coral orange `#f9654e`,
charcoal `#1e2834` and neutral slate observations. Single-star fits are
orange; coeval-binary fits are blue. Line styles also distinguish the models.

## Homepage assets

- [Logo](assets/sedkit-logo.png).
- [SED example](assets/sed-example.png), also available as [PDF](assets/sed-example.pdf).
- [Extinction and parallax priors](assets/extinction-parallax-priors.png),
  also available as [PDF](assets/extinction-parallax-priors.pdf).

Recreate the scientific figures from the repository root:

```bash
python scripts/plot_readme.py
```

The [plotting script](../scripts/plot_readme.py) reads the saved observations,
model fluxes, masks and prior moments for Gaia DR3 `858860697467058688` in
[the real-map example](../examples/results/extinction_20261003).
It requires no new fitting, catalogue acquisition or dust-map loading.
The prior curves are input constraints evaluated at the fitted distances,
not posterior uncertainty distributions. See [the extinction method](extinction.md).
