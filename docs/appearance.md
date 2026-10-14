# Visual identity

The spbin logo uses a blue primary, a coral-orange secondary and one
shared spectral curve. The lowercase wordmark is charcoal on white.

The plotting palette uses blue `#024397`, coral orange `#f9654e`,
charcoal `#1e2834` and neutral slate observations. Single-star fits are
orange; coeval-binary fits are blue. Line styles also distinguish the models.

## Homepage assets

- [Logo](assets/spbin-logo.png).
- [HR diagram of binary classes](assets/hero-hrd.png), with a
  [dark-theme version](assets/hero-hrd-dark.png) and PDFs of both.
- [SED example](assets/sed-example.png), also available as [PDF](assets/sed-example.pdf).
- [Extinction and parallax priors](assets/extinction-parallax-priors.png),
  also available as [PDF](assets/extinction-parallax-priors.pdf).

Recreate the scientific figures from the repository root:

```bash
python scripts/plot_readme.py
python scripts/plot_hero_hrd.py
```

The [plotting script](https://github.com/jiadonglee/spbin/blob/main/scripts/plot_readme.py) reads the saved observations,
model fluxes, masks and prior moments for Gaia DR3 `858860697467058688` in
[the real-map example](https://github.com/jiadonglee/spbin/tree/main/examples/results/extinction_20261003).
It requires no new fitting, catalogue acquisition or dust-map loading.
The HR-diagram figure draws its composites from the bundled models over a
binned Gaia DR3 HR diagram (stars within 400 pc at |b| > 25 deg) stored in
`docs/assets/gaia-hrd-density.npz`.
The prior curves are input constraints evaluated at the fitted distances,
not posterior uncertainty distributions. See [the extinction method](extinction.md).
