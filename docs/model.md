# Stellar model and limitations

The bundled model uses the J-CAPS v2.1 Ks-anchored empirical network and
PARSEC tables. The network is a 16-by-16 GELU MLP with optical and infrared
output heads. NumPy evaluates the stored weights and spectral calibration.
Its 168 output channels are absolute fluxes at 10 pc, in
`1e-18 W m^-2 nm^-1`.

PARSEC maps mass, age and metallicity to Teff, M_Ks and G-Ks. Tables are
interpolated in mass, log age and [M/H]. Two components share age and
metallicity. Their predicted fluxes are added before distance scaling.
The observed-scale model is `flux_10pc * (parallax_mas / 100)**2`.
No median normalization or unconstrained amplitude enters the fit.

The five broadband channels are learned catalogue-equivalent fluxes.
They are not obtained by interpolating the model spectrum at five
wavelengths. The model does not provide general synthetic photometry for
arbitrary filters or establish independent passband closure.

The likelihood includes measurement variance and the bundled fractional
model covariance, with a low-rank basis plus a diagonal term. Both binary
components share the primary's cold/warm error term and are fully
correlated. The covariance determinant is retained. Parameters are fitted
from multiple supported grid starts.

The G-band ratio uses the same PARSEC component magnitudes:
`beta_g = 10**(-0.4 * (M_G[1] - M_G[0]))`. It is model-dependent, rather
than an independent observation of resolved component light.

## Limitations

- Fits are exploratory local optima. No posterior uncertainty, calibrated
  binary probability or population inference is provided.
- Extinction is fixed to zero; use nearby, negligibly reddened targets.
- Age covers 0.5--10 Gyr, [M/H] -1--0.5. Both stars must lie inside the
  original network coverage. There is no atmosphere/BD fallback. Coverage
  of sparse ultracool training points is not a validated accuracy range.
- Old PARSEC tracks can cap the available primary mass below 1 solar mass.
  Gaps and support boundaries remain explicit, without extrapolation.
- GaiaXPy inter-channel measurement covariance is omitted; XP marginal
  errors are used. The model covariance does not replace that information.
- Variable sources, blends, triples and white dwarfs are not modeled.
  Catalogue flags are preserved but do not establish a clean binary sample.
- Fixed age/metallicity experiments are conditional on those choices.
  A matched mock checks the algorithm, not real-data model calibration.
