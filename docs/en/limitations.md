# Coverage and limitations

[中文](../zh/limitations.md) · [GitHub](https://github.com/jiadonglee/spbin)


| Route | Main support | Key condition |
| --- | --- | --- |
| Empirical stellar network | 2800–7498 K; age 0.5–10 Gyr; [M/H] −1 to +0.5 | Main-sequence dwarfs; mass support varies with age and metallicity |
| Sparse ultracool extension | Parts of the 2313–2929 K label domain | Numerical support with unvalidated accuracy |
| Hot main-sequence stars | 7000–30000 K; log g 3–4.75; [M/H] ±0.3 | `hot=True` fits only XP and J/H/Ks on its shared mask |
| Giant template | 3600–6800 K; log g 0–3.8; [M/H] −2.6 to +0.6 | Label priors and a hot main-sequence companion grid |
| Hot subdwarfs | 20–45 kK; log g 5–6.5 | He-rich tiers start at 32 kK |
| DA white dwarfs | 6–80 kK; log g 7–9.5 | Atmosphere/cooling support must overlap; weaker empirical evidence above 40 kK |

Fits are local optima. `converged=True` reports optimizer convergence; model support identifies computable parameters. Neither alone establishes physical accuracy.

Measurement errors enter diagonally; correlated low-rank model errors and their determinant are retained. Full Gaia XP measurement covariance is not directly used. W1/W2 are held out of base fits by default. SPHEREx is excluded for Gaia G<9. Warm-star SPHEREx predictions lack matching warm-star training spectra, with particular caution above about 6400 K.

Extinction is native ZGR23 E rather than E(B−V) or A_V. The curve shape is fixed and broadband attenuation uses nominal-wavelength approximations. The dust map also uses Gaia XP information, so the prior and spectra are not fully independent. Temperature/extinction degeneracy affects hot stars; age/companion-light degeneracy affects warm primaries.

`delta` signs and baselines vary by API. These statistics are not binary probabilities, Bayes factors or universally calibrated significance measures. WD light envelopes lack calibrated confidence coverage. Giant thresholds depend on control samples and fit settings. Objectives with different masks or route definitions cannot be ranked directly.

SPHEREx aperture extraction lacks encircled-energy correction, per-exposure motion tracking and neighbour deblending; crowded or high-proper-motion targets need further inspection. DR4 service compatibility has not been validated. Base `fit` does not provide general giant, brown-dwarf, triple or blended-source models.

