from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from dustmaps.edenhofer2023 import Edenhofer2023Query
from sedkit import EdenhoferPrior, StellarModel, download, fit, plot

SOURCE_ID = "858860697467058688"
CACHE_DIR = Path("data")
OUTPUT_DIR = Path("results/extinction_parallax_20261003")
MAP_FNAME = None  # or "/path/to/edenhofer_2023/samples_healpix.fits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sed = download(SOURCE_ID, cache_dir=CACHE_DIR)
model = StellarModel()
observations = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
print(f"Catalogue parallax: {sed.parallax_mas:.6f} +/- {sed.parallax_error_mas:.6f} mas")
print(f"Catalogue distance: {1000 / sed.parallax_mas:.3f} pc; {sed.fit_mask().sum()} fitted channels")

results = {}
results["fixed"] = fit(sed, model=model, age_gyr=5.0, feh=0.0)
for kind in ("single", "binary"):
    assert results["fixed"][kind]["parallax_mas"] == sed.parallax_mas
print("Baseline: E=0, catalogue parallax fixed.")

query = Edenhofer2023Query(map_fname=MAP_FNAME, integrated=True, load_samples=True)
prior = EdenhoferPrior(query)
mean_e, sigma_e = prior.moments(sed.metadata["ra"], sed.metadata["dec"],
                               1000 / sed.parallax_mas)
print(f"Dust prior at catalogue distance: E = {mean_e:.6f}, width = {sigma_e:.6f}")

results["dust_fixed"] = fit(
    sed, model=model, age_gyr=5.0, feh=0.0,
    extinction=None, dust_prior=prior,
)
results["dust_joint"] = fit(
    sed, model=model, age_gyr=5.0, feh=0.0,
    extinction=None, dust_prior=prior, fit_parallax=True,
)
print("Completed fixed-distance dust fit and joint extinction/parallax fit.")

rows = []
for case, result in results.items():
    for kind in ("single", "binary"):
        r = result[kind]
        rows.append(dict(case=case, kind=kind, m1=r["m1"], q=r["q"],
                         E=r["extinction_e"], parallax_mas=r["parallax_mas"],
                         parallax_shift_sigma=(r["parallax_mas"] - sed.parallax_mas) / sed.parallax_error_mas,
                         dust_mean=r["dust_prior_mean"] if r["dust_prior_mean"] is not None else np.nan,
                         dust_width=r["dust_prior_sigma"] if r["dust_prior_sigma"] is not None else np.nan,
                         chi2=r["chi2"], objective=r["objective"],
                         converged=r["converged"], at_bounds=", ".join(r["at_bounds"])))
summary = Table(rows=rows)
for name in ("m1", "q", "E", "parallax_mas", "dust_mean", "dust_width"):
    summary[name].format = ".6f"
for name in ("parallax_shift_sigma", "chi2", "objective"):
    summary[name].format = ".3f"
summary.write(OUTPUT_DIR / "summary.csv", overwrite=True)
for case, result in results.items():
    print(f"{case:12s} single minus binary objective: {result['delta']:.3f}")
summary

titles = {"fixed": "E=0; fixed parallax", "dust_fixed": "Fitted E; fixed parallax",
          "dust_joint": "Fitted E and parallax"}
fig, axes = plt.subplots(2, 3, figsize=(15, 5.5), sharex=True,
                         gridspec_kw={"height_ratios": [2.5, 1]}, layout="constrained")
for j, (case, result) in enumerate(results.items()):
    plot(sed, result, axes=axes[:, j], title=titles[case])
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=9)
fig.savefig(OUTPUT_DIR / "sed_comparison.png", dpi=180)
fig.savefig(OUTPUT_DIR / "sed_comparison.pdf")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
z_grid = np.linspace(-3, 3, 250)
axes[0].plot(z_grid, np.exp(-0.5 * z_grid**2) / np.sqrt(2*np.pi), color="black",
             label="Catalogue Gaussian")
for kind, color in (("single", "#eb6834"), ("binary", "#2a78d6")):
    r = results["dust_joint"][kind]
    z = (r["parallax_mas"] - sed.parallax_mas) / sed.parallax_error_mas
    axes[0].axvline(z, color=color, label=f"{kind} best fit")
    mean, width = r["dust_prior_mean"], r["dust_prior_sigma"]
    e_grid = np.linspace(max(0, mean - 4*width), max(mean + 4*width, r["extinction_e"] + width), 250)
    pdf = np.exp(-0.5 * np.array([prior.penalty(e, mean, width) for e in e_grid])) / np.sqrt(2*np.pi)
    axes[1].plot(e_grid, pdf, color=color,
                 label=f"{kind} distance: {1000 / r['parallax_mas']:.2f} pc")
    axes[1].axvline(r["extinction_e"], color=color, ls="--")
axes[0].set(xlabel="Parallax shift / catalogue uncertainty", ylabel="Prior density", title="Parallax constraint")
axes[1].set(xlabel="E (ZGR23 units)", ylabel="Prior density", title="Dust prior at fitted distances")
for ax in axes:
    ax.legend(fontsize=8)
fig.savefig(OUTPUT_DIR / "priors.png", dpi=180)
fig.savefig(OUTPUT_DIR / "priors.pdf")
plt.show()

for old, current in zip(observations, (sed.flux, sed.error, sed.mask)):
    np.testing.assert_array_equal(old, current)
for result in results.values():
    np.testing.assert_array_equal(result["single"]["mask"], result["binary"]["mask"])
for kind in ("single", "binary"):
    r = results["dust_joint"][kind]
    z = (r["parallax_mas"] - sed.parallax_mas) / sed.parallax_error_mas
    np.testing.assert_allclose(r["objective"], r["m2lnl"] + z*z + r["dust_prior_penalty"])
for kind in ("single", "binary"):
    assert results["dust_fixed"][kind]["parallax_mas"] == sed.parallax_mas
b_fixed, b_joint = results["dust_fixed"]["binary"], results["dust_joint"]["binary"]
print(f"Binary q: {b_fixed['q']:.6f} (fixed parallax) -> {b_joint['q']:.6f} (joint)")
print(f"Joint binary E: {b_joint['extinction_e']:.6f}; parallax: {b_joint['parallax_mas']:.6f} mas")
print("Observation arrays unchanged; shared masks and objective decomposition verified.")
