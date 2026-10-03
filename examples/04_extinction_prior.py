"""Compare zero-extinction and Edenhofer-prior fits of a Gaia DR3 SB2."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sedkit import EdenhoferPrior, StellarModel, download, fit, plot


def main(cache_dir="data", output_dir="data/extinction_example", map_fname=None):
    """Save fits and an absolute-flux comparison; reuse existing observations."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sed = download("858860697467058688", cache_dir=cache_dir)
    before = sed.flux.copy(), sed.error.copy(), sed.mask.copy()
    model = StellarModel()
    if map_fname is None:
        prior = EdenhoferPrior()
    else:
        from dustmaps.edenhofer2023 import Edenhofer2023Query

        prior = EdenhoferPrior(Edenhofer2023Query(
            map_fname=map_fname, integrated=True, load_samples=True))
    results = {
        "zero": fit(sed, fit_parallax=True, model=model),
        "dust": fit(sed, fit_parallax=True, model=model, extinction=None, dust_prior=prior),
    }
    for old, new in zip(before, (sed.flux, sed.error, sed.mask)):
        np.testing.assert_array_equal(old, new)
    for result in results.values():
        np.testing.assert_array_equal(result["single"]["mask"], result["binary"]["mask"])

    scalars = ("m1", "m2", "q", "age_gyr", "feh", "parallax_mas",
               "extinction_e", "dust_prior_mean", "dust_prior_sigma",
               "dust_prior_penalty", "objective", "m2lnl", "chi2",
               "n_fit", "converged", "at_bounds", "route")
    summary = {"source_id": sed.source_id, "extinction_unit": "ZGR23 E",
               "q_rv": 49.908 / 57.503, "catalogue_parallax_mas": sed.parallax_mas,
               "ra_deg": sed.metadata["ra"], "dec_deg": sed.metadata["dec"]}
    arrays = {}
    for name, result in results.items():
        summary[name] = {kind: {key: result[kind][key] for key in scalars}
                         for kind in ("single", "binary")}
        summary[name]["delta"] = result["delta"]
        for kind in ("single", "binary"):
            for key in ("flux", "components", "mask"):
                arrays[f"{name}_{kind}_{key}"] = result[kind][key]
    np.savez_compressed(output / "fits.npz", **arrays)
    sed.save(output / "sed.npz")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]}, layout="constrained")
    for column, (name, title) in enumerate((
            ("zero", "Fixed E = 0"), ("dust", "Edenhofer 3D prior"))):
        result = results[name]
        binary = result["binary"]
        plot(sed, result, axes=axes[:, column],
             title=f"{title}: q = {binary['q']:.3f}, E = {binary['extinction_e']:.4f}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=9)
    fig.savefig(output / "sed_comparison.png", dpi=180)
    fig.savefig(output / "sed_comparison.pdf")
    plt.close(fig)

    mean, sigma = prior.moments(sed.metadata["ra"], sed.metadata["dec"],
                                1000 / sed.parallax_mas)
    high = max(mean + 4*sigma, *(results["dust"][kind]["extinction_e"] + 2*sigma
                               for kind in ("single", "binary")))
    e_grid = np.linspace(0, high, 300)
    pdf = np.exp(-.5 * np.array([prior.penalty(e, mean, sigma) for e in e_grid])) / np.sqrt(2*np.pi)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), layout="constrained")
    ax.plot(e_grid, pdf, color="black", label="Map prior at catalogue distance")
    for kind, color in (("single", "#eb6834"), ("binary", "#2a78d6")):
        ax.axvline(results["dust"][kind]["extinction_e"], color=color, label=f"{kind} best fit")
    ax.set(xlabel="Extinction E (ZGR23 units)", ylabel="Prior density",
           title="Gaia DR3 " + sed.source_id)
    ax.legend(fontsize=8)
    fig.savefig(output / "dust_prior.png", dpi=180)
    fig.savefig(output / "dust_prior.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="data/extinction_example")
    parser.add_argument("--map", dest="map_fname", default=None,
                        help="Edenhofer posterior-sample FITS; omit for the dustmaps cache")
    main(**vars(parser.parse_args()))
