"""Render homepage figures from the saved real-map SB2 fit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sedkit import SED, EdenhoferPrior, plot
from sedkit.plot import BINARY, SINGLE, INK, INK2, PAPER_STYLE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/results/extinction_20261003"
OUTPUT = ROOT / "docs/assets"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sed = SED.load(SOURCE / "sed.npz")
    summary = json.loads((SOURCE / "summary.json").read_text())
    result = {"source_id": sed.source_id}
    with np.load(SOURCE / "fits.npz", allow_pickle=False) as arrays:
        for kind in ("single", "binary"):
            result[kind] = dict(summary["dust"][kind], kind=kind)
            for key in ("flux", "components", "mask"):
                result[kind][key] = arrays[f"dust_{kind}_{key}"]

    fig = plot(sed, result, title="Gaia DR3 " + sed.source_id + " | XP + 2MASS")
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"sed-example.{extension}", dpi=240,
                    bbox_inches="tight", pad_inches=.08)
    plt.close(fig)

    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25), layout="constrained")
        z_grid = np.linspace(-3.3, 3.3, 300)
        axes[0].plot(z_grid, np.exp(-.5 * z_grid**2) / np.sqrt(2*np.pi),
                     color=INK2, lw=1.5, label="Catalogue Gaussian")
        for bound in (-3, 3):
            axes[0].axvline(bound, color=INK2, alpha=.3, ls=":", lw=.8)
        for kind, color in (("single", SINGLE), ("binary", BINARY)):
            fitted = result[kind]
            z = (fitted["parallax_mas"] - sed.parallax_mas) / sed.parallax_error_mas
            axes[0].axvline(z, color=color, lw=1.8, label=f"{kind} best fit")
            mean, sigma = fitted["dust_prior_mean"], fitted["dust_prior_sigma"]
            e_grid = np.linspace(max(0, mean - 4*sigma),
                                 max(mean + 4*sigma, fitted["extinction_e"] + sigma), 300)
            pdf = np.exp(-.5 * np.array([EdenhoferPrior.penalty(e, mean, sigma)
                                        for e in e_grid])) / np.sqrt(2*np.pi)
            distance = 1000 / fitted["parallax_mas"]
            axes[1].plot(e_grid, pdf, color=color, lw=1.8,
                         label=f"{kind} distance: {distance:.2f} pc")
            axes[1].axvline(fitted["extinction_e"], color=color, ls="--", lw=1.6)
        axes[0].set(xlabel="Parallax shift / catalogue uncertainty", ylabel="Prior density", ylim=(0, None))
        axes[1].set(xlabel="E (ZGR23 units)", ylabel="Prior density", ylim=(0, None))
        for ax, title in zip(axes, ("Parallax constraint", "Dust prior at fitted distances")):
            ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
            ax.legend(fontsize=8, loc="upper left" if ax is axes[0] else "upper right")
        for extension in ("png", "pdf"):
            fig.savefig(OUTPUT / f"extinction-parallax-priors.{extension}", dpi=240,
                        bbox_inches="tight", pad_inches=.08)
        plt.close(fig)
    print("Homepage figures saved to", OUTPUT)


if __name__ == "__main__":
    main()
