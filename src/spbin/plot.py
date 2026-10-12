"""Paper-style SED and relative-flux diagnostics on the absolute flux scale."""

import numpy as np

from .model import StellarModel

SINGLE, BINARY, OBS = "#f9654e", "#024397", "#7e8a99"
INK, INK2 = "#1e2834", "#586475"
PAPER_STYLE = {
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "lines.solid_capstyle": "round",
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": False, "lines.linewidth": 1.2,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
}
BINARY_LS = (0, (4, 1.5))


def _segments(ax, wave, values, **kwargs):
    for start, stop in ((0, 61), (66, 168)):
        order = np.argsort(wave[start:stop]) + start
        ax.plot(wave[order], values[order], **kwargs)
        kwargs.pop("label", None)


def plot(sed, result=None, path=None, *, axes=None, components=True, title=None):
    """Return a paper-style Figure, optionally save it or draw into two axes.

    The upper panel is linear lambda F_lambda in physical units. The lower
    panel shows ln(F/F_single), with the binary/single model difference.
    Error bars use the unchanged measurement errors; log-panel errors are
    first-order error/flux. Open points are available but excluded bands.
    `axes=(sed_axis, ratio_axis)` supports multi-source layouts. With external
    axes the caller supplies a shared legend and saves the owning figure.
    """
    import matplotlib.pyplot as plt

    wave = StellarModel().wavelength_um
    fits = [] if result is None else (
        [result["single"], result["binary"]] if "binary" in result else [result])
    fitted = fits[-1]["mask"] if fits else sed.fit_mask()
    available = np.isfinite(sed.flux) & np.isfinite(sed.error) & (sed.error > 0)
    observed = wave * sed.flux * 1e-15
    values = observed[available]
    peak = np.max(values) if len(values) else 1e-15
    power = int(np.floor(np.log10(peak))) if peak > 0 else -15
    unit = 10.0**power
    standalone = axes is None
    with plt.rc_context(PAPER_STYLE):
        if standalone:
            fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7.087, 4.7), sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1]}, layout="constrained")
        else:
            ax, ratio = axes
            fig = ax.figure
        ref = fits[0]["flux"] if fits else None
        for segment, label, marker, size in (
                (slice(0, 61), "XP", "o", 2.1),
                (slice(61, 64), r"$JHK_s$", "s", 4.2),
                (slice(64, 66), "WISE", "s", 4.2),
                (slice(66, 168), "SPHEREx", "o", 2.1)):
            indices = np.arange(168)[segment]
            indices = indices[available[indices]]
            for keep, face, suffix in ((True, OBS, ""), (False, "white", " (excluded)")):
                index = indices[fitted[indices] == keep]
                if not len(index):
                    continue
                ax.errorbar(wave[index], observed[index] / unit,
                    yerr=wave[index] * sed.error[index] * 1e-15 / unit,
                    fmt=marker, ms=size, mfc=face, mec=INK2, mew=.45,
                    color=OBS, elinewidth=.35, lw=0, zorder=2,
                    label=label + suffix)
                if ref is not None:
                    good = index[(sed.flux[index] > 0) & (ref[index] > 0)]
                    ratio.errorbar(wave[good], np.log(sed.flux[good] / ref[good]),
                        yerr=sed.error[good] / sed.flux[good], fmt=marker,
                        ms=size, mfc=face, mec=INK2, mew=.4, color=OBS,
                        elinewidth=.3, lw=0, alpha=1 if keep else .55, zorder=2)
        for fit_result in fits:
            binary = fit_result["kind"] == "binary"
            color, ls = (BINARY, BINARY_LS) if binary else (SINGLE, "-")
            flux = wave * fit_result["flux"] * 1e-15 / unit
            _segments(ax, wave, flux, color=color, ls=ls,
                      lw=1.35, zorder=5, label=fit_result["kind"])
            ax.plot(wave[61:66], flux[61:66], "_", ms=8, mew=1.3, color=color, zorder=5)
            relative = np.log(fit_result["flux"] / ref)
            _segments(ratio, wave, relative, color=color, ls=ls, lw=1.35, zorder=5)
            ratio.plot(wave[61:66], relative[61:66], "_", ms=8, mew=1.3, color=color, zorder=5)
        if fits and fits[-1]["kind"] == "binary" and components:
            for component, ls, label, color in zip(
                    fits[-1]["components"], ("--", ":"), ("primary", "secondary"), (BINARY, SINGLE)):
                _segments(ax, wave, wave * component * 1e-15 / unit,
                          color=color, ls=ls, lw=.75, alpha=.65, label=label, zorder=1)
        if not fits:
            ratio.axhline(0, color=INK2, lw=.7)
            ratio.set_ylabel("No fitted model")
        else:
            reference = fits[0]["kind"]
            ratio.set_ylabel(r"$\ln(F/F_{\rm " + reference + r"})$")
        ax.set_ylim(min(0.0, float(np.min(values)) / unit * 1.05) if len(values) else 0, None)
        ax.set_ylabel(rf"$\lambda F_\lambda$ [$10^{{{power}}}$ W m$^{{-2}}$]")
        ax.set_title(title if title is not None else (
            "Gaia DR3 " + sed.source_id if sed.source_id else "Stellar SED"),
            loc="left", fontsize=9.5, pad=5)
        for axis in (ax, ratio):
            axis.set_xscale("log")
            axis.set_xlim(.37, 5.3)
            axis.set_xticks((.4, .7, 1, 2, 3, 5), ("0.4", "0.7", "1", "2", "3", "5"))
            axis.minorticks_off()
            axis.tick_params(labelsize=9)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        ratio.set_xlabel(r"Wavelength [$\mu$m]")
        if standalone:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncol=4,
                       fontsize=8.5, handlelength=1.8, columnspacing=1)
        if path is not None:
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.02)
    return fig
