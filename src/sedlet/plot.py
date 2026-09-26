"""SED plots in observed absolute flux, with residuals and components."""

import numpy as np

from .model import StellarModel


def plot(sed, result=None, *, path=None):
    """Return a Matplotlib Figure; optionally save PNG/PDF to path.

    Residuals use measurement errors, not the larger fitted model errors.
    Open symbols mark available measurements excluded from the fit.
    """
    import matplotlib.pyplot as plt

    model = StellarModel()
    wave = model.wavelength_um
    fig, (ax, residual) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    available = np.isfinite(sed.flux) & np.isfinite(sed.error) & (sed.error > 0)
    fitted = sed.fit_mask() if result is None else (
        result["binary"]["mask"] if "binary" in result else result["mask"])
    for segment, label, marker, color in (
            (slice(0, 61), "Gaia XP", ".", "#343a40"),
            (slice(61, 66), "2MASS / WISE", "s", "#a15d00"),
            (slice(66, 168), "SPHEREx", "o", "#27827e")):
        index = np.arange(168)[segment]
        index = index[available[index]]
        if not len(index):
            continue
        used = index[fitted[index]]
        held = index[~fitted[index]]
        ax.errorbar(wave[used], wave[used] * sed.flux[used],
                    yerr=wave[used] * sed.error[used], fmt=marker, ms=4,
                    color=color, alpha=0.8, label=label)
        if len(held):
            ax.errorbar(wave[held], wave[held] * sed.flux[held],
                        yerr=wave[held] * sed.error[held], fmt=marker, ms=5,
                        color=color, markerfacecolor="none", alpha=0.5,
                        label=label + " (excluded)")
    if result is not None:
        fits = [result["single"], result["binary"]] if "binary" in result else [result]
        for fit_result in fits:
            color = "#d87528" if fit_result["kind"] == "single" else "#3166b5"
            for start, stop in ((0, 61), (66, 168)):
                ax.plot(wave[start:stop], wave[start:stop] * fit_result["flux"][start:stop],
                        color=color, label=fit_result["kind"] if start == 0 else None)
            ax.scatter(wave[61:66], wave[61:66] * fit_result["flux"][61:66],
                       marker="_", s=65, color=color)
            residual.plot(wave[fitted],
                          ((sed.flux - fit_result["flux"]) / sed.error)[fitted],
                          ".", ms=3, color=color)
        chosen = fits[-1]
        if chosen["kind"] == "binary":
            for component, linestyle, label in zip(chosen["components"], ("--", ":"), ("primary", "secondary")):
                for start, stop in ((0, 61), (66, 168)):
                    ax.plot(wave[start:stop], wave[start:stop] * component[start:stop],
                            linestyle, color="#3166b5", alpha=0.6,
                            label=label if start == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\lambda F_\lambda$ [$10^{-15}$ W m$^{-2}$]")
    ax.set_title("Gaia DR3 " + sed.source_id if sed.source_id else "Stellar SED")
    ax.legend(fontsize=8, ncol=3)
    residual.axhline(0, color="0.5", lw=0.7)
    residual.set_ylabel("Residual / error")
    residual.set_xlabel("Wavelength [µm]")
    if path is not None:
        fig.savefig(path, dpi=160)
    return fig
