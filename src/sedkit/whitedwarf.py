"""DA white dwarfs, dwarf companions and conditional G-band light limits."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .extinction import extinction_curve
from .model import StellarModel, _bilinear
from .subdwarf import _planck, _galex_flux, _check_prior

DATA = Path(__file__).parent / "models/whitedwarf"
G_CGS, MSUN_G, RSUN_CM = 6.6743e-8, 1.988409870698051e33, 6.957e10


class WhiteDwarfModel:
    """Koester DA surface spectra and Bédard thick/thin-H C/O cooling tracks.

    No hot-star or subdwarf correction is applied. calibration=None uses
    raw spectra; 'bundled' or a dict supplies a WD-only correction/error.
    The bundled optical calibration follows real single-DA spectral
    labels and Gaia absolute fluxes. Thick hydrogen is the default.
    Unsupported infrared channels are NaN. Mass is in solar masses,
    radius in solar radii, temperatures in K and cooling ages in Gyr.
    """

    def __init__(self, calibration="bundled", hydrogen_layer="thick"):
        if hydrogen_layer not in ("thick", "thin"):
            raise ValueError("hydrogen_layer must be 'thick' or 'thin'")
        self.hydrogen_layer = hydrogen_layer
        self.summary = json.loads((DATA / "summary.json").read_text())
        with np.load(DATA / "whitedwarf_table.npz") as archive:
            self.table = {k: archive[k].astype(float) for k in archive.files}
        with np.load(DATA / ("cooling.npz" if hydrogen_layer == "thick" else "cooling_thin.npz")) as archive:
            self.cooling = {k: archive[k].astype(float) for k in archive.files}
        if isinstance(calibration, str):
            if calibration != "bundled":
                raise ValueError("calibration must be None, 'bundled' or a dict")
            with np.load(DATA / "calibration.npz") as archive:
                calibration = {k: archive[k] for k in archive.files}
        self.calibration = calibration
        self.wavelength_um = StellarModel().wavelength_um
        self.support = self.table["support"].astype(bool)
        self.coarse_wavelength_nm = self.table["coarse_wavelength_nm"]
        self.blue_wavelength_nm = self.table["blue_wavelength_nm"]
        uv_wave, uv = self.table["uv_curve_wavelength_nm"], self.table["uv_curve"]
        wave = self.coarse_wavelength_nm
        self.coarse_curve = np.where(wave < 392, np.interp(wave, uv_wave, uv),
                                     extinction_curve(np.maximum(wave, 392) / 1000))
        self.blue_curve = np.interp(self.blue_wavelength_nm, uv_wave, uv)
        self.passbands, self.galex_pivot_nm = {}, {}
        for name in ("G", "BP", "RP", "FUV", "NUV"):
            lam, trans = self.table[f"filter/{name}_wave_nm"], self.table[f"filter/{name}_trans"]
            weight = np.interp(wave, lam, trans, left=0, right=0) * wave
            self.passbands[name] = weight / weight.sum()
            if name in ("FUV", "NUV"):
                self.galex_pivot_nm[name] = float(np.sqrt(np.trapezoid(trans * lam, lam)
                                                        / np.trapezoid(trans / lam, lam)))

    def physical(self, teff, mass):
        """Cooling-model radius, gravity and age; no extrapolation."""
        c = self.cooling
        if not (np.isfinite(teff) and np.isfinite(mass) and c["teff_ax"][0] <= teff <= c["teff_ax"][-1]
                and c["mass_ax"][0] <= mass <= c["mass_ax"][-1]):
            raise ValueError("temperature/mass outside the cooling table")
        def interp(key):
            return float(_bilinear(c["teff_ax"], c["mass_ax"], c[key], np.array([teff]), np.array([mass]))[0])
        radius, age = interp("radius"), interp("age_gyr")
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(age):
            raise ValueError("cooling sequence does not reach this temperature/mass")
        logg = float(np.log10(G_CGS * mass * MSUN_G / (radius * RSUN_CM)**2))
        return dict(teff=float(teff), mass=float(mass), radius=radius, logg=logg,
                    cooling_age_gyr=age, luminosity=radius**2 * (teff / 5772)**4)

    def predict(self, teff, mass=.6, *, logg=None, radius=None):
        """Absolute 10-pc fluxes; supplying logg and radius uses the free-radius mode."""
        if (logg is None) != (radius is None):
            raise ValueError("free-radius prediction needs both logg and radius")
        if radius is None:
            physical = self.physical(teff, mass)
            logg, radius = physical["logg"], physical["radius"]
        else:
            mass = 10**logg * (radius * RSUN_CM)**2 / G_CGS / MSUN_G
            physical = dict(teff=float(teff), logg=float(logg), radius=float(radius), mass=float(mass),
                            cooling_age_gyr=None, luminosity=radius**2 * (teff / 5772)**4)
        t = self.table
        if not (t["teff_ax"][0] <= teff <= t["teff_ax"][-1] and t["logg_ax"][0] <= logg <= t["logg_ax"][-1]
                and np.isfinite(radius) and radius > 0):
            raise ValueError("outside DA support: Teff 6000--80000 K, log g 7--9.5")
        def interp(key):
            return _bilinear(t["teff_ax"], t["logg_ax"], t[key], np.array([teff]), np.array([logg]))[0]
        ln_flux, ln_blue, ln_coarse, w = interp("ln_flux"), interp("ln_blue"), interp("ln_coarse"), float(interp("balmer_w"))
        if self.calibration is not None:
            c = self.calibration
            correction = c["a"] + w * c["b"]
            blue_correction = c["blue_a"] + w * c["blue_b"]
            if "teff_knots" in c:
                knots = np.log(c["teff_knots"])
                x = np.clip(np.log(teff), knots[0], knots[-1])
                i = min(np.searchsorted(knots, x, side="right") - 1, len(knots) - 2)
                fraction = (x - knots[i]) / (knots[i + 1] - knots[i])
                correction += ((1 - fraction) * c["teff_flux"][i] + fraction * c["teff_flux"][i + 1]
                               + (logg - 8.) * c["logg_flux"])
                blue_correction += ((1 - fraction) * c["teff_blue"][i] + fraction * c["teff_blue"][i + 1]
                                    + (logg - 8.) * c["logg_blue"])
            ln_flux += correction
            ln_blue += blue_correction
            lam = np.r_[self.blue_wavelength_nm, self.wavelength_um[:61] * 1000]
            coarse_correction = np.interp(self.coarse_wavelength_nm, lam,
                                          np.r_[blue_correction, correction[:61]], left=0, right=0)
            ln_coarse += coarse_correction
        return dict(flux=np.exp(ln_flux) * radius**2, blue=np.exp(ln_blue) * radius**2,
                    coarse=np.exp(ln_coarse) * radius**2, balmer_w=w, **physical)

    def passband(self, coarse, name):
        return float(np.asarray(coarse) @ self.passbands[name])

    def teff_at_age(self, mass, age_gyr):
        """Temperature at a cooling age within this model's support."""
        temperatures, ages = [], []
        for t in self.cooling["teff_ax"]:
            try:
                p = self.physical(t, mass)
            except ValueError:
                continue
            temperatures.append(t)
            ages.append(p["cooling_age_gyr"])
        order = np.argsort(ages)
        ages, temperatures = np.array(ages)[order], np.array(temperatures)[order]
        if not len(ages) or not ages[0] <= age_gyr <= ages[-1]:
            raise ValueError("cooling age outside the supported DA temperature range")
        return float(np.interp(age_gyr, ages, temperatures))


class _Problem:
    def __init__(self, sed, model, stellar, extinction, extinction_prior, dust_prior,
                 parallax, fit_parallax, blue, galex, use_spherex, wd_prior, companion_prior):
        self.sed, self.wd, self.stellar = sed, model, stellar
        self.mask = sed.fit_mask() & model.support & stellar.support
        if not use_spherex:
            self.mask[66:] = False
        if not self.mask[:61].any():
            raise ValueError("WD fitting requires measured XP channels")
        self.idx = np.flatnonzero(self.mask)
        y, err = list(sed.flux[self.idx]), list(sed.error[self.idx])
        self.blue = blue is not None
        if blue is not None:
            bf, be = np.asarray(blue[0], float), np.asarray(blue[1], float)
            if bf.shape != (6,) or be.shape != (6,) or not np.isfinite(bf).all() or not np.all(np.isfinite(be) & (be > 0)):
                raise ValueError("blue needs six finite fluxes and positive errors")
            y.extend(bf); err.extend(be)
        self.galex = [b for b, v in (galex or {}).items() if v[2]]
        for b in self.galex:
            f, e = _galex_flux(*galex[b][:2], model.galex_pivot_nm[b], b)
            y.append(f); err.append(e)
        self.y, self.variance = np.array(y), np.array(err)**2
        if not np.isfinite(self.y).all() or not np.all(np.isfinite(self.variance) & (self.variance > 0)):
            raise ValueError("fitting data need finite fluxes and positive errors")
        self.extinction, self.extinction_prior, self.dust = extinction, extinction_prior, dust_prior
        self.parallax, self.fit_parallax = parallax, fit_parallax
        self.priors = (wd_prior, companion_prior)
        self.galex_upper_limits = {}
        self.curve = extinction_curve(model.wavelength_um)

    def extras(self, coarse, blue):
        values = list(blue) if self.blue else []
        values.extend(self.wd.passband(coarse, b) for b in self.galex)
        return np.array(values)

    def dwarf(self, mass, age, feh, scale, e):
        pred = self.stellar.evaluate(mass, 0, age, feh)
        if pred is None:
            return None
        full = pred["flux_10pc"] * scale
        att = np.exp(-e * self.curve)
        columns, variance = self.stellar.error_factors(pred, scale)
        columns, variance = columns * att[:, None], variance * att**2
        coarse = np.exp(np.interp(self.wd.coarse_wavelength_nm, self.wd.wavelength_um[:61] * 1000,
                                  np.log(np.maximum(full[:61], 1e-300))))
        bb = _planck(self.wd.coarse_wavelength_nm, pred["teff"][0])
        for selection, anchor in [(self.wd.coarse_wavelength_nm < 392, 0),
                                   (self.wd.coarse_wavelength_nm > 992, 60)]:
            coarse[selection] = bb[selection] * full[anchor] / _planck(self.wd.wavelength_um[anchor] * 1000, pred["teff"][0])
        coarse *= np.exp(-e * self.wd.coarse_curve)
        blue = np.interp(self.wd.blue_wavelength_nm, self.wd.coarse_wavelength_nm, coarse)
        extra = self.extras(coarse, blue)
        cols = np.zeros((len(self.y), columns.shape[1])); cols[:len(self.idx)] = columns[self.idx]
        return dict(flux=np.r_[full[self.idx] * att[self.idx], extra], columns=cols,
                    variance=np.r_[variance[self.idx], (.5 * extra)**2], full=full * att, coarse=coarse,
                    labels=dict(teff=float(pred["teff"][0]), mass=float(mass), logg=float(pred["logg"][0]),
                                radius=float(pred["radius"][0]), age_gyr=float(age), feh=float(feh)))

    def white_dwarf(self, teff, mass, radius, logg, scale, e):
        try:
            pred = self.wd.predict(teff, mass, radius=radius, logg=logg)
        except ValueError:
            return None
        full = pred["flux"] * scale * np.exp(-e * self.curve)
        coarse = pred["coarse"] * scale * np.exp(-e * self.wd.coarse_curve)
        blue = pred["blue"] * scale * np.exp(-e * self.wd.blue_curve)
        extra = self.extras(coarse, blue)
        c = self.wd.calibration
        if self.galex and c is not None and "uv_a" in c:
            indices = [0 if b == "FUV" else 1 for b in self.galex]
            extra[-len(self.galex):] *= np.exp(c["uv_a"][indices])
        diag = np.full(168, .03) if c is None else c["diag"]
        extra_error = np.r_[np.full(6 if self.blue else 0, .05), np.full(len(self.galex), .05)]
        if c is not None and self.blue:
            extra_error[:6] = c.get("blue_diag", extra_error[:6])
        if self.galex and c is not None and "uv_diag" in c:
            extra_error[-len(self.galex):] = c["uv_diag"][indices]
        if self.galex and not 6812 <= teff <= 41430:
            extra_error[-len(self.galex):] = np.maximum(extra_error[-len(self.galex):], .5)
        columns = np.zeros((len(self.y), 0))
        if c is not None and "scale_sigma" in c:
            sigma = float(np.interp(np.log(teff), np.log(c["teff_knots"]), c["scale_sigma"]))
            columns = np.r_[full[self.idx], extra][:, None] * sigma
        return dict(flux=np.r_[full[self.idx], extra], columns=columns,
                    variance=np.r_[(full[self.idx] * diag[self.idx])**2, (extra * extra_error)**2],
                    full=full, coarse=coarse, labels={k: pred[k] for k in
                        ("teff", "mass", "radius", "logg", "cooling_age_gyr", "luminosity")})

    def likelihood(self, components):
        flux = sum(c["flux"] for c in components)
        columns = np.concatenate([c["columns"] for c in components], axis=1)
        variance = self.variance + sum(c["variance"] for c in components)
        residual = flux - self.y
        small = np.eye(columns.shape[1]) + columns.T @ (columns / variance[:, None])
        nuisance = np.linalg.solve(small, columns.T @ (residual / variance))
        chi2 = float(np.sum((residual - columns @ nuisance)**2 / variance) + nuisance @ nuisance)
        return chi2 + float(np.log(variance).sum() + np.linalg.slogdet(small)[1]), chi2

    def uv_allowed(self, wd):
        if wd is None:
            return True
        c=self.wd.calibration
        for band,cap in self.galex_upper_limits.items():
            i=0 if band=="FUV" else 1
            correction=0. if c is None or "uv_a" not in c else c["uv_a"][i]
            sigma=.15 if c is None or "uv_diag" not in c else c["uv_diag"][i]
            if not 6812<=wd["labels"]["teff"]<=41430:sigma=max(sigma,.5)
            lower=self.wd.passband(wd["coarse"],band)*np.exp(correction-3*sigma)
            if lower>cap:return False
        return True


class _Hypothesis:
    def __init__(self, problem, kind, free_radius, age, feh, fixed=None, luminous_mass=None):
        self.p, self.kind, self.free_radius = problem, kind, free_radius
        self.age, self.feh, self.fixed, self.luminous_mass = age, feh, fixed or {}, luminous_mass
        self.bounds = {}
        if kind in ("wd", "wd+dwarf"):
            self.bounds.update(t=(np.log(6000), np.log(80000)))
            self.bounds.update(dict(g=(7., 9.5), r=(np.log(.002), np.log(.06))) if free_radius else dict(m=(.2, 1.3)))
        if kind != "wd":
            self.bounds["primary"] = (np.log(.1), np.log(2.2))
            if age is None:
                self.bounds["age"] = (np.log(.5), np.log(10.))
            if feh is None:
                self.bounds["feh"] = (-1., .5)
        if problem.extinction is None:
            self.bounds["e"] = (0., 2.)
        if problem.fit_parallax:
            self.bounds["z"] = (max(-3., -problem.parallax[0] / problem.parallax[1] + 1e-5), 3.)
        self.names = [k for k in self.bounds if k not in self.fixed]

    def unpack(self, x):
        return dict(self.fixed, **dict(zip(self.names, x)))

    def components(self, v):
        p = self.p
        e, z = v.get("e", p.extinction), v.get("z", 0.)
        parallax = p.parallax[0] + z * p.parallax[1] if p.fit_parallax else p.parallax[0]
        scale = (parallax / 100)**2
        components, wd, dwarf = [], None, None
        if self.kind in ("wd", "wd+dwarf"):
            wd = p.white_dwarf(float(np.clip(np.exp(v["t"]), 6000, 80000)), v.get("m", .6), np.exp(v["r"]) if self.free_radius else None,
                               v.get("g") if self.free_radius else None, scale, e)
            if wd is None:
                return None
            components.append(wd)
        if self.kind != "wd":
            age, feh = float(np.clip(np.exp(v["age"]), .5, 10.)) if self.age is None else self.age, v.get("feh", self.feh)
            dwarf = p.dwarf(np.exp(v["primary"]), age, feh, scale, e)
            if dwarf is None:
                return None
            components.append(dwarf)
            if self.kind == "ms+ms":
                second = p.dwarf(self.luminous_mass, age, feh, scale, e)
                if second is None:
                    return None
                components.append(second)
        penalty = z*z
        if p.extinction_prior is not None:
            penalty += ((e - p.extinction_prior[0]) / p.extinction_prior[1])**2
        if p.dust is not None:
            mean, sigma = p.dust.moments(p.sed.metadata["ra"], p.sed.metadata["dec"], 1000 / parallax)
            penalty += p.dust.penalty(e, mean, sigma)
        for comp, priors in zip((wd, dwarf), p.priors):
            if comp is not None:
                penalty += sum(((comp["labels"][k] - mean) / sigma)**2 for k, (mean, sigma) in priors.items())
        return components, wd, dwarf, float(e), float(parallax), float(penalty)

    def objective(self, x):
        result = self.components(self.unpack(x))
        if result is None:
            return 1e30
        if not self.p.uv_allowed(result[1]):
            return 1e30
        return self.p.likelihood(result[0])[0] + result[-1]

    def fit(self, initial=None):
        base = dict(t=np.log(15000.), m=.6, g=8., r=np.log(.013), primary=np.log(.8), age=np.log(5.),
                    feh=0., e=self.p.extinction_prior[0] if self.p.extinction_prior is not None else .01, z=0.)
        if self.p.dust is not None:
            base["e"] = self.p.dust.moments(self.p.sed.metadata["ra"], self.p.sed.metadata["dec"],
                                            1000 / self.p.parallax[0])[0]
        if "feh" in self.p.priors[1]:
            base["feh"] = np.clip(self.p.priors[1]["feh"][0], -1., .5)
        temperatures = [8000, 11000, 15000, 22000, 35000, 60000] if "t" in self.names else [15000]
        masses = [.2, .4, .7, 1., 1.3] if "primary" in self.names else [.8]
        ages = [5.]
        if self.kind == "dwarf" and self.age is None:
            masses = np.linspace(.1, 2.2, 22)
            ages = [1., 2., 5., 8., 9.8]
        seeds = []
        if initial is not None:
            seeds.extend({**base, **initial, "t": np.log(t)} for t in temperatures)
        for t in temperatures:
            for mass in masses:
                for age in ages:
                    v = dict(base, t=np.log(t), primary=np.log(mass), age=np.log(age))
                    if self.free_radius and "t" in self.names:
                        wd = self.p.white_dwarf(t, .6, .013, 8., 1, 0)
                        if wd is not None:
                            ratio = np.median(self.p.y[:len(self.p.idx)] / wd["flux"][:len(self.p.idx)])
                            if ratio > 0:
                                v["r"] = np.clip(np.log(.013 * np.sqrt(ratio) / (self.p.parallax[0] / 100)),
                                                 *self.bounds["r"])
                    seeds.append(v)
        if self.p.galex_upper_limits and "e" in self.names:
            seeds.extend([{**v,"e":e} for v in list(seeds) for e in [.15,.3,.6]])
        ranked = sorted(((self.objective([v[k] for k in self.names]), v) for v in seeds), key=lambda item: item[0])
        results = []
        steps = dict(t=.08, m=.05, g=.1, r=.05, primary=.05, age=.2, feh=.1, e=.02, z=.2)
        for score, v in ranked[:3]:
            if score >= 1e29:
                continue
            x0 = np.array([v[k] for k in self.names])
            simplex = np.vstack([x0, x0 + np.diag([steps[k] for k in self.names])])
            result = minimize(self.objective, x0, method="Nelder-Mead",
                              bounds=[self.bounds[k] for k in self.names],
                              options=dict(initial_simplex=simplex, maxfev=4000, xatol=1e-5, fatol=1e-4))
            results.append(result)
        if not results:
            return dict(objective=np.inf, converged=False, hypothesis=self.kind), None
        best = min(results, key=lambda r: r.fun)
        v = self.unpack(best.x)
        comps, wd, dwarf, e, plx, penalty = self.components(v)
        value, chi2 = self.p.likelihood(comps)
        summary = dict(hypothesis=self.kind, objective=float(best.fun), minus2lnL=value, chi2=chi2,
                       converged=bool(best.success), extinction_e=e, parallax_mas=plx, penalty=penalty,
                       model=sum(c["flux"] for c in comps), n_fit=len(self.p.y),
                       components={"whitedwarf": None if wd is None else wd["full"],
                                   "dwarf": None if dwarf is None else dwarf["full"]})
        if wd is not None:
            summary["whitedwarf"] = wd["labels"]
            total = sum(c["coarse"] for c in comps)
            summary["fractions"] = {f"beta_{b}": self.p.wd.passband(wd["coarse"], b) / self.p.wd.passband(total, b)
                                    for b in ("G", "BP", "RP")}
        if dwarf is not None:
            summary["companion"] = dwarf["labels"]
        if wd is not None and dwarf is not None:
            b = wd["labels"]["mass"] / (wd["labels"]["mass"] + dwarf["labels"]["mass"])
            summary["photocentre"] = dict(star1="whitedwarf", star2="dwarf",
                mass_fraction_star1=b, light_fraction_star1=summary["fractions"]["beta_G"],
                coefficient=b-summary["fractions"]["beta_G"])
        return summary, v


def _prepare(sed, model, stellar, extinction, extinction_prior, dust_prior, parallax, fit_parallax,
             blue, galex, use_spherex, whitedwarf_prior, companion_prior):
    model, stellar = WhiteDwarfModel() if model is None else model, StellarModel() if stellar is None else stellar
    parallax = (sed.parallax_mas, sed.parallax_error_mas) if parallax is None else parallax
    if not np.isfinite(parallax[0]) or parallax[0] <= 0 or (fit_parallax and not (np.isfinite(parallax[1]) and parallax[1] > 0)):
        raise ValueError("positive parallax and, when fitted, positive parallax error are required")
    if extinction is not None:
        if not np.isfinite(extinction) or extinction < 0 or extinction_prior is not None or dust_prior is not None:
            raise ValueError("fixed nonnegative extinction takes no extinction prior")
    elif (extinction_prior is None) == (dust_prior is None):
        raise ValueError("fitted extinction needs exactly one Gaussian or dust prior")
    if extinction_prior is not None:
        extinction_prior = _check_prior("extinction_prior", {"e": extinction_prior})["e"]
    for name, priors, keys in [("whitedwarf_prior", whitedwarf_prior, {"teff", "mass", "logg", "radius"}),
                               ("companion_prior", companion_prior, {"teff", "mass", "logg", "radius", "age_gyr", "feh"})]:
        if priors and set(priors) - keys:
            raise ValueError(f"unsupported labels in {name}: {sorted(set(priors) - keys)}")
    return _Problem(sed, model, stellar, extinction, extinction_prior, dust_prior, parallax, fit_parallax,
                    blue, galex, use_spherex, _check_prior("whitedwarf_prior", whitedwarf_prior),
                    _check_prior("companion_prior", companion_prior))


def fit_whitedwarf_companion(sed, *, model=None, stellar=None, companions=True, free_radius=False,
                            companion_age_gyr=5., companion_feh=0., extinction=0., extinction_prior=None,
                            dust_prior=None, parallax=None, fit_parallax=False, blue=None, galex=None,
                            use_spherex=False, whitedwarf_prior=None, companion_prior=None):
    """Compare one dwarf, one DA WD and DA+dwarf on identical data channels.

    WD parameters are Teff and mass, with radius from thick-H C/O cooling
    tracks. free_radius=True instead fits Teff, log g and radius. Dwarf
    age/metallicity are fixed unless None. Foreground extinction and
    parallax are shared; parallax can move within three catalogue sigma.
    beta_G is the reddened WD fraction of the total G light, not F2/F1.
    Delta objectives are diagnostics, not classification probabilities.
    """
    p = _prepare(sed, model, stellar, extinction, extinction_prior, dust_prior, parallax, fit_parallax,
                 blue, galex, use_spherex, whitedwarf_prior, companion_prior)
    kinds = ("dwarf", "wd", "wd+dwarf") if companions else ("dwarf", "wd")
    results = {}
    primary_seed = None
    for kind in kinds:
        results[kind], values = _Hypothesis(p, kind, free_radius, companion_age_gyr, companion_feh).fit(
            initial=primary_seed if kind == "wd+dwarf" else None)
        if kind == "dwarf":
            primary_seed = values
    best = min(results, key=lambda k: results[k]["objective"])
    if not np.isfinite(results[best]["objective"]):
        raise ValueError("no supported hypothesis could be fitted")
    return dict(hypotheses=results, preferred=best,
                detection_statistic=(min(results[k]["objective"] for k in ("dwarf","wd"))
                                     - results["wd+dwarf"]["objective"]) if companions else None,
                delta={k: r["objective"] - results[best]["objective"] for k, r in results.items()},
                mask=p.mask, n_fit=len(p.y), source_id=sed.source_id,
                data=dict(y=p.y, error=np.sqrt(p.variance)), free_radius=free_radius,
                hydrogen_layer=p.wd.hydrogen_layer if not free_radius else None)


def whitedwarf_light_limit(sed, *, masses=(.45, .6, .8, 1., 1.2), temperatures=None,
                          cooling_ages_gyr=None, delta=9., luminous_mass=None,
                          companion_age_gyr=5., companion_feh=0., galex_upper_limits=None, **kwargs):
    """Conditional WD beta_G envelope after profiling the luminous primary.

    Scans WD temperature or cooling age at each supplied WD mass, refitting
    primary mass and shared nuisance parameters. Returned allowed rows
    satisfy objective-minimum <= delta; disconnected intervals are retained.
    delta is an operational profile threshold, without confidence coverage
    calibration. The envelope is conditional on mass/temperature support.
    luminous_mass additionally compares a coeval MS companion of that mass.
    galex_upper_limits maps FUV/NUV to observed total-flux caps in SED
    units. The WD's conservative lower UV prediction must fit below each
    cap; this needs no primary-star UV template.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive")
    if temperatures is not None and cooling_ages_gyr is not None:
        raise ValueError("scan temperature or cooling age, not both")
    options = dict(model=None, stellar=None, extinction=0., extinction_prior=None, dust_prior=None,
                   parallax=None, fit_parallax=False, blue=None, galex=None, use_spherex=False,
                   whitedwarf_prior=None, companion_prior=None)
    options.update(kwargs)
    p = _prepare(sed, **options)
    for band,cap in (galex_upper_limits or {}).items():
        if band not in ("FUV","NUV") or not np.isfinite(cap) or cap<=0:
            raise ValueError("GALEX upper limits need FUV/NUV names and positive finite fluxes")
    p.galex_upper_limits=dict(galex_upper_limits or {})
    single, seed = _Hypothesis(p, "dwarf", False, companion_age_gyr, companion_feh).fit()
    profile = []
    def profile_row(t,mass,fit):
        row=dict(teff=float(t),mass=float(mass),objective=fit["objective"],converged=fit["converged"])
        if np.isfinite(fit["objective"]):
            row.update(beta_G=fit["fractions"]["beta_G"],cooling_age_gyr=fit["whitedwarf"]["cooling_age_gyr"],
                       extinction_e=fit["extinction_e"],parallax_mas=fit["parallax_mas"])
        else:
            row.update(beta_G=np.nan,cooling_age_gyr=None)
        return row
    masses = np.unique(np.atleast_1d(masses).astype(float))
    if not len(masses) or not np.all(np.isfinite(masses) & (masses >= .2) & (masses <= 1.3)):
        raise ValueError("scan masses must be within 0.2--1.3 solar masses")
    if temperatures is not None:
        temperatures = np.atleast_1d(temperatures).astype(float)
        if not len(temperatures) or not np.all(np.isfinite(temperatures) & (temperatures >= 6000) & (temperatures <= 80000)):
            raise ValueError("scan temperatures must be within 6000--80000 K")
    for mass in masses:
        grid = p.wd.table["teff_ax"] if temperatures is None else np.asarray(temperatures, float)
        if cooling_ages_gyr is not None:
            grid = np.array([p.wd.teff_at_age(mass, a) for a in cooling_ages_gyr])
        if p.galex_upper_limits and cooling_ages_gyr is None:
            # Resolve both sides of the empirical UV-error domain as well
            # as optical/UV constraint crossings on the supplied grid.
            edges=np.array([6811.,6813.,41429.,41431.])
            grid=np.r_[grid,edges[(edges>=np.min(grid)) & (edges<=np.max(grid))]]
        current = seed
        for t in np.sort(np.unique(grid)):
            hyp = _Hypothesis(p, "wd+dwarf", False, companion_age_gyr, companion_feh,
                              fixed=dict(t=np.log(t), m=mass))
            fit, current = hyp.fit(initial=current)
            profile.append(profile_row(t,mass,fit))
    best = min([single["objective"]] + [r["objective"] for r in profile])
    # Resolve crossings of the allowed-set boundary instead of reporting
    # an upper limit at an arbitrary temperature-grid node.
    if cooling_ages_gyr is None:
        for mass in masses:
            rows = [r for r in profile if r["mass"] == mass]
            for left, right in zip(rows[:-1], rows[1:]):
                if (left["objective"] - best <= delta) == (right["objective"] - best <= delta):
                    continue
                lo, hi = left, right
                current = seed
                for _ in range(8):
                    t = .5 * (lo["teff"] + hi["teff"])
                    hyp = _Hypothesis(p, "wd+dwarf", False, companion_age_gyr, companion_feh,
                                      fixed=dict(t=np.log(t), m=mass))
                    fit, current = hyp.fit(initial=current)
                    row=profile_row(t,mass,fit)
                    profile.append(row)
                    if (row["objective"] - best <= delta) == (lo["objective"] - best <= delta):
                        lo = row
                    else:
                        hi = row
        profile.sort(key=lambda r: (r["mass"], r["teff"]))
        best = min([single["objective"]] + [r["objective"] for r in profile])
    intervals = []
    for mass in masses:
        rows = [r for r in profile if r["mass"] == mass]
        for r in rows:
            r["delta"] = r["objective"] - best
            r["allowed"] = r["delta"] <= delta
        start = None
        for i, r in enumerate(rows):
            if r["allowed"] and start is None:
                start = i
            if start is not None and (not r["allowed"] or i == len(rows) - 1):
                end = i if r["allowed"] else i - 1
                intervals.append(dict(mass=float(mass), teff=[rows[start]["teff"], rows[end]["teff"]],
                                      beta_G=[min(a["beta_G"] for a in rows[start:end+1]),
                                              max(a["beta_G"] for a in rows[start:end+1])]))
                start = None
    allowed = [r for r in profile if r["allowed"]]
    upper = max((r["beta_G"] for r in allowed), default=np.nan)
    luminous = None
    if luminous_mass is not None:
        luminous = _Hypothesis(p, "ms+ms", False, companion_age_gyr, companion_feh,
                               luminous_mass=luminous_mass).fit(initial=seed)[0]
        luminous["delta"] = luminous["objective"] - best
    return dict(beta_G_upper=float(upper), flux_ratio_G_upper=float(upper / (1 - upper)),
                delta_threshold=float(delta), confidence_level=None, profile=profile, intervals=intervals,
                single_primary=single, luminous_companion=luminous, minimum_objective=float(best),
                mask=p.mask, source_id=sed.source_id, mass_grid=masses,
                hydrogen_layer=p.wd.hydrogen_layer,
                galex_upper_limits=p.galex_upper_limits,
                temperature_support=[float(p.wd.table["teff_ax"][0]), float(p.wd.table["teff_ax"][-1])])


def loglike_whitedwarf_sed(sed, *, teff, mass, primary_mass=None, age_gyr=5., feh=0.,
                          parallax_mas=None, extinction=0., model=None, stellar=None,
                          blue=None, galex=None, use_spherex=False):
    """DA or DA+dwarf log likelihood for composition with an orbital model.

    Masses are in solar units. The WD mass sets its cooling radius and
    can be the same parameter used by the orbit. No parameter priors or
    catalogue parallax penalty enter here. Model variance/determinants
    are retained; unsupported WD/stellar components return -inf.
    """
    if not np.isfinite(teff) or not 6000<=teff<=80000 or not np.isfinite(mass) or not .2<=mass<=1.3:
        return -np.inf
    if primary_mass is not None and (not np.isfinite(primary_mass) or primary_mass<=0):
        return -np.inf
    parallax=sed.parallax_mas if parallax_mas is None else parallax_mas
    if not np.isfinite(parallax) or parallax<=0: return -np.inf
    p=_prepare(sed,model,stellar,extinction,None,None,(parallax,0.),False,
               blue,galex,use_spherex,None,None)
    kind="wd" if primary_mass is None else "wd+dwarf"
    values=dict(t=np.log(teff),m=mass)
    if primary_mass is not None:values["primary"]=np.log(primary_mass)
    components=_Hypothesis(p,kind,False,age_gyr,feh).components(values)
    return -np.inf if components is None else -.5*p.likelihood(components[0])[0]
