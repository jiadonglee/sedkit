"""Benchmark the numpy and JAX fitters on synthetic stars.

Constructs SEDs from the model itself (no catalogue data needed), fits each
with both backends, and reports warm wall time, objective agreement and the
Nelder-Mead evaluation counts of the numpy path. The dust prior is a constant
mean/width stand-in, so no dust map is required.

Run on an idle machine: concurrent CPU work inflates the two backends
unequally and destroys the timing ratios.

Usage: python scripts/bench_fit.py [--stars 10] [--repeats 3] [--hot]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spbin import SED, StellarModel
from spbin.extinction import EdenhoferPrior


class FakeQuery:
    integrated = True
    n_samples = None

    def __call__(self, coords, mode="mean"):
        return 0.05 if mode == "mean" else 0.02


def make_stars(model, n, seed=0):
    """Synthetic observed pairs drawn across the dwarf parameter ranges."""
    rng = np.random.default_rng(seed)
    stars = []
    while len(stars) < n:
        pair = model.evaluate(rng.uniform(0.3, 1.6), rng.uniform(0.15, 0.95),
                              rng.uniform(1.0, 9.0), rng.uniform(-0.7, 0.3))
        if pair is None:
            continue
        flux = pair["flux_10pc"] * (100.0 / rng.uniform(30.0, 200.0))**2
        sed = SED(flux * (1 + rng.normal(0, 0.03, 168)), flux * 0.03,
                  np.ones(168, bool), float(rng.uniform(3.0, 30.0)),
                  float(rng.uniform(0.03, 0.10)))
        sed.metadata.update(ra=30.0, dec=10.0, phot_g_mean_mag=14.0)
        stars.append(sed)
    return stars


def timed(call, repeats):
    """Median warm wall time in seconds, after one compile/warm-up call."""
    call()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def count_numpy_evaluations(module, call):
    """Wall time and scipy Nelder-Mead evaluation counts of one numpy fit."""
    real = module.minimize
    counts = []

    def wrapper(function, *args, **kwargs):
        spent = [0]

        def counted(vector):
            spent[0] += 1
            return function(vector)

        result = real(counted, *args, **kwargs)
        counts.append(spent[0])
        return result

    module.minimize = wrapper
    try:
        start = time.perf_counter()
        result = call()
        wall = time.perf_counter() - start
    finally:
        module.minimize = real
    return wall, result, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stars", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hot", action="store_true",
                        help="benchmark the hot route instead of the core route")
    args = parser.parse_args()

    model = StellarModel(hot=args.hot)
    stars = make_stars(model, args.stars)
    prior = EdenhoferPrior(FakeQuery(), sigma=0.02)

    from spbin.fit import fit as numpy_fit
    from spbin._jax.fit import fit as jax_fit
    numpy_module = sys.modules["spbin.fit"]

    label = "hot_ms" if args.hot else "core"
    print(f"{len(stars)} synthetic stars, {label} route, "
          f"age/[M/H]/parallax/extinction free, kind=both")
    print(f"{'numpy s':>9} {'nm s':>7} {'bfgs+nm s':>9} {'NM evals':>9} "
          f"{'d_obj nm':>9} {'d_obj b+nm':>10}")
    totals = dict(numpy=0.0, nm=0.0, bfgs=0.0)
    worst = dict(nm=0.0, bfgs=0.0)
    for sed in stars:
        call = lambda: numpy_fit(sed, model=model, age_gyr=None, feh=None,
                                 fit_parallax=True, extinction=None, dust_prior=prior)
        wall_np, reference, counts = count_numpy_evaluations(numpy_module, call)

        wall_nm, result_nm = timed(
            lambda: jax_fit(sed, model=model, age_gyr=None, feh=None,
                            fit_parallax=True, extinction=None, dust_prior=prior,
                            optimizer="nm"), args.repeats)
        wall_bn, result_bn = timed(
            lambda: jax_fit(sed, model=model, age_gyr=None, feh=None,
                            fit_parallax=True, extinction=None, dust_prior=prior,
                            optimizer="bfgs+nm"), args.repeats)

        d_nm = max(result_nm[k]["objective"] - reference[k]["objective"]
                   for k in ("single", "binary"))
        d_bn = max(result_bn[k]["objective"] - reference[k]["objective"]
                   for k in ("single", "binary"))
        worst["nm"] = max(worst["nm"], d_nm)
        worst["bfgs"] = max(worst["bfgs"], d_bn)
        totals["numpy"] += wall_np
        totals["nm"] += wall_nm
        totals["bfgs"] += wall_bn
        print(f"{wall_np:9.2f} {wall_nm:7.2f} {wall_bn:9.2f} {sum(counts):9d} "
              f"{d_nm:9.4f} {d_bn:10.4f}")
    print(f"totals  {totals['numpy']:7.2f} {totals['nm']:7.2f} {totals['bfgs']:9.2f}   "
          f"speedup: nm {totals['numpy'] / totals['nm']:.1f}x, "
          f"bfgs+nm {totals['numpy'] / totals['bfgs']:.1f}x")
    print(f"worst objective delta vs numpy: nm {worst['nm']:+.4f}, "
          f"bfgs+nm {worst['bfgs']:+.4f} (negative = JAX found a lower minimum)")


if __name__ == "__main__":
    main()
