# Plotting and export

[中文](../zh/plotting.md) · [GitHub](https://github.com/jiadonglee/spbin)


```python
from spbin import plot

fig = plot(sed, result, path="sed.png", components=True, title="Gaia XP + infrared")
fig.savefig("sed.pdf", bbox_inches="tight")
```

Here `result` is the standard `fit` output from the quickstart. `plot` returns a Matplotlib Figure. `axes=(sed_axis, ratio_axis)` embeds it into a custom layout. The upper panel shows physical λF_lambda; the lower panel shows ln flux ratios relative to the single or binary model. Open points identify available but excluded channels. Single-star curves are orange and binary curves blue.

Dedicated giant, sdB and WD routes have different result structures. Plot their component outputs as appropriate; the standard `plot` input contract uses base `fit` results.

