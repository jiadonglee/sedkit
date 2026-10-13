# 轨道与联合似然

[English](../en/orbits.md) · [GitHub](https://github.com/jiadonglee/spbin)


### 同龄主序双星的分支

```python
from spbin.orbit import solve_orbit, rank_roots

roots = solve_orbit(
    a0_mas=0.6978, parallax_mas=13.913,
    period_day=339.57, m1=0.686,
)
ranked = rank_roots(sed, roots, parallax_mas=13.913, fit_parallax=True)
```

`solve_orbit` 返回暗伴星、较暗伴星或近等质量明亮分支；`rank_roots` 沿轨道分支重新拟合 SED 并排序。`solve_amrf` 接受无量纲天体测量质量比函数。排序输出的 `delta` 是该分支目标函数减去最佳值。

### 两种不同恒星组成的明亮系统

`solve_luminous_pair` 用总光份额 β 求两条 `B>beta` 和 `B<beta` 分支；`mass_from_rv` 从伴星 RV 振幅和倾角求星 1 质量；`branch_from_rv` 将 RV 与光心尺度联系起来。`solve_dark_companion` 求给定伴星/主星光度比下的伴星质量。周期单位是 day，视差与角半长轴为 mas，RV 为 km/s，角度为度。

这些接口的光度定义不同，必须按下表传参：

| 接口/字段 | 光度定义 |
| --- | --- |
| `StellarModel.evaluate()["beta_g"]`、`orbit.amrf`、`solve_dark_companion(beta=...)` | F2/F1 |
| WD、sdB 的 `fractions["beta_G"]` | WD 或 sdB 的 F1/(F1+F2) |
| `solve_luminous_pair(beta=...)` | 星 1 的 F1/(F1+F2) |
| `whitedwarf_light_limit()["beta_G_upper"]` | WD/(主星+WD) |
| `whitedwarf_light_limit()["flux_ratio_G_upper"]` | WD/主星 |

两种不同恒星的光心满足 `a0 = a × |B-beta|`，其中 B=M1/(M1+M2)。给 `solve_luminous_pair` 传入 WD 或 sdB 总光份额时，它们对应星 1。

### 外部采样器

```python
from spbin import loglike_sed

ll = loglike_sed(
    sed, m1=0.75, q=0.8, age_gyr=5.0, feh=0.0,
    parallax_mas=20.0, extinction=0.0, model=model,
)
```

`loglike_sed` 与 `loglike_whitedwarf_sed` 在共享试探参数上返回 SED 对数似然，包含模型误差及其行列式，不含参数先验或目录视差惩罚。参数不受支持时返回 −∞。在外部联合模型中统一加入一次先验与相应观测似然。`loglike_sed` 省略固定的 −N/2 log(2π)，适合在相同数据掩码上比较。

`loglike_whitedwarf_sed` 的 `primary_mass` 指主序分量质量；`mass` 指白矮星质量。将同一白矮星质量同时用于冷却半径和外部轨道。与 orblet 的参数映射见 [联合拟合指南](../orblet.md)。

