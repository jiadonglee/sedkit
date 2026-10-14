# 恒星预测与拟合 API

[English](../en/stellar.md) · [GitHub](https://github.com/jiadonglee/spbin)


### `StellarModel`

```python
from spbin import StellarModel

model = StellarModel()
pair = model.evaluate(m1=0.75, q=0.8, age_gyr=5.0, feh=0.0)
labels = model.labels([0.75, 0.60], age_gyr=5.0, feh=0.0)
inside = model.in_domain(labels)
```

`evaluate` 以质量、年龄、金属丰度推导各分量的恒星参数，叠加绝对通量；`q=0` 表示单星，`0<q<=1` 表示同龄双星。不支持的分量使返回值为 None。

主要输出：`flux_10pc`、`components`、`masses`、`labels`、`teff`、`logg`、`radius`、`M_G`、`hot_weight`、`beta_g`、`a_phot_over_a1`、`age_gyr`、`feh`、`route`。`labels` 每行顺序为 `[Teff, M_Ks, G-Ks, [M/H]]`。

`predict_labels(labels)` 直接使用四个标签预测；`predict_hot(teff, logg, radius)` 直接求热星表。`wavelength_um`、`channels`、`support` 提供波长、通道名与模型支持掩码。

观测尺度模型满足：

```text
F_observed = F_10pc × (parallax_mas / 100)^2 × exp(-E × k_lambda)
```

### `fit` 参数

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `kind` | `"both"` | `single`、`binary` 或两者 |
| `model` | `None` | 默认创建 `StellarModel()`；批量拟合可复用实例 |
| `age_gyr` | `5.0` | 固定年龄；None 释放年龄 |
| `feh` | `0.0` | 固定 [M/H]；None 释放金属丰度 |
| `q` | `None` | 自由质量比；数值固定；可调用对象将 q 绑定到其他参数 |
| `use_wise` | `False` | 是否把 W1/W2 纳入拟合 |
| `fit_parallax` | `False` | 在目录视差 ±3σ 内拟合，加入一次高斯约束 |
| `extinction` | `0.0` | 数值固定 E；None 拟合 E≥0 |
| `dust_prior` | `None` | 三维消光约束；自由 E 且未提供时创建 EdenhoferPrior |
| `age_prior` | `None` | `(mean, sigma)`，作用于 log10(age/yr)，需要释放年龄 |
| `logg_prior` | `None` | 主星 log g 的 `(mean, sigma)` |
| `teff_prior` | `None` | 主星温度 K 的 `(mean, sigma)` |

自由双星 q 的搜索范围为 0.1–1，并受各分量实际支持范围限制。可调用 q 的签名为 `q(m1, age_gyr, feh, parallax_mas)`。视差误差不存在或不为正时保持固定。算法使用多起点 Nelder–Mead 局部优化。

### `fit` 返回结构

`kind="both"` 返回 `single`、`binary`、`delta` 和 `source_id`。另两种 kind 直接返回单个拟合字典。

| 字段 | 含义 |
| --- | --- |
| `m1`, `m2`, `q` | 分量质量与质量比 |
| `age_gyr`, `feh`, `logg` | 年龄、金属丰度、主星重力 |
| `parallax_mas`, `extinction_e` | 本次模型使用的视差与消光 |
| `flux`, `components` | 实际距离并施加消光后的总/分量模型通量 |
| `objective` | −2 ln L 加视差、消光与标签先验项 |
| `m2lnl` | SED 的 −2 ln L，含模型协方差行列式、不含先验 |
| `chi2` | 使用测量误差与模型误差的残差二次项 |
| `n_fit`, `mask` | 实际使用的通道数与 168 通道掩码 |
| `converged` | 优化器是否满足收敛条件 |
| `at_bounds` | 靠近搜索边界的参数名 |
| `dust_prior_mean`, `dust_prior_sigma`, `dust_prior_penalty` | 消光先验均值、宽度、惩罚 |
| `label_prior_penalty` | 年龄、log g、Teff 先验的总惩罚 |
| `beta_g`, `M_G`, `a_phot_over_a1`, `route`, `kind` | 光度比、绝对星等、光心系数、模型路线及假设 |

### HR 图自动路由

```python
from spbin import locate, fit_star

location = locate(sed)
print(location["region"], location["reason"])
result = fit_star(sed, age_gyr=None, feh=0.0)
print(result["preferred"])
```

`locate` 需要正视差、Ks，以及元数据中的 `phot_g_mean_mag`、`phot_bp_mean_mag`、`phot_rp_mean_mag`。区域值为 `ms`、`hot_ms`、`giant`、`sdb`、`wd`、`unsupported`；同时返回去消光后的颜色、M_G、候选假设和原因。

`fit_star` 返回 `location`、`region`、路线自身的 `result`、`objectives`、`delta`、`preferred`。在最低目标函数的 `tie=1.0` 范围内优先较简单假设。`options` 将参数传入对应路线并优先于通用设置。巨星需要标签先验，例如：

```python
result = fit_star(
    sed, region="giant",
    labels={"teff": (4895, 150), "logg": (2.54, 0.3), "feh": (-1.22, 0.2)},
)
```

不支持的区域或缺少巨星标签时，`result` 为 None，原因写入 `location`。

