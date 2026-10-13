# 专用恒星与伴星模型

[English](../en/companions.md) · [GitHub](https://github.com/jiadonglee/spbin)


### 红巨星

```python
from spbin import fit_giant_companion

result = fit_giant_companion(
    sed, teff=(4895, 150), logg=(2.54, 0.3), feh=(-1.22, 0.2),
    threshold=10,
)
print(result["best_m2"], result["detection"], result["m2_excluded"])
```

`teff`、`logg`、`feh` 为 APOGEE 标签尺度上的高斯先验。默认用经验巨星模板，扫描 1.5–15 太阳质量、年龄 0.01 Gyr 的太阳金属丰度主序伴星。默认 `use_wise=True`，不同于基础 `fit`。

`GiantTemplate().evaluate(teff, logg, mh)` 返回模板通量及模型误差信息；`in_support` 检查标签支持。默认巨星亮度为自由缩放；`luminosity="parsec"` 加入演化亮度先验和质量边界，`"massfree"` 保留质量边界。`parallax=(mean, sigma)` 可替换目录视差；`dust_prior` 随试探距离求消光约束。

`rows` 保存每个伴星质量的拟合，其中 M2=0 为巨星单星。`detection` 为巨星单星目标函数减去扫描最低值；每行的 `delta` 则为该行目标函数减去巨星单星值。`m2_excluded` 是最佳质量以上、超过指定阈值的第一个网格质量。详见 [巨星验证](../validation-giant.md)。

### 热亚矮星

```python
from spbin import fit_subdwarf_companion

result = fit_subdwarf_companion(
    sed, companions=("dwarf", "subgiant"),
    extinction=0.0, subdwarf_prior={"logg": (5.6, 0.3)},
)
best = result["hypotheses"][result["preferred"]]
print(best["subdwarf"], best["fractions"]["beta_G"])
```

默认比较 `fgk`、`sdb`、`sdb+dwarf`、`sdb+subgiant`。`companions=()` 关闭复合假设，保留单星比较。默认三种氦层级为 `H`、`mid`、`He`。亚矮星半径自由，伴星可以是演化轨道上的矮星或经验模板上的次巨星。

`subdwarf_prior`、`companion_prior` 接受标签的 `(mean, sigma)` 字典。`companion_mass`、`companion_age_gyr`、`companion_feh` 可固定矮星伴星参数。默认 `extinction=None`，调用时提供消光先验或显式固定 E。默认不用 WISE、SPHEREx、蓝端 XP 或 GALEX。

输出 `hypotheses`、`preferred`、各假设相对最优值的 `delta`、`mask`、`n_fit` 和拟合数据。分量输出包括 `subdwarf`、`companion`、`fractions`、`profile`、`ranges`；`fractions["beta_G"]` 是亚矮星占总 G 光的份额。

`SubdwarfModel().predict(teff, logg, radius, tier)` 返回 10 pc 的 168 通道通量、蓝端通量及粗采样光谱。`spbin.subdwarf.blue_xp` 和 `galex` 提供可选诊断数据。见 [亚矮星验证](../validation-subdwarf.md)。

### DA 白矮星

```python
from spbin import WhiteDwarfModel, fit_whitedwarf_companion

wd = WhiteDwarfModel(hydrogen_layer="thick")
prediction = wd.predict(teff=12000, mass=0.6)
physical = wd.physical(teff=12000, mass=0.6)
result = fit_whitedwarf_companion(sed, model=wd, extinction=0.0)
best = result["hypotheses"][result["preferred"]]
```

默认使用经过单 DA 实测标定的 Koester 光谱和厚氢层 C/O 冷却关系。`hydrogen_layer="thin"` 用于薄氢层比较；`calibration=None` 使用未作经验校准的光谱。

`predict` 返回字典，`flux` 为 10 pc 通量，另有蓝端与粗采样光谱、物理参数。`physical` 返回 `teff`、`mass`、`radius`、`logg`、`cooling_age_gyr`、`luminosity`；`teff_at_age(mass, age_gyr)` 反求支持范围内的温度。自由半径预测需同时指定 `logg`、`radius`。

拟合默认比较 `dwarf`、`wd`、`wd+dwarf`；`hypotheses` 可选择这些假设及 `ms+ms`。`companions=False` 比较两种单星假设。`free_radius=True` 独立拟合白矮星温度、重力和半径，此时质量由 gR²/G 推导，冷却年龄不可用。

`whitedwarf_prior` 支持温度、质量、重力、半径约束；`companion_prior` 还支持年龄、金属丰度等约束。主序伴星默认 5 Gyr、太阳金属丰度，设置 None 可释放。

结果含 `hypotheses`、`preferred`、相对最优值的 `delta`、`detection_statistic`、数据掩码与氢层设置。`detection_statistic` 比较复合模型与已请求的最佳单星模型。`fractions["beta_G"]` 是白矮星占系统总 G 光的份额。见 [白矮星指南](../whitedwarf.md)。

### 白矮星光度包络

```python
from spbin import whitedwarf_light_limit

limit = whitedwarf_light_limit(
    sed, masses=[0.6, 0.8, 1.0],
    temperatures=[8000, 12000, 20000], delta=9.0,
)
print(limit["beta_G_upper"], limit["intervals"])
```

扫描给定质量与温度，或改用 `cooling_ages_gyr`。在每一点重新拟合主星与共享参数，保留可能不连通的允许区间。输出 `beta_G_upper`、`flux_ratio_G_upper`、`profile`、`intervals`、`single_primary` 和可选的 `luminous_companion` 比较。没有允许的白矮星模型时返回 NaN 上限。

`galex_upper_limits` 可提供系统实测总 UV 通量上界，单位与 `SED` 一致。`delta=9` 定义本次扫描的目标函数包络，返回 `confidence_level=None`。

