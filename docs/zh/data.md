# 数据对象、单位与通道

[English](../en/data.md) · [GitHub](https://github.com/jiadonglee/spbin)


`SED` 保存观测通量、误差、有效通道掩码、视差和目录元数据。观测值处在目标的实际距离；拟合不会重新归一化或修改观测误差。

| 量 | 单位/约定 |
| --- | --- |
| `flux`, `error` | `1e-18 W m^-2 nm^-1`，通道等效的 F_lambda |
| 波长 | μm；XP 网格为 392–992 nm，间隔 10 nm |
| `parallax_mas` | mas |
| `m1`, `m2`, `mass` | 太阳质量 |
| `radius` | 太阳半径 |
| `teff` | K |
| `logg` | log10(g / cm s^-2) |
| `age_gyr`, `cooling_age_gyr` | Gyr |
| `feh` | 恒星模型中的 [M/H]，dex |
| `extinction` | ZGR23 原生 E |
| `q` | M2/M1 |

数组固定为 168 个通道，使用 Python 的零起始索引：

| 索引切片 | 通道 |
| --- | --- |
| `0:61` | Gaia XP，61 点 |
| `61:64` | 2MASS J/H/Ks |
| `64:66` | AllWISE W1/W2 |
| `66:168` | SPHEREx，102 点 |

Gaia G/BP/RP 保存为元数据，不作为额外拟合通道。J/H/Ks/W1/W2 是宽带测量，以目录等效通量表示。

```python
import numpy as np
from spbin import SED, StellarModel

model = StellarModel()
prediction = model.evaluate(m1=0.75, q=0.8, age_gyr=5.0, feh=0.0)
flux = prediction["flux_10pc"] * (20.0 / 100.0)**2
error = np.maximum(np.abs(flux) * 0.03, 1e-6)

mock = SED(
    flux=flux,
    error=error,
    mask=np.isfinite(flux),
    parallax_mas=20.0,
    parallax_error_mas=0.1,
    source_id="mock",
)
mock.save("data/mock.npz")
restored = SED.load("data/mock.npz")
```

这里生成的是无噪声模型演示数据。真实观测需要填入测量值、测量误差和相应质量掩码。缺失通道用 NaN 且设为无效。


## 数据获取 API

### `download`

按精确 source ID 获取一颗星，或按 ICRS 坐标搜索：

```python
sed = download("858860697467058688", cache_dir="data")
sed = download(ra=120.0, dec=30.0, radius_arcsec=2.0, cache_dir="data")
```

返回 `SED`。目录、原始 XP、校准后 XP 和 `sed.npz` 保存在单源目录中。已有缓存会复用；`refresh=True` 更新源数据。测光质量选择保留唯一点源匹配与 A 级测光，WISE 还要求污染标记为零。下载使用 Gaia DR3 原始视差。

### `query_gaia` 与 `download_gaia`

```python
from spbin import query_gaia, download_gaia

catalogue = query_gaia(
    "SELECT TOP 20 source_id, parallax, phot_g_mean_mag "
    "FROM gaiadr3.gaia_source "
    "WHERE parallax > 10 AND has_xp_continuous = 'true'",
    cache_dir="data/query_nearby",
    tap="esa",
)
batches = download_gaia(
    catalogue["source_id"],
    products=["XP_CONTINUOUS", "RVS"],
    cache_dir="data/products_nearby",
    batch_size=100,
    workers=2,
)
for batch in batches:
    print(batch["product"], batch["path"], batch["unavailable_source_ids"])
```

`query_gaia` 返回 Astropy Table，保留整数、单位、数组列和掩码。`tap` 支持 `esa`、`ari` 或完整 TAP URL。服务器报告结果溢出时会报错。

`download_gaia` 返回每种产品、每批次的记录列表：`product`、`source_ids`、`unavailable_source_ids`、`path`、`cached`。`path` 是原始 FITS ZIP 的 Path，产品全部缺失时为 None。它不自动生成校准后的 `SED`。

ID 使用整数或十进制字符串；浮点数 ID 会被拒绝。同一缓存目录可恢复同一请求；改变查询、ID 顺序、产品、数据发布版本、批次大小或数据结构时使用新目录。完整缓存规则见 [Gaia 数据指南](../gaia.md)。

### SPHEREx

```python
from spbin import download_spherex, load_spherex

sed = download_spherex(sed, cache_dir="data", radius_arcmin=0.8)
# 或导入现有 XphereX PSF 提取结果：
sed = load_spherex(sed, "psf_spectrum.csv", method="psf")
```

下载接口执行 QR2 孔径提取；第一次调用会由 uv 准备 Python 3.12 和独立依赖。导入 CSV 需要 `wavelength_um`、`flux_jy`、`error_jy` 列；接受模型网格的通道子集，转换 Jy 到 `SED` 单位。

`sed.with_spherex(wavelength_um, flux, error, valid=None)` 接受完整 102 通道网格和已经转换到 `SED` 单位的数组，返回新对象。该方法与 CSV 导入都按通道赋值，不做插值。见 [SPHEREx 数据指南](../spherex.md)。

