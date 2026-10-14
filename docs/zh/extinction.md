# 消光与视差

[English](../en/extinction.md) · [GitHub](https://github.com/jiadonglee/spbin)


```python
from spbin import EdenhoferPrior

dust = EdenhoferPrior()
result = fit(sed, extinction=None, dust_prior=dust, fit_parallax=True)
```

`EdenhoferPrior(query=None, sigma=None)` 包装三维消光查询。`moments(ra, dec, distance_pc)` 给出该方向与距离的 E 均值和标准差；`penalty(extinction, mean, sigma)` 给出截断高斯约束项。

地图需提前配置，构造函数与拟合不会自动下载地图：

```python
from dustmaps.edenhofer2023 import fetch
fetch(fetch_samples=True)
```

后验样本地图约 19 GB，下载并加载时需安排足够磁盘和内存。坐标来自 `sed.metadata["ra"]`、`["dec"]`，单位为 ICRS 度。均值地图需要显式指定 E 先验宽度。地图范围之外的处理取决于拟合路线，见 [消光指南](../extinction.md) 和巨星接口说明。

