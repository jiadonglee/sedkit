# 安装

[English](../en/getting-started.md) · [GitHub](https://github.com/jiadonglee/spbin)



Python 版本要求：3.10 或更高。通过 GitHub 安装：

```bash
python -m pip install "spbin @ git+https://github.com/jiadonglee/spbin.git"
```

包含数据下载功能：

```bash
python -m pip install "spbin[download] @ git+https://github.com/jiadonglee/spbin.git"
```

按需安装扩展：

| 扩展 | 用途 | 依赖 |
| --- | --- | --- |
| `download` | Gaia、测光与相关目录下载 | Astropy、Astroquery、GaiaXPy、Requests、PyVO |
| `dust` | Edenhofer 三维消光先验 | dustmaps、Astropy |
| `spherex` | 隔离环境中的 SPHEREx 提取 | uv |
| `notebook` | 交互式示例 | JupyterLab、ipykernel |
| `test` | 仓库测试 | pytest 及相应数据工具 |

```bash
python -m pip install "spbin[download,dust,spherex,notebook] @ git+https://github.com/jiadonglee/spbin.git"
```

从本地源码开发：

```bash
git clone https://github.com/jiadonglee/spbin.git
cd spbin
python -m pip install -e ".[download,notebook]"
```

检查导入：

```python
import spbin
print(spbin.__version__)
```

需要固定分析版本时，用 Git 提交或已存在的标签代替浮动分支，例如将安装 URL 写为 `git+https://github.com/jiadonglee/spbin.git@<commit>`。


## 第一次拟合

```python
from spbin import download, fit, plot

sed = download("858860697467058688", cache_dir="data")
result = fit(sed, age_gyr=5.0, feh=0.0)

print("Single objective:", result["single"]["objective"])
print("Binary objective:", result["binary"]["objective"])
print("Delta:", result["delta"])
print("Mass ratio:", result["binary"]["q"])

sed.save("data/example_sed.npz")
fig = plot(sed, result, path="example_sed.png")
```

这个调用固定年龄为 5 Gyr、金属丰度为太阳值、消光为零，并固定目录视差。`delta = objective_single - objective_binary`；正值表示双星拟合的目标函数更低。它衡量当前模型与约束下的拟合改善。

释放年龄和金属丰度：

```python
result = fit(sed, age_gyr=None, feh=None)
```

只拟合单星：

```python
single = fit(sed, kind="single", age_gyr=5.0, feh=0.0)
print(single["m1"], single["converged"], single["at_bounds"])
```

