# 绘图与导出

[English](../en/plotting.md) · [GitHub](https://github.com/jiadonglee/spbin)


```python
from spbin import plot

fig = plot(sed, result, path="sed.png", components=True, title="Gaia XP + infrared")
fig.savefig("sed.pdf", bbox_inches="tight")
```

本例的 `result` 使用快速开始的标准 `fit` 输出。`plot` 返回 Matplotlib Figure，支持 `axes=(sed_axis, ratio_axis)` 嵌入自定义布局。上图显示物理单位下的 λF_lambda，下图显示相对单星或双星模型的 ln 通量比。空心点表示可用但未参与拟合的通道。单星为橙色，双星为蓝色。

专用巨星、sdB、WD 路线的结果结构不同，应按它们的分量输出绘图；`plot` 的标准输入契约是基础 `fit` 结果。

