# 常见问题

[English](../en/faq.md) · [GitHub](https://github.com/jiadonglee/spbin)


**为什么 `fit` 提示需要正视差？** 绝对通量模型使用视差设置距离。使用有物理依据的正视差与不确定性；需要更一般距离后验时采用外部联合模型。

**为什么热星没有使用 SPHEREx？** `StellarModel(hot=True)` 的共同支持掩码仅含 XP 与 J/H/Ks。

**为什么返回 None？** `StellarModel.evaluate` 的分量可能超出支持；`fit_star` 则可能遇到不支持区域或缺少巨星标签。读取对应原因并选择覆盖该物理区域的路线。

**为什么设置 `extinction=None` 后报地图错误？** 基础 `fit` 会使用 EdenhoferPrior；需要安装 dust 扩展、配置地图并提供坐标。固定 E 时可用数值 `extinction=`。

**为什么 batch 下载后没有 SED？** `download_gaia` 返回原始产品。单源校准后的对象使用 `download`。

**如何重复分析？** 保存原始 `SED`，记录软件提交、拟合参数、先验、模型路线和最终数据掩码，再保存结果与图。


## 示例、支持与许可

- [基础 SED notebook](https://github.com/jiadonglee/spbin/blob/main/examples/01_sed_fit.ipynb)
- [Gaia 明亮孪星分支](https://github.com/jiadonglee/spbin/blob/main/examples/02_gaia_orbit_twin.ipynb)
- [暖星双星实验](https://github.com/jiadonglee/spbin/blob/main/examples/03_warm_binaries.ipynb)
- [消光与视差 notebook](https://github.com/jiadonglee/spbin/blob/main/examples/05_extinction_parallax.ipynb)
- [白矮星脚本](https://github.com/jiadonglee/spbin/blob/main/examples/06_whitedwarf.py)
- [GitHub Issues](https://github.com/jiadonglee/spbin/issues)
- [源码许可](https://github.com/jiadonglee/spbin/blob/main/LICENSE)：项目元数据声明 MIT AND BSD-2-Clause，保留对应版权与许可文本。

使用数据和模型时，同时遵循 Gaia、2MASS、WISE、SPHEREx 与相关大气/演化模型的引用要求。来源汇总见 [数据与来源](../data.md)。

