# 项目介绍

[English](../en/index.md) · [GitHub](https://github.com/jiadonglee/spbin)


spbin 是用于下载、建模和拟合恒星光谱能量分布（SED）的 Python 软件包。它把 Gaia XP 光谱、2MASS 和 AllWISE 测光以及可选的 SPHEREx 光谱放到同一物理通量体系中，通过恒星演化模型和光谱模型比较单星与未分辨双星，并为 Gaia 光心轨道提供伴星光度信息。

核心拟合依赖 NumPy、SciPy 和 Matplotlib。恒星网络、演化表和各类光谱模型随软件包提供；基础拟合无需 JAX、GPU 或单独安装 J-CAPS。

网站首页介绍文案：

> 从 Gaia XP 到物理双星模型。spbin 在真实距离下联合建模光谱与红外测光，比较单星和双星，分析白矮星及热亚矮星伴星，并连接 Gaia 光心轨道。

主要使用者是进行恒星、双星和 Gaia 数据分析的研究人员。用户可以从 Gaia source ID 开始，也可以导入自己的通量数组，在本地 Python 环境中完成计算。


## 功能总览

| 功能 | 主要入口 | 输入 | 输出 |
| --- | --- | --- | --- |
| 单源 Gaia XP 与红外数据下载 | `download` | source ID 或天球坐标 | 标准化 `SED` 对象与本地缓存 |
| Gaia 目录查询与批量产品下载 | `query_gaia`, `download_gaia` | ADQL 或 source ID 列表 | Astropy Table、原始 FITS ZIP 文件记录 |
| SPHEREx 获取与导入 | `download_spherex`, `load_spherex` | `SED` 或 XphereX CSV | 附加 SPHEREx 通道的 `SED` |
| 单星与同龄双星拟合 | `fit` | `SED`、年龄与金属丰度设置 | 质量、质量比、模型通量、拟合统计量 |
| HR 图位置与自动选择模型 | `locate`, `fit_star` | Gaia 测光、Ks、视差 | 区域、所选假设及其拟合结果 |
| 红巨星与热主序伴星 | `fit_giant_companion` | `SED` 和巨星标签先验 | 伴星质量扫描、改善量、条件排除质量 |
| 热亚矮星与冷伴星 | `fit_subdwarf_companion` | `SED`、标签与消光约束 | 单星/复合假设、半径和通带光度份额 |
| DA 白矮星与主序伴星 | `fit_whitedwarf_companion` | `SED`、冷却关系或自由半径设置 | 白矮星与伴星参数、假设比较 |
| 白矮星光度包络 | `whitedwarf_light_limit` | 质量/温度或冷却年龄网格 | G 波段光度上包络、允许区间 |
| Gaia 光心轨道分支 | `spbin.orbit` | 轨道尺度、周期、视差、质量/光度 | 暗伴星、较暗伴星和明亮分支的质量解 |
| 外部联合拟合 | `loglike_sed`, `loglike_whitedwarf_sed` | 一组共享物理参数 | SED 对数似然 |
| 科研绘图 | `plot` | `SED` 与标准 `fit` 结果 | Matplotlib Figure、PNG/PDF |

