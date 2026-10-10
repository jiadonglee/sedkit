# sedkit 白矮星路线：计划 2026-10-10

目标服务于 Gaia DR4 双星论文：给出白矮星（WD）伴星的光比 β_G，或其上限，供 q/AMRF 和轨道解使用。sdB 路线已经搭好了这套基础设施，WD 路线照搬它的结构。

## 范围

| 用途 | 内容 | 定位 |
|---|---|---|
| (a) 单颗 WD | Teff 和 M（R 由质量–半径关系给出），与光谱 Teff、log g 对照 | 前置验证；不单独作为科学目标 |
| (b) WD + K/M 复合星 | WD 在 XP 蓝端可见，三假设比较，对应 sdB 路线 | 主要新功能 |
| (c) FGK + 暗 WD 伴星的光心轨道 | `solve_dark_companion` 已有；SED 负责给出 β_G 上限，并排除发光伴星 | 直接服务 DR4 的 q/AMRF |

默认做 (c) + (b)，(a) 是它们的前提。只做单颗 WD 参数的话可以去掉 (b)，工作量约减半，但 DR4 论文用不上。

不覆盖：
- DB 以外的光谱型：DC、DQ、DZ，以及磁 WD；
- 6 kK 以下的冷 WD：Balmer 线消失，碰撞诱导吸收无法建模；
- log g 5–6.5 的 ELM WD：两张表都不覆盖，拒绝拟合；
- 双简并星、三体。

## 已核实的事实

- 本轮 SVO `koester2` DA 表实际选择 Teff 6000–80000 K、log g 7.0–9.5，共 858 节点。光谱覆盖约 90–3000 nm，包含 GALEX 与 Gaia。
- 运行表已包含 Bédard 厚/薄氢层 C/O 冷却关系。可分发的 DB 光谱表尚未落实。

## 任务

### 0. 锚点规模（最先做，决定后面的判据）

DR3 的普通星 XP 发布主要限于 G < 17.65，但额外包含约十万颗更暗的 WD 候选。按 `has_xp_continuous` 查询实际可用性，不以亮度切掉 WD。先做交叉匹配，看手上有多少锚点：
- 光谱参数 DA/DB：SDSS DR16 DA（Kepler+2021）、DESI EDR（Manser+2024）、Montreal WD Database 中参数来自光谱拟合的文献子样本；
- 光谱分类、测光参数：Kilic+2025（100 pc，SDSS 天区），用于覆盖率和对照，不作独立光谱定标；
- 只有测光参数的：Gentile Fusillo+2021，仅用于零检验和覆盖率统计，不作定标；
- WD+MS：Rebassa-Mansergas SDSS WDMS 目录（有 Teff_WD、log g 和 M 型伴星次型），Nayak+2024 Gaia/GALEX 测光 WD+MS 候选；
- 暗 WD 伴星轨道：Yamaguchi+2024（有 RV 证认的宽 post-CE WD+MS）、Shahaf+2024（non-class-I 父样本，需区分 II/III 类和颜色过量筛选）。

按光谱 DA 锚点数分三档：
- ≥ 200 颗：照搬 sdB 的做法，exp(a + W b) 校正加交叉验证；
- 50–200 颗：只拟合一个与 Teff 有关的低阶校正；
- < 50 颗：不做经验校正，只报告残差，由模型误差项吸收。

### 1. DA 大气表（`models/whitedwarf/`，`scripts/build_whitedwarf_model.py`）

- 复用 `build_subdwarf_model.py` 的 select / fetch / table 流程和 XP 正向模型（332–992 nm、168 通道、G/BP/RP/GALEX、2 nm 粗谱）。
- 网格：Teff 6–80 kK，log g 7.0–9.5。
- 质量–半径关系：Bédard+2020 冷却轨道（厚氢层 DA），制成 R(Teff, M) 和冷却年龄表。
- 坏点筛查沿用 `_screen`。在 `docs/data.md` 里写网格概况。

### 2. 校正形式（首先要测的风险）

- 热星表的 `delta_a + W·delta_b` 按 B 型星的 `ew_max_nm` 归一。sdB 校正是在 sdB 的 Balmer 等值宽度上拟合的。
- DA 在 10–20 kK 的 Balmer 线远宽于这两者，直接沿用等于外推。
- 做法：
  1. 先用无校正的表拟合锚点，看残差随 W 和 Teff 怎么变；
  2. 再决定 WD 表的校正形式，只用 WD 锚点拟合；
  3. 不默认继承 `correction="ab"`。

### 3. `fit_whitedwarf_companion`（`src/sedkit/whitedwarf.py`）

- 结构照搬 `subdwarf.py` 的 `_Problem` / `_Hypothesis`。不抽象出公共的"热致密星"基类，等第三条路线出现再说。
- WD 参数：Teff 和 M（或 log g），R 由质量–半径关系给出，共享视差与消光。默认用 M–R 关系，因为仅凭 XP 线宽，log g 约束很弱；代价是结果依赖冷却模型，作为局限写明。提供 `free_radius=True` 选项，供验证用。
- 三个假设：单颗 FGKM、单颗 WD、WD + 矮星。不做亚巨星伴星（没有需要它的样本）。
- 暗伴星模式（用途 c）：
  - 输入为主星的 SED；
  - 输出该 SED 允许的最大 WD 光比 β_G，按冷却年龄或 Teff 扫描；
  - 再输出 WD + 发光伴星假设被拒绝的程度。
  - 这些输出直接接 `orbit.solve_dark_companion`。
- GALEX 单独验证：sdB 路线关掉 GALEX，原因是较暗的 Luo 星使 Teff 偏 −5 kK。WD 大多低于饱和阈值，紫外又承载了主要的 Teff 信息。在 WD 锚点上重新做开关决定。
- 392 nm 以下的 XP 同样单独验证。

### 4. 验证（判据在运行前写进 `docs/validation-whitedwarf.md`）

判据中的具体数值要等任务 0 得出锚点数后再定。下表是草案。

| 项 | 内容 | 判据草案 |
|---|---|---|
| 1 | 单颗 DA，交叉验证，对照光谱 Teff、log g | Teff 偏差 ≤ 3%、离散 ≤ 8%；M 离散 ≤ 0.08 M☉。冷于 13 kK 用 3D 修正后的 log g |
| 2 | 零检验：单颗 WD 不判为复合星；APOGEE 恒定 RV 的 FGK 星不判出 WD 伴星 | 假阳性 95% 上限 ≤ 2% |
| 3 | WD+MS，对照 SDSS WDMS 目录 | Teff_WD 离散 ≤ 15%；伴星次型 ±1 |
| 4 | β：对照文献谱分解的光比 | 偏差 ≤ 0.05 |
| 5 | 暗伴星轨道：Yamaguchi/Shahaf 系统用 M_WD 已知的 DR3 轨道 | SED 给出的 β_G 上限与 RV 质量自洽；β 对 M_WD 的影响 ≤ 0.02 M☉ |
| 6 | 注入恢复：WD + K/M，以及 FGK + 暗 WD | β、M_c 离散。失配注入：Teff 标度、M–R 关系（薄氢层或 C/O 核换成 He 核）、消光先验偏差。检测阈值在 WD+M 注入和 FGK 零样本上重新定，不沿用 sdB 的 Δ > 25 |

### 5. 文档、测试、提交

- README：Models 表和"What the fits measure"中新增 WD 行。
- `docs/{model,api,data,orbit}.md` 相应更新。
- 新增 `tests/test_whitedwarf.py`。
- 新分支 `feat/whitedwarf`，从 `feat/subdwarf` 切出，按功能分提交。

## 时间安排（以 DR4 为界）

DR4 前必须完成：
- 任务 0；
- 任务 1：DA 表；
- 任务 2：校正；
- 验证 1、2、5；
- 暗伴星 β 上限模式。

这是论文直接用到的部分。

DR4 后再做：
- WD + K/M 复合星的定标（验证 3、4、6）；
- DB 档；
- 392 nm 以下 XP 在复合星中的使用。

## 风险

- 锚点数可能只有几十颗。这时经验校正退化为误差项，判据要相应放宽。
- 质量–半径关系依赖冷却模型和氢层厚度；薄氢层会使 M 偏约 0.02 M☉。
- 光谱 log g 本身存在系统偏差：热 DA 的光度 Teff 与光谱 Teff 不一致（Genest-Beaulieu & Bergeron 2019），冷 DA 需要 3D 修正。
- XP 在暗蓝源上的定标误差可能主导 WD 的残差。

## 执行状态

已实现 DA 模型、校正、(b)/(c) 两种拟合路线、厚/薄氢层敏感性与联合轨道似然，
并完成单 DA、零检验、SDSS WDMS、谱分解光比、31 个轨道和 48 次注入检查。
结果和仍未达到的科学精度写在 [progress.md](progress.md)。
DB、He 核冷却、长波红外及更强的热 DA/伴星谱约束仍需补充数据；
原定精度判据保留，未通过的部分不标为完成验证。
