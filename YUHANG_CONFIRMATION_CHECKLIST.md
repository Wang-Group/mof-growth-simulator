# 宇航确认清单

2026-10-29 更新，替代旧版缺失清单。对应文件名为 2026-10-18 的最新 MS
和 2026-10-19 SI，以 `metadata/release_document_baseline.json` 的哈希为准。
没有修改文稿，也不表示宇航已接受分工。

**不用重新提供的内容：** Fig. 5A 的 Index_2 来源、归档主程序和数值缩放；
S25 两条原始 CSV；Table S28 的公共 selector/runner、种晶/表面生长及有限池代码；
当前响应面条件表、轨迹统计、部分文献及各图表整理来源。
完整证据和精确缺失路径见 [投稿发布清单](docs/PUBLICATION_RELEASE.md)。

## 最优先补充

- [ ] 找到新版实际执行的 `code/candidate_corrected/` 引擎和 `models/` 文件；UiO-66 引擎需匹配结果中的 SHA-256 `4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9`。
- [ ] 确认 `/home/wave/MOL_growth_recalculation_20260830/` 在哪台计算主机；旧账号路径不等于当前执行位置，不需要发送账号密码或私钥。
- [ ] 补 Fig. 5C 的 `analyze_figure6_current_parameter_phase_v1.py` 确切版本、相界处理规则和命令；Fig. 5B 的数值已独立复算对上。
- [ ] 确认清单中 3 个哈希变化文件是否为后续合法修改，并提供版本说明。它们暂未纳入发布内容。
- [ ] 确认公开代码/数据的许可、贡献者和引用方式；确定哪些原始轨迹与 checkpoint 进入 Zenodo。

## 优先确认：决定本包数据怎么用于现稿

- [ ] **确认图号基准和最终拼图文件。** 请给当前 Fig. 3、5 及 S7、S21–S24 的可编辑拼图/导出文件名称，确认 `FIGURE_MAP.csv` 的对应关系。原始程序的旧文件名不代表当前编号。
- [ ] **Fig. 3D 来源登记。** 我们已对上 24 条轨迹、9 个阈值、216 个数值。SI 搜索 `Main panel: inherited first-passage plot, exact trace list unresolved.` 及 `Exact trace/frequency export for A and first-passage cohort for D are unmatched.`：其中 D 可改为已对应本包的 cohort；A 仍单独处理。
- [ ] **Fig. 3D / S7 时间口径。** 主文历史 D 图仍用 ×0.175 的秒轴；S7 已用原始算法时间 u。决定主文是否也改为 u，并同步 Table S27。不要仅根据旧文件 `_s` 后缀认定物理标定已验证。
- [ ] **24-run 与 20-run 分开。** Fig. 3A / S8–S10 的历史 motif 来源不能由 24 条 long-growth 数据替代。请确认那组独立轨迹的 ID、频率表、条件和结构选取规则。
- [ ] **entity-13 表述。** 本包结构是最后一次保存，分类 9 类，不是首次到达必然出现的唯一临界核。确认是否保留 I03 的长延迟例外及 first/last-passage 的区分。
- [ ] **DLS 方法。** 确认处理步骤、117–1170 s 的拟合范围与 0–600 s 的展示窗口；R²=0.972658 对处理后的 apparent progress。nG 从 {1,2,3} 中选择，不写成 a priori 固定 2。
- [ ] **Fig. 5A 时间及零点。** 已确认 Index_2 原运行来源和 raw timing × 0.2405 的绘图约定。请核对最终图与 SI `Source-scaled elapsed ms after subtraction of the original 620 ms offset.` 是否一致；不要把数值缩放写成已验证的物理标定。
- [ ] **Fig. 5D 结构映射。** 将图片逐一对应 `entity_N/unique_structure_ID.mol2`；数值百分比已重算对上。确认跨 71 条件、284 行记录的筛选标准、重复数和缺失值处理，不写成新扫描概率。
- [ ] **S21–S22 混合种子。** 本包是 `...regenerated...rerun_20260410` 的四次重复。确认现稿图采用的是此再生成家族还是更早的原始种子；不要把两者当同一原始运行。
- [ ] **S23–S24 修复。** 确认对应同一条连续轨迹及原始空位分母 591/54，而非新 5090 ensemble。初始、N=2800 与 N=100001 的图像来源分别登记；大终态 checkpoint 未随轻量包发送。

## 上游数据的最后闭环

- [ ] 每个新版家族补齐必要 seed/host/checkpoint 的确切版本和位置；已收录统计表不等于全部逐事件结构历史。续跑不能作为新增独立重复。
- [ ] Fig. 7 / S42-S45 的已收曲线/频率输入与最终结构拼图逐一对应；必要大 pickle 进入外部归档，不用旧 preassociation Figure 7 替代。
- [ ] Table S1/S2 的发表级文献纳入与编码、S4 的 NMR 线宽逐行来源、S6 的 Gaussian 原始输入/输出，按 `metadata/table_source_limits.csv` 补齐。

## 文稿编辑者最后处理

- [ ] SI Table S7 的 rho_off=30 中 2922 是最高 entity，实际终点为 2911；统一 maximum / endpoint 定义，并避免将 N=10000 写成所有条件均成功。
- [ ] SI 两个 `S4.4` 标题（Size-resolved finite cluster PXRD / Established-seed growth）在最终视图里都存在，需统一编号和引用。
- [ ] 数据来源确认后再由一个编辑副本统一 MS/SI 时间标签、caption 和 cross-reference，避免多人同时改同一文档。

确认时可逐项填：结论、对应路径/文件名、采用或不采用、负责人和日期。不确定项保持待核实，不用本包的重画预览替代最终图源证据。
