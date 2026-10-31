# 宇航确认清单

2026-10-31 按用户最新确认更新。**以下主要是来源核对和投稿收尾，不等于数据有错。**
文稿审计基准仍见 `metadata/release_document_baseline.json`；本次未重新审读或修改 MS/SI。
用户确认与独立程序验证分开记录于 `metadata/publication_decisions_20261031.json`。

**不用重新提供的内容：** Fig. 5A 的 Index_2 来源、归档主程序和数值缩放；
S25 两条原始 CSV；Table S28 的公共 selector/runner、种晶/表面生长及有限池代码；
当前响应面条件表、轨迹统计、部分文献及各图表整理来源；5090 的两个修正版引擎和八个模型；
Fig. 6/S7 三处差异的旧版/修订版；S21 April 3 主版本与备选版本。
完整证据和精确缺失路径见 [投稿发布清单](docs/PUBLICATION_RELEASE.md)。

## 当前剩余事项

1. **Fig. 5C：补底层曲面分析模块。** 共享盘已有绘图脚本和数据；尚未定位
   `analyze_figure6_current_parameter_phase_v1.py`，需核对曲面插值/筛选规则及实际运行命令。
2. **Fig. 5D：补逐行轨迹来源。** 百分比已独立复算，结构按用户本次确认已核实。
   现在需要把分类表每一行对应到原始轨迹文件、运行 ID 和条件；不再笼统写为“百分比/结构未核实”。
3. **模拟依赖：核对各家族是否完整对应。** 两个修正版引擎和八个模型已归档；
   继续逐家族登记 seed、host/checkpoint、实际代码版本和续跑关系，而不是把依赖全部称作缺失。
4. **Tables S1/S2、S6、S8：补来源链。** 分别核对最终文献分类、熵/Gaussian 计算链、
   12 条诱导时间对应的原始实验记录。已有派生表不能替代这些逐行来源。
5. **Entity 13：SI 补说明。** 24 条轨迹的结构是最后保存的 entity-13 快照，分为 9 类；
   保留 I03 延迟增长例外，并区分首次通过时间与最后保存结构，避免写成唯一或必然触发生长的临界核。
6. **最终文件登记。** Fig. 6 尚需登记采用文件的路径、SHA-256 和版本并更新图源索引。
   **S7 已确定算法时间 `u`，正式入口已同步为 `python 10_plot_versions/render_s7.py`；秒轴仅留作历史。**
7. **投稿收尾。** 由文稿负责人清洁正文/SI、统一图号和交叉引用；确认代码/数据许可、贡献者及大文件公开范围。
   本次不编辑 Word，不自动选择许可，不上传 Zenodo 或推送 GitHub。

## 已完成，不必重复索取

- [x] 两个修正版引擎及八个模型已从 5090 找回；UiO-66 引擎匹配六条结果中的确切哈希。
- [x] 已确认 5090 上的 `/home/wave/MOL_growth_recalculation_20260830/`；不需要发送账号密码或私钥。
- [x] Fig. 5B 的数值已从 1260 个条件独立复算对上。
- [x] 三个哈希差异已逐行解释，原版和修订版分别存入 `10_plot_versions/`；不再列为代码缺失。
- [x] S7 发布坐标已确认为算法时间 `u`；正式入口不调用秒轴版本，也不覆盖历史代码。
- [x] S21 April 3 主来源、initialize_structures1 备选与 April 10 重跑分别保留，三组保存均值均已复算。

## 历史审计详情

以下保留早前核对背景，供最终文稿编辑时参考；不自动表示这些问题在最新文稿中仍存在。

- [ ] **确认图号基准和最终拼图文件。** 请给当前 Fig. 3、5 及 S7、S21–S24 的可编辑拼图/导出文件名称，确认 `FIGURE_MAP.csv` 的对应关系。原始程序的旧文件名不代表当前编号。
- [ ] **Fig. 3D 来源登记。** 我们已对上 24 条轨迹、9 个阈值、216 个数值。SI 搜索 `Main panel: inherited first-passage plot, exact trace list unresolved.` 及 `Exact trace/frequency export for A and first-passage cohort for D are unmatched.`：其中 D 可改为已对应本包的 cohort；A 仍单独处理。
- [x] **S7 时间口径。** 发布采用 `u = CSV time_s / 0.175`。主文历史 Fig. 3D 的秒轴不因本次修改自动改变；需由文稿负责人同步各图说明。
- [ ] **24-run 与 20-run 分开。** Fig. 3A / S8–S10 的历史 motif 来源不能由 24 条 long-growth 数据替代。请确认那组独立轨迹的 ID、频率表、条件和结构选取规则。
- [ ] **entity-13 表述。** 按当前清单补入最后保存、9 类、I03 例外和 first/last-passage 区分；本次尚未写入 SI。
- [ ] **DLS 方法。** 确认处理步骤、117–1170 s 的拟合范围与 0–600 s 的展示窗口；R²=0.972658 对处理后的 apparent progress。nG 从 {1,2,3} 中选择，不写成 a priori 固定 2。
- [ ] **Fig. 5A 时间及零点。** 已确认 Index_2 原运行来源和 raw timing × 0.2405 的绘图约定。请核对最终图与 SI `Source-scaled elapsed ms after subtraction of the original 620 ms offset.` 是否一致；不要把数值缩放写成已验证的物理标定。
- [ ] **Fig. 5D 行到轨迹映射。** 结构已按用户确认核实；剩余的是跨 71 条件、284 行分类记录各自对应的原始轨迹及筛选依据，不能写成新扫描概率。
- [ ] **S21–S22 混合种子。** `11_s21_april03/` 已保留共享说明指认为 S21 的 April 3 `Zr6_Zr12_mix`、以及单独的 `initialize_structures1` 备选；`05` 的 April 10 重跑未动。三组均值均已复算。请最终确认 S21/S22 对应哪组及结构文件，不能混用或合并为 12 个独立重复。
- [ ] **S23–S24 修复。** 确认对应同一条连续轨迹及原始空位分母 591/54，而非新 5090 ensemble。初始、N=2800 与 N=100001 的图像来源分别登记；大终态 checkpoint 未随轻量包发送。

## 上游数据的最后闭环

- [ ] 每个新版家族补齐必要 seed/host/checkpoint 的确切版本和位置；已收录统计表不等于全部逐事件结构历史。续跑不能作为新增独立重复。
- [ ] Fig. 7 / S42-S45 的已收曲线/频率输入与最终结构拼图逐一对应；必要大 pickle 进入外部归档，不用旧 preassociation Figure 7 替代。
- [ ] 本轮优先核对 Tables S1/S2、S6、S8。此前 S4 等逐表审计细节仍保留在 `metadata/table_source_limits.csv`，本次没有重新复算或自动注销这些历史记录。

## 文稿编辑者最后处理

- [ ] SI Table S7 的 rho_off=30 中 2922 是最高 entity，实际终点为 2911；统一 maximum / endpoint 定义，并避免将 N=10000 写成所有条件均成功。
- [ ] SI 两个 `S4.4` 标题（Size-resolved finite cluster PXRD / Established-seed growth）在最终视图里都存在，需统一编号和引用。
- [ ] 数据来源确认后再由一个编辑副本统一 MS/SI 时间标签、caption 和 cross-reference，避免多人同时改同一文档。

确认时可逐项填：结论、对应路径/文件名、采用或不采用、负责人和日期。不确定项保持待核实，不用本包的重画预览替代最终图源证据。
