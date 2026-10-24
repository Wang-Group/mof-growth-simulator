# 宇航确认清单

对应 2026-10-18 MS / 2026-10-19 SI。以下只是待确认事项；没有修改文稿，也不表示宇航已接受分工。

## 优先确认：决定本包数据怎么用于现稿

- [ ] **确认图号基准和最终拼图文件。** 请给当前 Fig. 3、5 及 S7、S21–S24 的可编辑拼图/导出文件名称，确认 `FIGURE_MAP.csv` 的对应关系。原始程序的旧文件名不代表当前编号。
- [ ] **Fig. 3D 来源登记。** 我们已对上 24 条轨迹、9 个阈值、216 个数值。SI 搜索 `Main panel: inherited first-passage plot, exact trace list unresolved.` 及 `Exact trace/frequency export for A and first-passage cohort for D are unmatched.`：其中 D 可改为已对应本包的 cohort；A 仍单独处理。
- [ ] **Fig. 3D / S7 时间口径。** 主文历史 D 图仍用 ×0.175 的秒轴；S7 已用原始算法时间 u。决定主文是否也改为 u，并同步 Table S27。不要仅根据旧文件 `_s` 后缀认定物理标定已验证。
- [ ] **24-run 与 20-run 分开。** Fig. 3A / S8–S10 的历史 motif 来源不能由 24 条 long-growth 数据替代。请确认那组独立轨迹的 ID、频率表、条件和结构选取规则。
- [ ] **entity-13 表述。** 本包结构是最后一次保存，分类 9 类，不是首次到达必然出现的唯一临界核。确认是否保留 I03 的长延迟例外及 first/last-passage 的区分。
- [ ] **DLS 方法。** 确认处理步骤、117–1170 s 的拟合范围与 0–600 s 的展示窗口；R²=0.972658 对处理后的 apparent progress。nG 从 {1,2,3} 中选择，不写成 a priori 固定 2。
- [ ] **Fig. 5A 时间及模拟来源。** 原 pickle、原绘图脚本和 PNG 已找到，但需要当时的 `--x-scale`、截取参数、实际模拟主程序和输入记录。文件名参数不是执行日志；`index2` 也尚不能当成 repeat ID。
- [ ] **Fig. 5A 零点。** SI 搜索 `Source-scaled elapsed ms after subtraction of the original 620 ms offset.`：现行图仍显示 620–690 ms。请确定保留原时间还是减去 620，并使图和表一致。
- [ ] **Fig. 5D 结构映射。** 将图片逐一对应 `entity_N/unique_structure_ID.mol2`；数值百分比已重算对上。确认跨 71 条件、284 行记录的筛选标准、重复数和缺失值处理，不写成新扫描概率。
- [ ] **S21–S22 混合种子。** 本包是 `...regenerated...rerun_20260410` 的四次重复。确认现稿图采用的是此再生成家族还是更早的原始种子；不要把两者当同一原始运行。
- [ ] **S23–S24 修复。** 确认对应同一条连续轨迹及原始空位分母 591/54，而非新 5090 ensemble。初始、N=2800 与 N=100001 的图像来源分别登记；大终态 checkpoint 未随轻量包发送。

## 需从新版计算负责人补齐，不在本包中

- [ ] `simulation_review_gate_20260831/code/proposed/` 下的实际执行代码及依赖，尤其 `common_kmc_selector.py`、`common_molecular_runner.py`。
- [ ] `common_response_surface_v2_final_condition_summary.csv`、`...trace_metrics.csv`、`...pooled_committor.csv` 及新版 Fig. 5B–C 方差分析/相界绘图代码。
- [ ] `seeded_propagation_v1_final_*`、`seeded_molecular_runner.py`、`surface_secondary_nucleation_runner.py`、`figure5_surface_vs_seeded_growth_sc1_v1_*`，对应主文 Fig. 4。
- [ ] `finite_bath_population_runner.py`、`chemistry_mapping.py`、`figure8_absolute_outcomes_v1_*`，对应主文 Fig. 8。
- [ ] UiO-66 N=10000 / BTB N=2000 的结构与 PXRD、explicit-cap geometry、defect-intermediate 新 ensemble 的启动参数、run ID、续跑账本、截尾和占位分析来源。
- [ ] 当前 `panel_source_correspondence.csv`、`LEGACY_PANEL_CORRESPONDENCE.md`、`all_catalogued_calculation_families.csv`、`kmc_trace_catalog_20260827`，把本包恢复的历史映射合进去。
- [ ] Fig. 7 / S42–S45 的 HKUST-1、PCN-240 精确曲线、结构频数及分母；S25 两条 illustrative repair 曲线的确切来源。这里只确认有历史代码，未完成图级对应，不标为已核实。
- [ ] `Table_S5_1_literature_catalogue.xlsx` 及来源证据，单独归档。

每个新家族请保留：执行源代码哈希、参数、seed/host 结构哈希、原始 run ID、停止标准、时间定义、排除规则和绘图命令；续跑不能算额外独立重复。旧服务器归档不自动等于新版 5090 计算目录。

## 文稿编辑者最后处理

- [ ] MS 搜索 `growing one N = 10,000 product`，与 SI Table S7 的 rho_off=30 终止于 N=2922 核对，避免写成全部条件都到 10000。
- [ ] SI 两个 `S4.4` 标题（Size-resolved finite cluster PXRD / Established-seed growth）在最终视图里都存在，需统一编号和引用。
- [ ] 数据来源确认后再由一个编辑副本统一 MS/SI 时间标签、caption 和 cross-reference，避免多人同时改同一文档。

确认时可逐项填：结论、对应路径/文件名、采用或不采用、负责人和日期。不确定项保持待核实，不用本包的重画预览替代最终图源证据。
