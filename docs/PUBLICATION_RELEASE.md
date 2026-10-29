# 投稿代码与数据发布清单

核查日期：2026-10-29。当前状态为**本地候选版本，不是整篇论文已经完全可复现的发布版**。
MS/SI 没有修改，也没有放进代码包；核查使用含插入、不含删除的审阅最终视图。
当前 MS 为 Fig. 1-7；程序中遗留的 Figure 6/7/8 文件名不能直接当作现稿图号。
文档版本以 [文件哈希](../metadata/release_document_baseline.json) 为准，而不是文件名日期。

## 已经找回并核对的内容

| 内容 | 包内位置 | 本次确认到什么程度 |
|---|---|---|
| 24 条 UiO-66 long-growth、DLS、历史 BTB、混合种子及空位修复 | `01` 至 `06` 目录 | 保留原轻量包的数据、绘图入口和验证方法；不是重新运行全部 KMC |
| 当前图表整理包 | `07_publication_sources/` | 670 份小型源文件逐份校验；原目录、修订版本和来源限制均保留 |
| 新 common/seeded/surface/finite-bath 代码 | `07_publication_sources/SI_tables_sources_20261023/Table_S28/code/` | 已找到，不再列为全部缺失；6 条 UiO-66 结果中的 runner/selector 哈希均与所收代码一致 |
| Fig. 5A 的实际历史运行 | `03_btb_trajectory/source/` | 原 pickle 对应 Index_2；找回归档主程序及 Assembly，确认保存的是原始 timing，后续 plotter 乘 0.2405 |
| Fig. 5B 方差分解 | `07_publication_sources/MS_Figure_5/` | 从 1260 个条件、每条件 32 条记录的条件统计表独立重算一阶/总效应，吻合；没有重算 bootstrap 区间 |
| S25 两条修复曲线 | `07_publication_sources/SI_Figure_S25/` | CSV 与已找到原文件哈希相同，分别从 N=2124 到 3651 / 100001；不是两条相同终点的曲线 |
| 熵相关表格 | `08_entropy_inputs/` | 13 个派生表格按原字节提取；不能替代原 Gaussian 输入/输出或声称已逐值验证 Table S6 |

核查结果见 [publication_verification.json](../publication_verification.json)。
`PASS_FOR_DECLARED_LOCAL_CHECKS` 仅针对列出的本地检查；`publication_ready` 保持 `false`。

## 优先补齐：执行版本与上游输入

1. **新版几何引擎和模型文件。** Table S28 的 runner 引用了以下路径，不能拿旧版本同名或相似代码代替：

```text
code/candidate_corrected/UiO66_Assembly_Large_Correction_20250811_candidate.py
code/candidate_corrected/MOL_Assembly_Large_Correction_20250811_candidate.py
models/workstation_source_20260831/mol/Zr6_AA.mol
models/workstation_source_20260831/mol/Zr12_AA.mol
models/workstation_source_20260831/mol/Zr6.mol
models/workstation_source_20260831/mol/Zr12.mol
models/workstation_source_20260831/mol/BTBa.mol
models/workstation_source_20260831/mol/BTBb.mol
models/workstation_source_20260831/mol/BTC.mol
models/approved_derived/BDC_explicit_H.mol
```

6 条 UiO-66 结果共同记录的引擎 SHA-256 为
`4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9`，
本候选包尚无这个确切版本。还需保留用到的 seed/host/checkpoint 哈希和续跑规则。
找回文件后先核对，不需要立刻重跑大型计算。

**定位线索：** 新结果记录的运行根目录是 `/home/wave/MOL_growth_recalculation_20260830/`。
旧 `/public/home/yibinjiang/...` 在部分记录中只是被重定向的历史来源。
本次对旧服务器候选路径的只读检查未找到所需引擎，不能据此断言其他机器上不存在。
请计算负责人确认实际主机及原运行目录；“在某人的账号下”不足以确定文件所在机器。

2. **Fig. 5C 原分析版本。** 条件网格和相界相关导出已收录，但在本次检查的图表包及历史审计范围内，
尚未定位 `analyze_figure6_current_parameter_phase_v1.py` 原版。
SVG 的文字/布局修补脚本不能替代数值分析。需要相界插值/筛选规则、运行命令及实际分析脚本。

3. **三个文件的清单哈希已变化。** 以下文件没有混入 `COPY` 内容，记录于来源清单的
`HASH_MISMATCH_REVIEW_REQUIRED` 行。请确认是否是合法后续编辑，并给对应版本说明及新清单：

```text
MS_Figure_6/content_checked_20261023/original_code/build_figure7_kinetic_window_v2.py
MS_Figure_6/content_checked_20261023/panel_runtime.py
SI_Figure_S7/content_checked_20261023/panel_runtime.py
```

## 作者需要统一的科学口径

- **时钟和图号：** 24 条轨迹历史表的 `_s` 为 raw u × 0.175；Fig. 5A 的数值绘图约定为 raw timing × 0.2405。找回缩放不等于验证物理标定。沿用 SI Table S27 和各 caption 的限定，最后逐图核对，不自动重画或改单位。
- **Table S7 的最大值与终点：** rho_off=30 的结果是 maximum_entities=2922、final_state.entities=2911；SI 将 2922 写作 endpoint，需要作者确认。同时不能把到达 N=10000 写成所有条件均成功。
- **结构/频率来源：** 24 条 entity-13 最后保存结构不是首次通过结构，也不是历史 20-run cohort；BTB 历史 motif 频率不等于新响应面概率。最终结构图片仍需对上具体 MOL2 ID。
- **实验/熵表格：** DLS 拟合对象是处理后的 apparent progress，范围 117-1170 s，展示可为 0-600 s。原 NMR 采集候选不等于线宽逐行拟合来源，Gaussian 原始数值链仍需补齐。其他表格的具体限制见 [table_source_limits.csv](../metadata/table_source_limits.csv)。

上述是待确认清单，不是在本次替作者修改 MS/SI，也不意味着所有现稿表述都未加限定。

## 发布门槛与交付

- [ ] 收齐并校验实际执行的引擎、必要模型、seed/host 及原分析版本，或明确缩小公开可复现范围。
- [ ] 作者确认上述时间、终点、cohort、最终图号和结构映射；由单一文稿编辑副本同步 MS/SI。
- [ ] 确认代码与数据的分别授权、贡献者名单、引用方式；再添加 LICENSE/CITATION，不擅自指定许可或署名。
- [ ] 按 [Zenodo 计划](ZENODO_DATA_PLAN.md) 选择必要大文件及校验清单，不上传私钥、账号配置或整台服务器备份。
- [ ] 两个验证入口通过后，固定 Git commit/tag，并在批准的 GitHub/Zenodo 发布记录中交叉引用；目前尚未推送、上传或生成 DOI。

日常复现入口仍是根目录 `reproduce.py`。07 目录是来源档案，不承诺所有历史脚本均可直接运行。
原 `main` 和 `data/verified-contributions-20261024` 分支保留；本候选分支不应覆盖历史仓库。
