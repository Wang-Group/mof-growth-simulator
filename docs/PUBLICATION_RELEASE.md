# 投稿代码与数据发布清单

发布状态更新：2026-11-01，S1 的 69/53、S2 的 23/30 已确认，S8 已销项；S6 代码、构型及安全复算入口已入包。
当前状态为**已推送GitHub的候选分支，不是整篇论文已经完全可复现的发布版**。
MS/SI 没有修改，也没有放进代码包。整篇审查仍沿用 10 月 29 日最终视图基准；10 月 31 日仅补读两份 SI 的
S1/S2/S6/S8（含 10 月 31 日 text_synced 版），四张表在该历史基准中一致，哈希另存于 `13`。
11 月 1 日作者确认保留已核对文献并更新正文；再次只读核对新版 MS/SI，S2 均为 30 篇中 23 篇变小/变薄，差异已关闭；
新版哈希及 NAS 熵计算核验记录见 `metadata/publication_decisions_20261031.json`，不覆盖旧来源文件。
该基准的 MS 为 Fig. 1-7；程序中遗留的 Figure 6/7/8 文件名不能直接当作现稿图号。
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
| 熵计算来源 | `08_entropy_inputs/`、`13_table_source_audit/` | 13个原表不变；新增6份代码/notebook及33个构型，六物种映射和18值可离线复算；notebook输出已清除而代码单元保留 |
| 5090 修正版引擎和模型 | `09_recovered_runtime/` | 两个引擎、八个模型均与历史索引哈希一致；六条 UiO-66 结果的 engine/runner/selector 共 18 项引用全部对应包内确切版本 |
| Fig. 6 / S7 绘图版本 | `10_plot_versions/` | 历史原版与共享目录修订版分开保留；三个旧哈希差异已查明，不再视为代码丢失 |
| S21 April 3 轨迹 | `11_s21_april03/` | 主版本及 initialize_structures1 备选来源均保留；和 `05` 中 April 10 重跑分开；三组均值分别复算一致 |
| seed / RNG / 代码版本 | `12_runtime_provenance/` | 206 个来源位置、197 项精确引用；24 个 frozen-v2 seed 及 reviewed seed/host，六条 S7 和四条历史抽样结果与参数表对应 |
| checkpoint 续跑关系 | `12_runtime_provenance/checkpoint_lineage.csv` | 943 条续跑和 384 个配对分支对应 975 个独立 checkpoint/result 对，远端哈希均一致；配对起点只有 32 个粒子 |
| S1 原始分类证据 | `13_table_source_audit/s1_original/` | 作者核过文献并确认采用69/53；旧分组统计保留为历史来源，不再作为待改计数 |

核查结果见 [publication_verification.json](../publication_verification.json)。
`PASS_FOR_DECLARED_LOCAL_CHECKS` 仅针对列出的本地检查；`publication_ready` 保持 `false`。

## 已解决的来源缺口

**Table S1 的 69/53 已确认。** 用户明确表示已核对文献且无误；正文口径保留，不再重复索取确认。
本次记录作者决定，不声称独立复核了全部文献或重建了方向分类。

**Table S2 的 23/30 已确认且正文与 SI 一致。** 保留作者确认纳入的文献，不再自动剔除或改回22。
包内31篇提取仅为历史基准；最新正文哈希及采用口径已登记，不将文稿一致性冒称为逐篇独立复核。

**Table S8 已关闭。** 作者确认现有 12 行表格就是原始时间记录，用户已接受；本次不再要求额外
观测记录或另做 Excel。数据与分钟单位不变，不把作者确认写成独立实验复测，也不与 DLS 混用。

5090 主机及 `/home/wave/MOL_growth_recalculation_20260830/` 已确认。UiO-66 引擎
SHA-256 为 `4cda9182267ef4fc29a5537a60c23d0fd592b27c92221c7dc1b7731b291e24e9`，
现已收录。两个引擎及八个模型原字节保存在 `09`；账号凭据不进入发布内容。

三个脚本差异现记为 `VERSION_VARIANT_ARCHIVED`，不再要求寻找已经存在的旧版。
Fig. 6 builder 仅把散点透明度 0.20 改为 0.55；其 runtime 增加参考版式辅助代码；
S7 runtime 改为调用秒轴绘图器。原版和修改版均保留，详见 [版本说明](../10_plot_versions/README.md)。
Fig. 6 的采用文件仍待登记；用户随后已确认 S7 发布使用算法时间 `u`。
正式入口为 `python 10_plot_versions/render_s7.py`，秒轴 runtime 仅保留为历史档案。

新增 S21 文件夹的说明将 April 3 `Zr6_Zr12_mix` 版本对应到 S21。
本次独立核对其哈希和均值计算，但未重新比对当前 Word 图片；另两个来源不能混称为同一次运行。

本轮约 5.4 MiB 的新增源文件保留 seed、host、参数 manifest 与精确代码版本，重复文件按哈希引用，
大型重复文本无损 gzip 保存。975 个 checkpoint 和对应 result 共 545,387,724 字节仍留远端，
发布包只保存本次只读哈希证据，不反序列化 pickle。旧 SC1 runner 的 `d4824c72...` 版本从备份找回，
与 S7 的 `b5354813...` 分开保存。固定步数、size-followon、配对响应的终点定义和重复数不能混用。

## 尚未找到或未完整核实

当前工作以 [用户确认清单](../YUHANG_CONFIRMATION_CHECKLIST.md) 为准，主要是来源核对和投稿收尾，
不等于数据有错。以下条目区分底层文件尚未定位、对应关系未闭合和作者版本登记。

1. **Fig. 5C 原始数值分析脚本。** `analyze_figure6_current_parameter_phase_v1.py`
   11月1日扩查共享项目28,395个文件、5个ZIP目录，以及5090四个相关项目、29个归档包，仍未找到。
   具体代码读取条件、排除项及报告见 `14_fig5_source_followup/search_evidence/`；不表示搜过所有电脑。
   共享盘已有绘图脚本和数据；还需该底层模块、插值/筛选规则及实际命令，不能把它写成整张图的数据缺失。
2. **运行覆盖与公开归档。** 本轮列出的 seed/代码/续跑输入链已对应；未证明每个设计表中的所有任务
   均已完成，也未收齐所有家族的全部事件历史和终态。checkpoint 已定位不等于已公开或已完整重跑。
3. **其他表格审计范围。** S1/S2 计数、S8 来源及 S6 打包复算已完成，不再重复索取。历史 S4 等审计细节保留在
   [表格限制清单](../metadata/table_source_limits.csv)，不在本次自动认定已解决或重新复算。
4. **Fig. 5D 与最终版本登记。** Fig. 5D 百分比已独立复算，结构按用户最新确认已核实；
   `14_fig5_source_followup/` 新增分类notebook、排版代码、9个大结构和284行证据表。
   8行只有大结构参数匹配，4行涉及3条轨迹的文件/目录编号冲突，272行仍无匹配证据。
   大结构不能替代原始小entity快照，全部分类行仍未闭合；需原MOL2目录或逐文件分配记录。
   Fig. 6 需登记最终采用文件与哈希。
   其他历史 cohort/图像映射记录保留作参考，本次未重新审查最新 Word。

## 作者需要统一的科学口径

- **时钟和图号：** 24 条轨迹历史表的 `_s` 为 raw u × 0.175；Fig. 5A 的数值绘图约定为 raw timing × 0.2405。找回缩放不等于验证物理标定。沿用 SI Table S27 和各 caption 的限定，最后逐图核对，不自动重画或改单位。
- **S7 与 S21 版本：** S7 正式发布固定为算法时间 `u`，秒轴只保留历史；CSV 已乘 0.175，恢复 `u` 时除以 0.175。S21 的三个来源有不同均值归零时间，不能合并或相互改名替代。
- **Table S7 的最大值与终点：** rho_off=30 的结果是 maximum_entities=2922、final_state.entities=2911；SI 将 2922 写作 endpoint，需要作者确认。同时不能把到达 N=10000 写成所有条件均成功。
- **结构/频率来源：** 24 条 entity-13 最后保存结构不是首次通过结构，也不是历史 20-run cohort；SI 建议说明 9 类及 I03 例外。BTB 历史 motif 频率不等于新响应面概率；Fig. 5D 当前缺口是分类行到原始轨迹的对应，而不是已确认的百分比和结构。
- **实验/熵表格：** DLS 拟合对象是处理后的 apparent progress，范围 117-1170 s，展示可为 0-600 s。原 NMR 采集候选不等于线宽逐行拟合来源。S6 源文件已入包，离线复算沿用原始模型和舍入输入，不等于独立验证全部物理近似。具体限制见 [table_source_limits.csv](../metadata/table_source_limits.csv)。

上述是待确认清单，不是在本次替作者修改 MS/SI，也不意味着所有现稿表述都未加限定。

## 发布门槛与交付

- [x] 收齐此次列出的两个修正版引擎、八个模型及三个绘图版本差异的原版/修订版。
- [x] 本轮列出的 197 项代码/seed 引用及 975 个 checkpoint/result 输入对已对应。
- [ ] 选定外部 checkpoint 公开范围，补 Fig. 5C 原分析版本及剩余表格来源，或明确缩小公开可复现范围。
- [ ] 作者确认上述时间、终点、cohort、最终图号和结构映射；由单一文稿编辑副本同步 MS/SI。
- [ ] 确认代码与数据的分别授权、贡献者名单、引用方式；再添加 LICENSE/CITATION，不擅自指定许可或署名。
- [ ] 按 [Zenodo 计划](ZENODO_DATA_PLAN.md) 选择必要大文件及校验清单，不上传私钥、账号配置或整台服务器备份。
- [ ] 候选分支已推送 GitHub（初次发布提交 `53b258e`）；尚未建立正式 Release/tag、上传 Zenodo 或生成 DOI。最终确认后固定版本并交叉引用。

日常复现入口仍是根目录 `reproduce.py`。07 目录是来源档案，不承诺所有历史脚本均可直接运行。
原 `main` 和 `data/verified-contributions-20261024` 分支保留；本候选分支不应覆盖历史仓库。
