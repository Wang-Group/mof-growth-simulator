# Figure 1B C E 原代码与数据追溯

2026-10-24核对：原数值绘图脚本仍未找到，不能称这三图的代码已补齐。

当前正文B是Framework families，C是Zr/Hf dose studies，E是XRD and DLS size；早期B/C顺序相反。

- B：原分类表和汇总已找回，可重新算出53/6/4/2/2/1/1。找到的build_seed_manifest.py是上游统计代码，已另存为B_upstream_statistics_NOT_plotting.py，并非绘图入口。
- C：当前SI表S1写有53、40/12/1；已保存展示表。最终53篇的逐篇方向标签仍未唯一闭合，不能将已有筛选表冒充已核实最终分类表。
- E：Schaate2011 DOI 10.1002/chem.201003211，Table1。BA/Zr=0/1/3/5/10；PXRD=85/39/67/95/120 nm；水中DLS分布最大值=220/255/164/164/105 nm。数据来自保存的文本表格记录，没有从图像或SVG坐标反算。20/30当量的PXRD超出原作者Scherrer方法范围，本图未纳入。

这些是文献统计/原文发表数值，不能称为本研究原始实验仪器数据。完整原始文件绝对路径和SHA256见source_provenance.json。

注意：原资料COHORT_CORRECTIONS.md明确记载69个DOI记录仅对应68个独立出版物组（同篇文章德文与国际版）；其中Zr/Hf为53条DOI记录、52组出版物。为保存当前正文对应，本次没有擅自更改图中数量。文章若仍称69篇独立文献，需要作者复核计数口径。

检索范围包括D盘论文目录中的脚本、notebook代码单元、0915/0906/0926交接ZIP及本机历史会话。仅找到旧build_figure1_panel_bundle_v6/v7，它们读取已有SVG并分离图组，不是数值绘图源代码。本次按原代码要求未以补写代码充数。
