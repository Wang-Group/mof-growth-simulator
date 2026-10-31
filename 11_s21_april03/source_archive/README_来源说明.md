# S21 轨迹来源副本（2026-10-30）

## 当前 SI 曲线对应的版本

匹配版本来自 `Zr6_Zr12_mix`，不是 `initialize_structures1`。两处文件名及 RNG 编号相同，但轨迹数据不同。

原始轨迹路径：
`D:\SYH\课题\MOL生长模拟\论文\data1\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\Zr6_Zr12_mix\output\mixed_nuclei\zr6_only_growth_runs\oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned\oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned.summary_trace.json`

SHA-256：`d7be8f68b070266eb327d84e2d62d3cb00b7cb4cfa4008a6652b29e9e7391a89`；大小：4110673 字节。
4 次重复的随机种子为 2026040311–2026040314；均值采用共同时间网格 400 点、前值保持取样、4 条轨迹平均。到达终点的轨迹在后续网格保持末值，未删除已结束重复。用原代码逐项复算，均值表最大绝对差为 0。
Zr12 均值首次为零：0.701326598313 s。

`original_artwork/*.replot.svg` 和 `*.replot.png` 为 D 盘原有成图的原样副本，与 20261019 SI S21 的三条均值曲线、细轨迹和归零线吻合。SI 的标题和轴标经过后续编辑；不声称整张图片文件逐字节相同。

## 文件分类和运行

- `source_data`：保存的逐事件回放轨迹 JSON、同目录汇总及重复记录；`mean_trace_checked.csv` 为本次直接从 JSON 导出并复算的均值，不是从图片反推。
- `original_artwork`：原目录已有的 SVG、PNG，未改动。
- `plot_code/plot_zr6_only_growth_trace.py`：原模拟回放/绘图脚本原样保留。它会重新运行模拟并依赖外部模拟模块，不宜用重新回放结果冒充已存轨迹。
- `plot_code/replot_saved_trace.py`：本次新增的纯绘图复核入口，只读保存的 JSON，不重新运行模拟。依赖 numpy、matplotlib；运行 `python plot_code/replot_saved_trace.py`，输出在 `redrawn_verification`。它复现曲线，版式不是作者手工编辑后母版的逐像素复制。
- `verification`：均值复算记录、原文件路径及 SHA-256 清单、Word 图片参考。
- `other_versions/initialize_structures1`：按用户请求保留此前 C 盘副本及复核图；此 JSON 的 SHA-256 为 F08AD2680B27BF1F337AA424DE66ABB3279D14BBF1E650BC5CDDE03CB69FCA03。它与 D 盘 `initialize_structures1` 原件相同，但不与本 SI 曲线一致，不得与主目录匹配版本混用。此前审计报告将其写成 SI 图源，现予更正。

本次只新增本目录；未覆盖、删除原有 S21 包文件，未改 D 盘原件或 Word。
