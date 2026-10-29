# SI_Figure_S19 绘图入口

本目录已剔除误放的31份历史候选脚本。

运行：Python 3.10 或更高版本执行 `python run_panels.py`；只画一张使用 `python run_panels.py --panel A`。输出在本目录的 panel_A、panel_B。

- original_code：实际调用的原脚本，按字节保留。
- panel_runtime.py：限定本图分支的本地路径适配器，不是另建模拟模型。
- source_data：3份实际绘图输入，与已有原文件SHA256一致。属于统计汇总，不是逐事件全量原始记录。
- PANEL_SOURCE_MAP.json：每份输入的原始绝对路径、SHA256、子图函数映射。

历史脚本中的 Figure5、S27、S55 是旧编号。S17只调用 export_s27_standalone；S19只调用 panel_c/panel_d，不运行脚本的其他图。其他函数可能含图像处理代码，但当前分支只读取以上CSV，不读取图像。

依赖：numpy、pandas、matplotlib；S17历史模块还导入 scipy 和 Pillow。独立原脚本可能含旧绝对路径，请使用本目录入口，不直接执行原脚本。

人工修改的最终Word拼版不属于逐像素一致保证范围。
