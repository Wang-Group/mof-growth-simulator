import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


def read_pkl_file(file_path):
    """读取单个 pkl 文件。"""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def add_break_marks(ax, side, size=9):
    """在坐标轴左侧或右侧添加斜线断轴标记。"""
    x_position = 0 if side == "left" else 1

    # 使用点大小控制标记，避免窄坐标轴导致斜线变形
    ax.plot(
        [x_position, x_position],
        [0, 1],
        transform=ax.transAxes,
        marker=[(-1, -0.6), (1, 0.6)],
        markersize=size,
        linestyle="none",
        color="black",
        markeredgewidth=1,
        clip_on=False,
    )


def plot_and_save(
    data,
    file_name,
    save_dir,
    x_range=None,
    y_range=None,
    figsize=(11, 6),
    dpi=300,
    x_label="Growth Time",
    y_label="Number of Entities",
    zero_fraction=0.04,
    output_name=None,
    y_ticks=None,
):
    """
    绘制单文件数据，支持横坐标断轴。

    x_range:
        主绘图区的 (最小值, 最大值)。
        最小值 > 0 时，自动在左侧增加零点区间和断轴标记。
        None 表示自动显示全部数据，不使用断轴。

    y_range:
        Y 轴范围；None 表示自动设置。

    figsize:
        图片 (宽, 高)，单位为英寸。

    dpi:
        输出分辨率，像素尺寸 = figsize × dpi。
    """
    values = np.asarray(data, dtype=float)

    if values.size == 0:
        raise ValueError("pkl 数据为空。")

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("数据格式应为 [(time, count), ...]。")

    if not np.isfinite(values).all():
        raise ValueError("数据包含 NaN 或无穷大，请先检查数据。")

    x = values[:, 0] * 0.2405
    y = values[:, 1]

    if x_range is not None:
        x_min, x_max = map(float, x_range)
        if not np.isfinite([x_min, x_max]).all() or x_min >= x_max:
            raise ValueError("X 轴范围必须为有限数，且最小值小于最大值。")

    if y_range is not None:
        y_min, y_max = map(float, y_range)
        if not np.isfinite([y_min, y_max]).all() or y_min >= y_max:
            raise ValueError("Y 轴范围必须为有限数，且最小值小于最大值。")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    use_break = x_range is not None and x_min > 0

    if use_break:
        # 左侧窄区间保留零点，右侧显示用户指定的区间
        fig, (ax_left, ax_main) = plt.subplots(
            1,
            2,
            figsize=figsize,
            sharey=True,
            layout="constrained",
            gridspec_kw={
                "width_ratios": [0.05, 1],
                "wspace": 0.008,
            },
        )
        axes = (ax_left, ax_main)

        # 断轴前保留一小段真实曲线，但横坐标只标注 0，
        # 不标注断轴前的终点时间。
        zero_end = (x_max - x_min) * zero_fraction
        ax_left.set_xlim(0, zero_end)
        ax_left.set_xticks([0])

        # 断轴后先留一段可见的短轴线，再到第一个标注时间 x_min。
        x_padding = (x_max - x_min) * 0.04
        ax_main.set_xlim(x_min - x_padding, x_max)

        # 主区间刻度包含用户设置的最小值和最大值
        ax_main.set_xticks(np.linspace(x_min, x_max, 6))

        # 隐藏断轴处的竖直边框和右图的 Y 轴刻度
        ax_left.spines["right"].set_visible(False)
        ax_main.spines["left"].set_visible(False)
        ax_main.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

        add_break_marks(ax_left, "right")
        add_break_marks(ax_main, "left")

    else:
        fig, ax_main = plt.subplots(
            figsize=figsize,
            layout="constrained",
        )
        ax_left = ax_main
        axes = (ax_main,)

        if x_range is not None:
            ax_main.set_xlim(x_min, x_max)

    try:
        if use_break:
            # 左侧绘制起始阶段的真实曲线。
            left_visible = (x >= 0) & (x <= zero_end)
            ax_left.plot(
                x[left_visible],
                y[left_visible],
                color="#1f77b4",
                linewidth=0.6,
            )

            # 右侧留白段不绘制数据，曲线从第一个标注时间 x_min 开始。
            visible = (x >= x_min) & (x <= x_max)
            ax_main.plot(
                x[visible],
                y[visible],
                color="#1f77b4",
                linewidth=0.6,
            )
        else:
            ax_main.plot(
                x,
                y,
                color="#1f77b4",
                linewidth=0.6,
            )

        for ax in axes:
            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                alpha=0.25,
            )
            ax.tick_params(axis="both", labelsize=15)
            ax.ticklabel_format(
                axis="x",
                style="plain",
                useOffset=False,
            )

        if y_range is not None:
            ax_main.set_ylim(y_min, y_max)

        # 实体数量的纵坐标刻度只显示整数；可按原图指定刻度。
        if y_ticks is None:
            ax_left.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax_left.set_yticks(y_ticks)

        ax_left.set_ylabel(y_label, fontsize=15)
        fig.supxlabel(x_label, fontsize=15)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if output_name is None:
            output_name = f"{file_name}_broken_x_short_gap_v3.png"
        image_path = save_path / output_name

        if use_break:
            ax_left.xaxis.set_major_formatter(
                FuncFormatter(lambda value, pos: f"{value:g}")
            )
        # 不使用 bbox_inches="tight"，保证输出像素尺寸按 figsize × dpi
        fig.savefig(
            image_path,
            dpi=dpi,
            facecolor="white",
        )

        print(f"图片已保存：{image_path}")
        print(
            f"图片尺寸：{figsize[0]} × {figsize[1]} 英寸，"
            f"分辨率：{dpi} dpi"
        )

        if use_break:
            print(f"横坐标在 0 与 {x_min:g} 之间断开")

        return image_path

    finally:
        plt.close(fig)


def process_single_pkl(
    pkl_file_path,
    save_dir,
    **plot_options,
):
    """读取单个 pkl 文件并绘图。"""
    pkl_path = Path(pkl_file_path)
    data = read_pkl_file(pkl_path)
    print(f"成功读取文件：{pkl_path}")

    return plot_and_save(
        data=data,
        file_name=pkl_path.stem,
        save_dir=save_dir,
        **plot_options,
    )


if __name__ == "__main__":
    # ========== 自定义配置区 ==========
    # 本地 pkl 文件的绝对路径（示例，请替换为实际位置）
    PKL_FILE = Path(r"C:\Users\SYH\Desktop\entities_number_3124.8659233585645_index2.pkl")

    # 图片自动保存到 pkl 文件所在文件夹
    SAVE_DIR = Path(
        r"D:\SYH\课题\MOL生长模拟\论文\student_handoff_20260915_v5\MS-SI\Figure\Figure5"
    )
    # PKL_FILE = (
    #     "/mnt/syh/MOF_growth_data/Fe_Co3_BDC_OH/Fe_Co3_BDC_OH_260206/Stage_1/Zr_22.9154705859894_FA_200_L_34.57984361265395_Ratio_0_Step_1e15_Index_4_SC_0.789387907185137_KC_1.32975172557788_Nmax_500_2026-02-06_21_28_12/entities_number.pkl"
    # )

    # SAVE_DIR = "/mnt/syh/MOF_growth_data/Fe_Co3_BDC_OH/Fe_Co3_BDC_OH_260206/Stage_1/Zr_22.9154705859894_FA_200_L_34.57984361265395_Ratio_0_Step_1e15_Index_4_SC_0.789387907185137_KC_1.32975172557788_Nmax_500_2026-02-06_21_28_12/"

    # 右侧主绘图区范围：0 与 3365 之间显示断轴
    X_RANGE = (620, 690)

    # Y 轴范围
    Y_RANGE = (0, 20)

    # 图片尺寸：(宽, 高)，单位为英寸
    FIG_SIZE = (9, 6)

    # 分辨率：上述设置输出 2700 × 1800 像素
    DPI = 300

    # 时间单位需要根据模拟代码确认；确认是毫秒后可加上 (ms)
    X_LABEL = "Growth Time"
    Y_LABEL = "Number of Entities"

    try:
        process_single_pkl(
            pkl_file_path=PKL_FILE,
            save_dir=SAVE_DIR,
            x_range=X_RANGE,
            y_range=Y_RANGE,
            figsize=FIG_SIZE,
            dpi=DPI,
            x_label=X_LABEL,
            y_label=Y_LABEL,
        )
    except Exception as e:
        print(f"处理失败：{e}")
