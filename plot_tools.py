from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from random_variable import FiniteDist

# 网格线1 < 填充2 < 横线3 < 竖线4 < 曲线5 < 标记点6 < 右侧文字7 < 分位数文字8 < 期望文字9 < 众数文字10 < 水印11

# z_order 常量定义
Z_ORDER_GRID = 1
Z_ORDER_FILL = 2
Z_ORDER_HLINE = 3
Z_ORDER_VLINE = 4
Z_ORDER_CURVE = 5
Z_ORDER_MARKER = 6
Z_ORDER_TEXT_RIGHT = 7
Z_ORDER_TEXT_QUANTILE = 8
Z_ORDER_TEXT_EXPECT = 9
Z_ORDER_TEXT_MODE = 10
Z_ORDER_WATERMARK = 11

# matplotlib 绘图设置
plt.rcParams["font.sans-serif"] = ["Source Han Sans SC"]
# plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["figure.dpi"] = 300
# # 默认添加 minor tick
# plt.rcParams["xtick.minor.visible"] = True
# plt.rcParams["ytick.minor.visible"] = True
# # 默认添加 major 和 minor 的网格
# plt.rcParams["axes.grid"] = True
# plt.rcParams["axes.grid.which"] = "both"

# 描边预设
stroke_white = [pe.withStroke(linewidth=2.5, foreground="white")]
stroke_black = [pe.withStroke(linewidth=2.5, foreground="black")]

suptitle_text = "明日方舟寻访机制解析　bilibili@Bio-Hazard"
watermark_text = "bilibili@Bio-Hazard\n森空岛@BioHazard\nNGA@Bio-Hazard"


def make_valid_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", s).replace("\n", "")


# def calc_quantile_point(cdf, quantile_p):
#     """返回分位点位置"""
#     return np.searchsorted(cdf, quantile_p, side="left")


def draw_pmf_cdf_fig(
    *,
    dist: FiniteDist,
    title: str,
    x_max: float | None | Literal["auto"],
    quantile_poses: Sequence[float] | None = None,
    save: bool = False,
    times_text: str = "次",
    x_label_text: str = "寻访次数",
) -> tuple[Figure, tuple[Axes, Axes]]:
    if quantile_poses is None:
        quantile_poses = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ax1_y_top = np.max(dist.pk) * 1.27
    ax2_y_top = 1.11
    x = np.arange(len(dist.pk))
    if x_max == "auto":
        x_max = dist.ppf(0.99) * 1.25
    quantile_points = dist.ppf(quantile_poses).astype(int)

    fig = plt.figure(figsize=(10, 10), layout="tight")
    ax1, ax2 = fig.subplots(2, 1)
    ax1: Axes
    ax2: Axes

    # 绘制 PMF 和 CDF，并在曲线下方填充颜色
    ax1.fill_between(x, dist.pk, step="mid", alpha=0.3, zorder=Z_ORDER_FILL)
    ax1.plot(
        x,
        dist.pk,
        drawstyle="steps-mid",
        path_effects=stroke_white,
        zorder=Z_ORDER_CURVE,
    )
    ax2.fill_between(x, dist.cdf(x), step="mid", alpha=0.3, zorder=Z_ORDER_FILL)
    ax2.plot(
        x,
        dist.cdf(x),
        drawstyle="steps-mid",
        path_effects=stroke_white,
        zorder=Z_ORDER_CURVE,
    )

    # 期望值
    ax1.axvline(dist.expect(), linestyle="--", color="C1", zorder=Z_ORDER_VLINE)
    ax1.annotate(
        f"期望\n{dist.expect():.2f}{times_text}",
        (dist.expect(), ax1_y_top),
        ha="center",
        va="top",
        xytext=(0, -5),
        textcoords="offset points",
        color="C1",
        fontweight="medium",
        path_effects=stroke_white,
        zorder=Z_ORDER_TEXT_EXPECT,
    )

    # 分位数
    [
        ax1.axvline(quantile_point, color="gray", linestyle="--", zorder=Z_ORDER_VLINE)
        for quantile_point in quantile_points
    ]
    [
        ax1.annotate(
            f"{quantile_point}{times_text}\n{dist.cdf(quantile_point):.0%}",
            (quantile_point, ax1_y_top),
            ha="center",
            va="top",
            xytext=(0, -5),
            textcoords="offset points",
            color="gray",
            fontweight="medium",
            path_effects=stroke_white,
            zorder=Z_ORDER_TEXT_QUANTILE,
        )
        for quantile_point in quantile_points
    ]

    # 众数
    ax1.scatter(
        np.argmax(dist.pk),
        np.max(dist.pk),
        color="C1",
        marker=".",
        path_effects=stroke_white,
        zorder=Z_ORDER_MARKER,
    )
    ax1.annotate(
        f"最可能\n{np.argmax(dist.pk)}{times_text}",
        (np.argmax(dist.pk), np.max(dist.pk)),
        ha="center",
        va="bottom",
        xytext=(0, 3),
        textcoords="offset points",
        color="C1",
        fontweight="medium",
        path_effects=stroke_white,
        zorder=Z_ORDER_TEXT_MODE,
    )

    # 期望值
    ax2.axvline(dist.expect(), linestyle="--", color="C1", zorder=Z_ORDER_VLINE)
    ax2.annotate(
        f"期望\n{dist.expect():.2f}{times_text}",
        (dist.expect(), ax2_y_top),
        ha="center",
        va="top",
        xytext=(0, -5),
        textcoords="offset points",
        color="C1",
        fontweight="medium",
        path_effects=stroke_white,
        zorder=Z_ORDER_TEXT_EXPECT,
    )

    # 分位数
    ax2.scatter(
        quantile_points,
        dist.cdf(quantile_points),
        color="C0",
        marker=".",
        path_effects=stroke_white,
        zorder=Z_ORDER_MARKER,
    )
    [
        ax2.annotate(
            f"{quantile_point}{times_text}\n{dist.cdf(quantile_point):.0%}",
            (quantile_point, dist.cdf(quantile_point)),
            ha="right",
            va="baseline",
            xytext=(-5, 5),
            textcoords="offset points",
            color="gray",
            fontweight="medium",
            path_effects=stroke_white,
            zorder=Z_ORDER_TEXT_QUANTILE,
        )
        for quantile_point in quantile_points
    ]

    # 图表标题、坐标轴标签
    fig.suptitle(
        f"{suptitle_text}\n{title}",
        fontweight="bold",
        fontsize="x-large",
    )
    ax1.set_title("概率质量函数", fontweight="bold")
    ax2.set_title("累积分布函数", fontweight="bold")
    ax1.set_ylabel(f"本{times_text}概率", fontweight="bold")
    ax2.set_ylabel("累积概率", fontweight="bold")
    ax2.set_xlabel(x_label_text, fontweight="bold")

    for ax in (ax1, ax2):
        # 绘制水印
        ax.text(
            0.98,
            0.02,
            watermark_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="medium",
            color="gray",
            alpha=0.8,
            path_effects=stroke_white,
            zorder=Z_ORDER_WATERMARK,
        )

        # 显示网格
        ax.minorticks_on()
        ax.grid(True, which="major", linewidth=1.2)
        ax.grid(True, which="minor", linewidth=0.6)

        # 坐标轴范围
        ax.set_xlim(0, x_max)

        # y 轴标签显示为百分比
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax1.set_ylim(0, ax1_y_top)
    ax2.set_ylim(0, ax2_y_top)

    if save:
        file_name = make_valid_filename(f"{title}.png")
        fig.savefig(f"图片/{file_name}", dpi=300)

    return fig, (ax1, ax2)


def draw_multi_pmf_cdf_fig(
    dists: Sequence[FiniteDist],
    title: str,
    labels: Sequence[str],
    *,
    x_max: float | Literal["auto"],
    quantile_poses: Sequence[float] | None = None,
    ax1_y_top: float | None = None,
    save: bool = False,
    times_text: str = "次",
    x_label_text: str = "寻访次数",
) -> tuple[Figure, tuple[Axes, Axes]]:
    if quantile_poses is None:
        quantile_poses = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    colors = [f"C{i}" for i in range(len(dists))]
    if ax1_y_top is None:
        ax1_y_top = np.max(dists[1].pk) * 1.27
    ax2_y_top = 1.15
    if x_max == "auto":
        x_max = dists[-1].ppf(0.99) * 1.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), layout="tight")
    ax1: Axes
    ax2: Axes

    for dist, label, color in zip(dists, labels, colors, strict=True):
        x = np.arange(len(dist.pk))
        ax1.fill_between(x, dist.pk, step="mid", alpha=0.3, zorder=Z_ORDER_FILL)
        ax1.plot(
            x,
            dist.pk,
            label=label,
            color=color,
            drawstyle="steps-mid",
            path_effects=stroke_white,
            zorder=Z_ORDER_CURVE,
        )
        # ax2.fill_between(x, dist.cdf(x), step="mid", alpha=0.3, zorder=3)
        ax2.plot(
            x,
            dist.cdf(x),
            label=label,
            color=color,
            drawstyle="steps-mid",
            path_effects=stroke_white,
            zorder=Z_ORDER_CURVE,
        )

        # 期望值
        ax1.axvline(dist.expect(), linestyle="--", color=color, zorder=Z_ORDER_VLINE)
        ax1.annotate(
            f"期望\n{dist.expect():.2f}{times_text}",
            (dist.expect(), ax1_y_top),
            ha="center",
            va="top",
            xytext=(0, -5),
            textcoords="offset points",
            color=color,
            fontweight="medium",
            path_effects=stroke_white,
            zorder=Z_ORDER_TEXT_EXPECT,
        )
        ax2.axvline(dist.expect(), linestyle="--", color=color, zorder=Z_ORDER_VLINE)
        ax2.annotate(
            f"期望\n{dist.expect():.2f}{times_text}",
            (dist.expect(), ax2_y_top),
            ha="center",
            va="top",
            xytext=(0, -5),
            textcoords="offset points",
            color=color,
            fontweight="medium",
            path_effects=stroke_white,
            zorder=Z_ORDER_TEXT_EXPECT,
        )

        # 众数
        # ax1.scatter(
        #     np.argmax(dist.pk),
        #     np.max(dist.pk),
        #     s=10,
        #     color=color,
        #     marker=".",
        #     path_effects=stroke_white,
        #     zorder=Z_ORDER_MARKER,
        # )
        # ax1.annotate(
        #     f"最可能\n{np.argmax(dist.pk)}{times_text}",
        #     (np.argmax(dist.pk), np.max(dist.pk)),
        #     ha="center",
        #     va="bottom",
        #     xytext=(0, 3),
        #     textcoords="offset points",
        #     color=color,
        #     fontweight="medium",
        #     path_effects=stroke_white,
        #     zorder=Z_ORDER_TEXT_MODE,
        # )

        # 分位数
        quantile_points = dist.ppf(quantile_poses).astype(int)
        ax2.scatter(
            quantile_points,
            dist.cdf(quantile_points),
            s=10,
            color=color,
            marker=".",
            path_effects=stroke_white,
            zorder=Z_ORDER_MARKER,
        )
        [
            ax2.annotate(
                quantile_point,
                (quantile_point, quantile_pos),
                ha="right",
                va="baseline",
                xytext=(-2, 2),
                textcoords="offset points",
                color="gray",
                fontweight="medium",
                path_effects=stroke_white,
                zorder=Z_ORDER_TEXT_QUANTILE,
            )
            for quantile_point, quantile_pos in zip(
                quantile_points, quantile_poses, strict=True
            )
        ]
        [
            ax2.axhline(
                quantile_pos, color="gray", linestyle="--", zorder=Z_ORDER_HLINE
            )
            for quantile_pos in quantile_poses
        ]
        [
            ax2.annotate(
                f"{quantile_pos:.0%}",
                (x_max, quantile_pos),
                ha="right",
                va="center",
                xytext=(-5, 0),
                textcoords="offset points",
                color="gray",
                fontweight="medium",
                path_effects=stroke_white,
                zorder=Z_ORDER_TEXT_RIGHT,
            )
            for quantile_pos in quantile_poses
        ]

    # 图表标题
    fig.suptitle(
        f"{suptitle_text}\n{title}",
        fontweight="bold",
        fontsize="x-large",
    )
    ax1.set_title("概率质量函数", fontweight="bold")
    ax2.set_title("累积分布函数", fontweight="bold")
    ax1.set_ylabel(f"本{times_text}概率", fontweight="bold")
    ax2.set_ylabel("累积概率", fontweight="bold")
    ax2.set_xlabel(x_label_text, fontweight="bold")

    for ax in (ax1, ax2):
        # 绘制水印
        ax.text(
            0.98,
            0.02,
            watermark_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="medium",
            color="gray",
            alpha=0.8,
            path_effects=stroke_white,
            zorder=Z_ORDER_WATERMARK,
        )

        # 显示网格
        ax.minorticks_on()
        ax.grid(True, which="major", linewidth=1.2)
        ax.grid(True, which="minor", linewidth=0.6)

        # 坐标轴范围
        ax.set_xlim(0, x_max)

        # y 轴标签显示为百分比
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 坐标轴范围
    ax1.set_ylim(0, ax1_y_top)
    ax2.set_ylim(0, ax2_y_top)

    # 添加图例
    ax1.legend()

    if save:
        file_name = make_valid_filename(f"{title}.png")
        fig.savefig(f"图片/{file_name}", dpi=300)

    return fig, (ax1, ax2)


def _pct_str(q: float) -> str:
    """格式化百分比，如 0.25 -> '25%', 0.9999 -> '99.99%'"""
    pct = q * 100
    if pct == pct // 1:
        return f"{int(pct)}%"
    return f"{pct}%"


_DEFAULT_QUANTILES: list[tuple[float, str]] = [
    (0.25, "较欧"),
    (0.50, "凡人"),
    (0.75, "较非"),
    (0.90, "非酋"),
    (0.95, "很非"),
    (0.99, "极非"),
    (0.9999, "万里挑一的非"),
]


def print_distribution_table(
    dists: list[FiniteDist],
    labels: list[str],
    *,
    quantiles: list[tuple[float, str]] | None = None,
    expected_precision: int = 2,
) -> pd.DataFrame:
    """打印多个分布的期望和分位数表格（含趣味标签）。

    Parameters
    ----------
    dists : list[FiniteDist]
        分布列表。
    labels : list[str]
        每个分布对应的标签（如 "抽1个", "抽2个"）。
    quantiles : list[tuple[float, str]] | None
        分位数配置列表，每项为 (概率, 趣味文本)。
        默认为 [(0.25, "较欧"), (0.50, "凡人"), ...]。
    expected_precision : int
        期望值保留的小数位数。

    Returns
    -------
    pd.DataFrame
        生成的表格 DataFrame（三层列索引）。
    """
    if quantiles is None:
        quantiles = _DEFAULT_QUANTILES

    data: list[dict] = []
    for dist, label in zip(dists, labels, strict=True):
        row: dict = {
            "标签": label,
            "期望抽数": round(dist.expect(), expected_precision),
        }
        for q, _ in quantiles:
            row[_pct_str(q)] = int(dist.ppf(q))
        data.append(row)

    df = pd.DataFrame(data).set_index("标签")

    # 三层列索引：第一层为类别，第二层为百分比，第三层为趣味文本
    df.columns = pd.MultiIndex.from_tuples(
        [("期望抽数", "", "")]
        + [("达成指定概率所需抽数", _pct_str(q), label) for q, label in quantiles]
    )

    return df
