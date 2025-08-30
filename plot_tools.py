from collections.abc import Sequence
from typing import Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from random_variable import FiniteDist

# 网格线1 < 填充2 < 横线3 < 竖线4 < 曲线5 < 标记点6 < 右侧文字7 < 分位数文字8 < 期望文字9 < 众数文字10 < 水印11

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


# def calc_quantile_point(cdf, quantile_p):
#     """返回分位点位置"""
#     return np.searchsorted(cdf, quantile_p, side="left")


def draw_pmf_cdf_fig(dist: FiniteDist,
                     title: str,
                     *,
                     x_max: float | None | Literal["auto"],
                     quantile_poses: Sequence[float] | None = None,
                     save: bool = False) -> tuple[Figure, tuple[Axes, Axes]]:
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
    ax1: plt.Axes  # type: ignore
    ax2: plt.Axes  # type: ignore

    # 绘制 PMF 和 CDF，并在曲线下方填充颜色
    ax1.fill_between(x, dist.pk, step="mid", alpha=0.3, zorder=2)
    ax1.plot(x, dist.pk, drawstyle="steps-mid", path_effects=stroke_white, zorder=5)
    ax2.fill_between(x, dist.cdf(x), step="mid", alpha=0.3, zorder=2)
    ax2.plot(x, dist.cdf(x), drawstyle="steps-mid", path_effects=stroke_white, zorder=5)

    # 期望值
    ax1.axvline(dist.expect(), linestyle="--", color="C1", zorder=4)  # type: ignore
    ax1.annotate(f"期望\n{dist.expect():.2f}次", (dist.expect(), ax1_y_top),  # type: ignore
                 ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                 color="C1", fontweight="medium", path_effects=stroke_white, zorder=9)

    # 分位数
    [
        ax1.axvline(quantile_point, color="gray", linestyle="--", zorder=5)
        for quantile_point in quantile_points
    ]
    [
        ax1.annotate(f"{quantile_point}次\n{dist.cdf(quantile_point):.0%}", (quantile_point, ax1_y_top),
                     ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                     color="gray", fontweight="medium", path_effects=stroke_white, zorder=8)
        for quantile_point in quantile_points
    ]

    # 众数
    ax1.scatter(np.argmax(dist.pk), np.max(dist.pk),
                color="C1", marker=".", path_effects=stroke_white, zorder=6)
    ax1.annotate(f"最可能\n{np.argmax(dist.pk)}次", (np.argmax(dist.pk), np.max(dist.pk)),  # type: ignore
                 ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                 color="C1", fontweight="medium", path_effects=stroke_white, zorder=10)

    # 期望值
    ax2.axvline(dist.expect(), linestyle="--", color="C1", zorder=4)  # type: ignore
    ax2.annotate(f"期望\n{dist.expect():.2f}次", (dist.expect(), ax2_y_top),  # type: ignore
                 ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                 color="C1", fontweight="medium", path_effects=stroke_white, zorder=9)

    # 分位数
    ax2.scatter(quantile_points, dist.cdf(quantile_points),
                color="C0", marker=".", path_effects=stroke_white, zorder=6)
    [
        ax2.annotate(f"{quantile_point}次\n{dist.cdf(quantile_point):.0%}", (quantile_point, dist.cdf(quantile_point)),  # type: ignore
                     ha="right", va="baseline", xytext=(-5, 5), textcoords="offset points",
                     color="gray", fontweight="medium", path_effects=stroke_white, zorder=8)
        for quantile_point in quantile_points
    ]

    # 图表标题、坐标轴标签
    fig.suptitle(f"明日方舟寻访机制解析　bilibili@Bio-Hazard\n{title}", fontweight="bold", fontsize="x-large")
    ax1.set_title("概率质量函数", fontweight="bold")
    ax2.set_title("累积分布函数", fontweight="bold")
    ax1.set_ylabel("本次概率", fontweight="bold")
    ax2.set_ylabel("累积概率", fontweight="bold")
    ax2.set_xlabel("寻访次数", fontweight="bold")

    for ax in (ax1, ax2):
        # 绘制水印
        ax.text(0.98, 0.02, "bilibili@Bio-Hazard", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=14, fontweight="medium", color="black", alpha=0.3, path_effects=stroke_white, zorder=12)

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
        fig.savefig(f"图片/{title}.png", dpi=300)

    return fig, (ax1, ax2)


def draw_multi_cdf_fig(dists: Sequence[FiniteDist],
                       title: str,
                       labels: Sequence[str],
                       *,
                       x_max: float | Literal["auto"],
                       quantile_poses: Sequence[float] | None = None,
                       save: bool = False):
    if quantile_poses is None:
        quantile_poses = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    colors = [f"C{i}" for i in range(len(dists))]
    ax_y_top = 1.15
    if x_max == "auto":
        x_max = dists[-1].ppf(0.99) * 1.25

    fig = plt.figure(figsize=(10, 6), layout="tight")
    ax: Axes = fig.subplots(1, 1)

    for dist, label, color in zip(dists, labels, colors):
        x = np.arange(len(dist.pk))
        ax.plot(x, dist.cdf(x), color=color, drawstyle="steps-mid", path_effects=stroke_white, label=label, zorder=5)

        # 期望值
        ax.axvline(dist.expect(), linestyle="--", color=color, zorder=4)  # type: ignore
        ax.annotate(f"期望\n{dist.expect():.2f}次", (dist.expect(), ax_y_top),  # type: ignore
                    ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                    color=color, fontweight="medium", path_effects=stroke_white, zorder=9)
        # ax.annotate(f"{dist.expect():.2f}", (round(dist.expect()), dist.cdf(round(dist.expect()))),
        #             ha="center", va="baseline",
        #             color=color, fontweight="medium", path_effects=stroke_white, zorder=20)

        # 分位数
        quantile_points = dist.ppf(quantile_poses).astype(int)
        ax.scatter(quantile_points, dist.cdf(quantile_points),
                   s=10, color=color, marker=".", path_effects=stroke_white, zorder=6)
        [
            ax.annotate(quantile_point, (quantile_point, quantile_pos),
                        ha="right", va="baseline", xytext=(-2, 2), textcoords="offset points",
                        color="gray", fontweight="medium", path_effects=stroke_white, zorder=8)
            for quantile_point, quantile_pos in zip(quantile_points, quantile_poses)
        ]
        [
            ax.axhline(quantile_pos, color="gray", linestyle="--", zorder=4)
            for quantile_pos in quantile_poses
        ]
        [
            ax.annotate(f"{quantile_pos:.0%}", (x_max, quantile_pos),
                        ha="right", va="center", xytext=(-5, 0), textcoords="offset points",
                        color="gray", fontweight="medium", path_effects=stroke_white, zorder=7)
            for quantile_pos in quantile_poses
        ]

    # 图表标题
    fig.suptitle(f"明日方舟寻访机制解析　bilibili@Bio-Hazard\n{title}", fontweight="bold", fontsize="x-large")
    ax.set_title("累积分布函数", fontweight="bold")
    ax.set_ylabel("累积概率", fontweight="bold")
    ax.set_xlabel("寻访次数", fontweight="bold")

    # 启用 minor tick
    ax.minorticks_on()

    # 显示网格
    ax.grid(True, which="major", linewidth=1.2)
    ax.grid(True, which="minor", linewidth=0.6)

    # 坐标轴范围
    ax.set_ylim(0, ax_y_top)
    ax.set_xlim(0, x_max)

    # y 轴标签显示为百分比
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # 在图表下方添加图例
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6)

    # 绘制水印
    ax.text(0.98, 0.02, "bilibili@Bio-Hazard", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14, fontweight="medium", color="black", alpha=0.3, path_effects=stroke_white, zorder=12)

    if save:
        fig.savefig(f"图片/{title}.png", dpi=300)

    return fig, (ax,)


def draw_multi_pmf_cdf_fig(dists: Sequence[FiniteDist],
                           title: str,
                           labels: Sequence[str],
                           *,
                           x_max: float | Literal["auto"],
                           quantile_poses: Sequence[float] | None = None,
                           ax1_y_top: float | None = None,
                           save: bool = False) -> tuple[Figure, tuple[Axes, Axes]]:
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

    for dist, label, color in zip(dists, labels, colors):
        x = np.arange(len(dist.pk))
        ax1.fill_between(x, dist.pk, step="mid", alpha=0.3, zorder=2)
        ax1.plot(x, dist.pk, label=label,
                 color=color, drawstyle="steps-mid", path_effects=stroke_white, zorder=5)
        # ax2.fill_between(x, dist.cdf(x), step="mid", alpha=0.3, zorder=3)
        ax2.plot(x, dist.cdf(x), label=label,
                 color=color, drawstyle="steps-mid", path_effects=stroke_white, zorder=5)

        # 期望值
        ax1.axvline(dist.expect(), linestyle="--", color=color, zorder=4)  # type: ignore
        ax1.annotate(f"期望\n{dist.expect():.2f}次", (dist.expect(), ax1_y_top),  # type: ignore
                     ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                     color=color, fontweight="medium", path_effects=stroke_white, zorder=9)
        ax2.axvline(dist.expect(), linestyle="--", color=color, zorder=4)  # type: ignore
        ax2.annotate(f"期望\n{dist.expect():.2f}次", (dist.expect(), ax2_y_top),  # type: ignore
                     ha="center", va="top", xytext=(0, -5), textcoords="offset points",
                     color=color, fontweight="medium", path_effects=stroke_white, zorder=9)

        # 众数
        ax1.scatter(np.argmax(dist.pk), np.max(dist.pk),
                    s=10, color=color, marker=".", path_effects=stroke_white, zorder=6)
        ax1.annotate(f"最可能\n{np.argmax(dist.pk)}次", (np.argmax(dist.pk), np.max(dist.pk)),  # type: ignore
                     ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                     color=color, fontweight="medium", path_effects=stroke_white, zorder=10)

        # 分位数
        quantile_points = dist.ppf(quantile_poses).astype(int)
        ax2.scatter(quantile_points, dist.cdf(quantile_points),
                    s=10, color=color, marker=".", path_effects=stroke_white, zorder=6)
        [
            ax2.annotate(quantile_point, (quantile_point, quantile_pos),
                         ha="right", va="baseline", xytext=(-2, 2), textcoords="offset points",
                         color="gray", fontweight="medium", path_effects=stroke_white, zorder=8)
            for quantile_point, quantile_pos in zip(quantile_points, quantile_poses)
        ]
        [
            ax2.axhline(quantile_pos, color="gray", linestyle="--", zorder=3)
            for quantile_pos in quantile_poses
        ]
        [
            ax2.annotate(f"{quantile_pos:.0%}", (x_max, quantile_pos),
                         ha="right", va="center", xytext=(-5, 0), textcoords="offset points",
                         color="gray", fontweight="medium", path_effects=stroke_white, zorder=7)
            for quantile_pos in quantile_poses
        ]

    # 图表标题
    fig.suptitle(f"明日方舟寻访机制解析　bilibili@Bio-Hazard\n{title}", fontweight="bold", fontsize="x-large")
    ax1.set_title("概率质量函数", fontweight="bold")
    ax2.set_title("累积分布函数", fontweight="bold")
    ax1.set_ylabel("本次概率", fontweight="bold")
    ax2.set_ylabel("累积概率", fontweight="bold")
    ax2.set_xlabel("寻访次数", fontweight="bold")

    for ax in (ax1, ax2):
        # 绘制水印
        ax.text(0.98, 0.02, "bilibili@Bio-Hazard", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=14, fontweight="medium", color="black", alpha=0.3, path_effects=stroke_white, zorder=12)

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
        fig.savefig(f"图片/{title}.png", dpi=300)

    return fig, (ax1, ax2)
