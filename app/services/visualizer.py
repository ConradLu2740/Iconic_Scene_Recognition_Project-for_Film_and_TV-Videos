import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

TYPE_COLORS = {
    "action": "#FF4444",
    "drama": "#4444FF",
    "emotion": "#FF69B4",
    "comedy": "#FFD700",
    "suspense": "#9932CC",
    "other": "#808080",
}


@dataclass
class TimelineSegment:
    start_time: float
    end_time: float
    highlight_id: int
    score: float
    segment_type: str
    keyframe_path: str | None = None


def generate_timeline_visualization(
    highlights: list[dict],
    video_duration: float,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 8),
    show_scores: bool = True,
) -> Path | None:
    """生成高光片段时间轴可视化

    Args:
        highlights: 高光片段列表
        video_duration: 视频总时长（秒）
        output_path: 输出图片路径，默认为可视化管理目录
        figsize: 图表大小
        show_scores: 是否显示评分

    Returns:
        输出图片路径
    """
    if not highlights:
        logger.warning("没有高光片段可可视化")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 2)

    ax.axhline(y=0.5, color="lightgray", linestyle="-", linewidth=1)

    time_ticks = np.arange(0, video_duration, max(10, video_duration / 10))
    time_labels = [format_time(t) for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)

    ax.set_yticks([])
    ax.set_xlabel("时间", fontsize=12)

    ax.set_title("视频名场面时间轴", fontsize=16, fontweight="bold", pad=20)

    y_position = 0.5
    bar_height = 0.4

    for idx, hl in enumerate(highlights):
        start = hl.get("start_seconds", 0)
        end = hl.get("end_seconds", 0)
        duration = end - start
        segment_type = hl.get("type", "other")
        score = hl.get("scores", {}).get("total", 0)
        hl_id = hl.get("id", idx + 1)

        color = TYPE_COLORS.get(segment_type, TYPE_COLORS["other"])

        alpha = min(1.0, 0.4 + (score / 10) * 0.6)

        bar = ax.barh(
            y_position,
            duration,
            left=start,
            height=bar_height,
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=1,
        )

        center = start + duration / 2
        label_text = f"#{hl_id}"
        if show_scores:
            label_text += f"\n{score:.1f}"

        ax.text(
            center,
            y_position,
            label_text,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

    legend_patches = [
        mpatches.Patch(color=color, label=type_name.capitalize())
        for type_name, color in TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()

    if output_path is None:
        from app.config_v2 import config
        output_path = config.VISUALIZATION_DIR / "timeline.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"时间轴可视化已保存: {output_path}")
    return output_path


def generate_score_distribution_chart(
    highlights: list[dict],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> Path | None:
    """生成评分分布图表

    Args:
        highlights: 高光片段列表
        output_path: 输出图片路径
        figsize: 图表大小

    Returns:
        输出图片路径
    """
    if not highlights:
        return None

    dimensions = [
        "visual_impact", "cinematography", "emotion_intensity",
        "facial_expression", "plot_importance", "action_intensity",
        "audio_energy", "memorability",
    ]

    dim_labels = [
        "视觉", "镜头", "情感", "表情",
        "剧情", "动作", "音频", "记忆",
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    all_scores = [[] for _ in dimensions]
    for hl in highlights:
        scores = hl.get("scores", {})
        for i, dim in enumerate(dimensions):
            all_scores[i].append(scores.get(dim, 0))

    means = [np.mean(scores) for scores in all_scores]
    stds = [np.std(scores) for scores in all_scores]

    colors = [TYPE_COLORS.get(hl.get("type", "other"), TYPE_COLORS["other"]) for hl in highlights]
    individual_scores = [
        [hl.get("scores", {}).get(dim, 0) for dim in dimensions]
        for hl in highlights
    ]

    x = np.arange(len(dimensions))
    width = 0.6

    bars = axes[0].bar(x, means, width, yerr=stds, capsize=5, color="steelblue", alpha=0.7)
    axes[0].set_ylabel("评分", fontsize=11)
    axes[0].set_title("各维度平均评分", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dim_labels, rotation=45, ha="right")
    axes[0].set_ylim(0, 12)
    axes[0].axhline(y=6, color="red", linestyle="--", alpha=0.5, label="阈值6.0")

    for bar, mean in zip(bars, means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    im = axes[1].imshow(individual_scores, cmap="YlOrRd", aspect="auto", vmin=0, vmax=10)
    axes[1].set_title("片段评分热力图", fontsize=14, fontweight="bold")
    axes[1].set_yticks(range(len(highlights)))
    axes[1].set_yticklabels([f"#{hl.get('id', i+1)}" for i, hl in enumerate(highlights)])
    axes[1].set_xticks(range(len(dim_labels)))
    axes[1].set_xticklabels(dim_labels, rotation=45, ha="right")
    plt.colorbar(im, ax=axes[1], label="评分")

    plt.tight_layout()

    if output_path is None:
        from app.config_v2 import config
        output_path = config.VISUALIZATION_DIR / "score_distribution.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"评分分布图表已保存: {output_path}")
    return output_path


def generate_type_distribution_pie(
    highlights: list[dict],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> Path | None:
    """生成类型分布饼图

    Args:
        highlights: 高光片段列表
        output_path: 输出图片路径
        figsize: 图表大小

    Returns:
        输出图片路径
    """
    if not highlights:
        return None

    type_counts: dict[str, int] = {}
    for hl in highlights:
        segment_type = hl.get("type", "other")
        type_counts[segment_type] = type_counts.get(segment_type, 0) + 1

    labels = []
    sizes = []
    colors = []

    for type_name in TYPE_COLORS:
        if type_name in type_counts:
            labels.append(type_name.capitalize())
            sizes.append(type_counts[type_name])
            colors.append(TYPE_COLORS[type_name])

    if "other" in type_counts and "other" in [l.lower() for l in labels]:
        other_idx = [l.lower() for l in labels].index("other")
        labels[other_idx] = "Other"
    elif "other" in type_counts:
        labels.append("Other")
        sizes.append(type_counts["other"])
        colors.append(TYPE_COLORS["other"])

    fig, ax = plt.subplots(figsize=figsize)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.02] * len(sizes),
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title("名场面类型分布", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path is None:
        from app.config_v2 import config
        output_path = config.VISUALIZATION_DIR / "type_distribution.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"类型分布饼图已保存: {output_path}")
    return output_path


def generate_full_report_visualization(
    highlights: list[dict],
    video_duration: float,
    analysis_config: dict | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """生成完整的可视化报告

    Args:
        highlights: 高光片段列表
        video_duration: 视频总时长
        analysis_config: 分析配置信息
        output_dir: 输出目录

    Returns:
        各图表的输出路径
    """
    if output_dir is None:
        from app.config_v2 import config
        output_dir = config.VISUALIZATION_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    timeline_path = output_dir / "timeline.png"
    if generate_timeline_visualization(highlights, video_duration, timeline_path):
        paths["timeline"] = timeline_path

    score_path = output_dir / "score_distribution.png"
    if generate_score_distribution_chart(highlights, score_path):
        paths["score_distribution"] = score_path

    pie_path = output_dir / "type_distribution.png"
    if generate_type_distribution_pie(highlights, pie_path):
        paths["type_distribution"] = pie_path

    return paths


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"
