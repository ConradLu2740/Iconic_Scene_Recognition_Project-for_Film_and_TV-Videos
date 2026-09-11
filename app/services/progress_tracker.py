import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    IDLE = "idle"
    SCENE_DETECTION = "scene_detection"
    KEYFRAME_EXTRACTION = "keyframe_extraction"
    AUDIO_EXTRACTION = "audio_extraction"
    VLM_ANALYSIS = "vlm_analysis"
    WHISPER_TRANSCRIPTION = "whisper_transcription"
    AUDIO_ANALYSIS = "audio_analysis"
    MERGING_FILTERING = "merging_filtering"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressInfo:
    stage: PipelineStage
    stage_progress: float
    overall_progress: float
    current: int
    total: int
    message: str
    details: dict = field(default_factory=dict)
    elapsed_time: float = 0.0


class StreamingProgressCallback:
    """流式进度回调

    支持多种输出模式：实时打印、回调、生成器
    """

    STAGE_WEIGHTS = {
        PipelineStage.SCENE_DETECTION: 0.05,
        PipelineStage.KEYFRAME_EXTRACTION: 0.10,
        PipelineStage.AUDIO_EXTRACTION: 0.05,
        PipelineStage.VLM_ANALYSIS: 0.40,
        PipelineStage.WHISPER_TRANSCRIPTION: 0.15,
        PipelineStage.AUDIO_ANALYSIS: 0.10,
        PipelineStage.MERGING_FILTERING: 0.10,
        PipelineStage.COMPLETED: 1.0,
    }

    def __init__(
        self,
        callback: Callable[[ProgressInfo], None] | None = None,
        verbose: bool = True,
    ):
        self.callback = callback
        self.verbose = verbose
        self._current_stage = PipelineStage.IDLE
        self._stage_start_time = time.time()
        self._total_start_time = time.time()
        self._last_update_time = 0
        self._min_update_interval = 0.5

    def __call__(self, stage: PipelineStage, current: int, total: int, message: str = "", **details):
        """进度回调入口"""
        current_time = time.time()

        if current_time - self._last_update_time < self._min_update_interval and total > 0:
            if current < total:
                return

        self._update(stage, current, total, message, **details)

    def _update(self, stage: PipelineStage, current: int, total: int, message: str = "", **details):
        """内部更新逻辑"""
        if stage != self._current_stage:
            self._current_stage = stage
            self._stage_start_time = time.time()

        stage_progress = current / total if total > 0 else 0.0

        overall_progress = self._calculate_overall_progress(stage, stage_progress)

        elapsed = time.time() - self._total_start_time

        progress_info = ProgressInfo(
            stage=stage,
            stage_progress=round(stage_progress, 3),
            overall_progress=round(overall_progress, 3),
            current=current,
            total=total,
            message=message,
            details=details,
            elapsed_time=round(elapsed, 1),
        )

        if self.callback:
            self.callback(progress_info)

        if self.verbose:
            self._print_progress(progress_info)

        self._last_update_time = time.time()

    def _calculate_overall_progress(self, stage: PipelineStage, stage_progress: float) -> float:
        """计算总体进度"""
        if stage == PipelineStage.COMPLETED:
            return 1.0

        if stage == PipelineStage.FAILED:
            return 0.0

        base_progress = 0.0
        for s, weight in self.STAGE_WEIGHTS.items():
            if s == stage:
                return base_progress + weight * stage_progress
            base_progress += weight

        return min(1.0, base_progress)

    def _print_progress(self, info: ProgressInfo):
        """打印进度信息"""
        stage_name = self._get_stage_display_name(info.stage)
        bar_length = 30
        filled = int(bar_length * info.stage_progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        percentage = int(info.stage_progress * 100)
        elapsed_str = f"{info.elapsed_time:.1f}s"

        status = f"\r[{bar}] {percentage:3d}% | {stage_name} | {elapsed_str}"
        if info.message:
            status += f" | {info.message}"

        print(status, end="", flush=True)

        if info.stage == PipelineStage.COMPLETED:
            print()

    def _get_stage_display_name(self, stage: PipelineStage) -> str:
        """获取阶段显示名称"""
        names = {
            PipelineStage.IDLE: "等待",
            PipelineStage.SCENE_DETECTION: "场景检测",
            PipelineStage.KEYFRAME_EXTRACTION: "关键帧提取",
            PipelineStage.AUDIO_EXTRACTION: "音频提取",
            PipelineStage.VLM_ANALYSIS: "VLM分析",
            PipelineStage.WHISPER_TRANSCRIPTION: "语音转写",
            PipelineStage.AUDIO_ANALYSIS: "音频分析",
            PipelineStage.MERGING_FILTERING: "合并过滤",
            PipelineStage.COMPLETED: "完成",
            PipelineStage.FAILED: "失败",
        }
        return names.get(stage, str(stage.value))


class ProgressGenerator:
    """进度生成器

    支持异步生成器模式，实时产生进度更新
    """

    def __init__(self):
        self._callbacks: list[Callable[[ProgressInfo], None]] = []

    def add_callback(self, callback: Callable[[ProgressInfo], None]):
        """添加进度回调"""
        self._callbacks.append(callback)

    def emit(self, stage: PipelineStage, current: int, total: int, message: str = "", **details):
        """发送进度更新"""
        elapsed = time.time() - getattr(self, "_start_time", time.time())

        info = ProgressInfo(
            stage=stage,
            stage_progress=current / total if total > 0 else 0.0,
            overall_progress=0.0,
            current=current,
            total=total,
            message=message,
            details=details,
            elapsed_time=elapsed,
        )

        for callback in self._callbacks:
            callback(info)

    def start(self):
        """标记开始"""
        self._start_time = time.time()


async def progress_stream(
    stages: list[tuple[PipelineStage, int]],
) -> Generator[ProgressInfo, None, None]:
    """异步进度流

    用法示例:
        async for progress in progress_stream([(stage1, 100), (stage2, 50)]):
            print(f"进度: {progress.overall_progress:.1%}")
    """
    start_time = time.time()
    total_weights = sum(StreamingProgressCallback.STAGE_WEIGHTS.get(s, 0) for s, _ in stages)

    accumulated_progress = 0.0

    for stage, total in stages:
        for current in range(total + 1):
            await asyncio.sleep(0)

            stage_progress = current / total if total > 0 else 1.0
            stage_weight = StreamingProgressCallback.STAGE_WEIGHTS.get(stage, 0)

            overall = accumulated_progress + stage_weight * stage_progress

            yield ProgressInfo(
                stage=stage,
                stage_progress=stage_progress,
                overall_progress=min(1.0, overall),
                current=current,
                total=total,
                message="",
                details={},
                elapsed_time=time.time() - start_time,
            )

        accumulated_progress += stage_weight

    yield ProgressInfo(
        stage=PipelineStage.COMPLETED,
        stage_progress=1.0,
        overall_progress=1.0,
        current=1,
        total=1,
        message="完成",
        details={},
        elapsed_time=time.time() - start_time,
    )


class ProgressTracker:
    """进度追踪器

    追踪pipeline各阶段的时间消耗和性能指标
    """

    def __init__(self):
        self._stage_times: dict[PipelineStage, list[float]] = {}
        self._current_stage: PipelineStage | None = None
        self._stage_start: float = 0.0
        self._total_start: float = 0.0

    def start(self):
        """开始追踪"""
        self._total_start = time.time()

    def begin_stage(self, stage: PipelineStage):
        """开始一个阶段"""
        if self._current_stage:
            self.end_stage()

        self._current_stage = stage
        self._stage_start = time.time()

    def end_stage(self):
        """结束当前阶段"""
        if self._current_stage:
            elapsed = time.time() - self._stage_start
            if self._current_stage not in self._stage_times:
                self._stage_times[self._current_stage] = []
            self._stage_times[self._current_stage].append(elapsed)

    def get_summary(self) -> dict[str, Any]:
        """获取性能摘要"""
        self.end_stage()

        total_time = time.time() - self._total_start

        summary = {
            "total_time": round(total_time, 2),
            "stages": {},
        }

        for stage, times in self._stage_times.items():
            avg_time = sum(times) / len(times) if times else 0
            summary["stages"][stage.value] = {
                "count": len(times),
                "total_time": round(sum(times), 2),
                "avg_time": round(avg_time, 2),
                "min_time": round(min(times), 2) if times else 0,
                "max_time": round(max(times), 2) if times else 0,
            }

        return summary

    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("性能摘要")
        print("=" * 60)
        print(f"总耗时: {summary['total_time']:.2f}秒")
        print("-" * 60)

        for stage_name, stats in summary["stages"].items():
            pct = (stats["total_time"] / summary["total_time"] * 100) if summary["total_time"] > 0 else 0
            print(f"{stage_name:30s} | {stats['total_time']:6.2f}s | {pct:5.1f}% | 均值: {stats['avg_time']:.2f}s")

        print("=" * 60)
