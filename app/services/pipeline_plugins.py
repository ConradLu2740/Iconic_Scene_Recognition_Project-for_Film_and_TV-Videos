import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config_v2 import config
from app.services.progress_tracker import PipelineStage, StreamingProgressCallback, ProgressTracker
from app.services.smart_cache import smart_cache, incremental_cache

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """分析上下文，贯穿整个Pipeline"""

    video_path: Path
    video_name: str
    video_md5: str
    scenes: list = field(default_factory=list)
    keyframe_paths: dict[int, list[str]] = field(default_factory=dict)
    audio_path: str | None = None
    transcript: dict | None = None
    audio_features: dict[int, dict] = field(default_factory=dict)
    scene_analyses: list = field(default_factory=list)
    highlights: list = field(default_factory=list)
    progress_callback: StreamingProgressCallback | None = None
    metadata: dict = field(default_factory=dict)

    def to_result(self) -> dict:
        """转换为最终结果格式"""
        return {
            "input_video": self.video_name,
            "scenes": self.scenes,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


class AnalyzerPlugin(ABC):
    """分析器插件基类"""

    name: str = "base"
    priority: int = 0
    is_parallel_safe: bool = True

    def __init__(self):
        self.context: AnalysisContext | None = None
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @abstractmethod
    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        """执行分析逻辑

        Args:
            context: 分析上下文

        Returns:
            更新后的上下文
        """
        pass

    def before_run(self, context: AnalysisContext):
        """运行前钩子"""
        pass

    def after_run(self, context: AnalysisContext):
        """运行后钩子"""
        pass

    async def run(self, context: AnalysisContext) -> AnalysisContext:
        """运行插件"""
        if not self._enabled:
            logger.debug(f"插件 {self.name} 已禁用，跳过")
            return context

        logger.info(f"运行插件: {self.name}")
        self.before_run(context)
        self.context = context

        try:
            context = await self.analyze(context)
        except Exception as e:
            logger.error(f"插件 {self.name} 执行失败: {e}")
            raise

        self.after_run(context)
        return context


class SceneDetectorPlugin(AnalyzerPlugin):
    """场景检测插件"""

    name = "scene_detector"
    priority = 1

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.scene_detector import detect_scenes

        if context.progress_callback:
            context.progress_callback(PipelineStage.SCENE_DETECTION, 0, 1, "检测场景中...")

        scenes = detect_scenes(context.video_path)
        context.scenes = [s.__dict__ for s in scenes]

        incremental_cache.save_scene_detection(context.video_path, context.scenes)

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.SCENE_DETECTION,
                1, 1,
                f"检测到 {len(scenes)} 个场景"
            )

        logger.info(f"场景检测完成: {len(scenes)} 个场景")
        return context


class KeyframeExtractorPlugin(AnalyzerPlugin):
    """关键帧提取插件"""

    name = "keyframe_extractor"
    priority = 2

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.scene_detector import SceneSegment
        from app.services.frame_quality import extract_best_keyframes

        scenes = [SceneSegment(**s) for s in context.scenes]

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.KEYFRAME_EXTRACTION,
                0, len(scenes),
                "提取关键帧..."
            )

        for idx, scene in enumerate(scenes):
            if context.progress_callback:
                context.progress_callback(
                    PipelineStage.KEYFRAME_EXTRACTION,
                    idx, len(scenes),
                    f"提取场景 {idx + 1} 关键帧"
                )

            num_samples = config.pipeline.keyframe_samples_per_scene
            best_frames = extract_best_keyframes(
                context.video_path,
                scene,
                num_samples=num_samples,
                quality_threshold=config.pipeline.quality_threshold,
            )

            keyframe_list = []
            for position, quality_score in best_frames:
                kf_path = self._extract_single_frame(
                    context.video_path,
                    position,
                    context.video_md5,
                    idx,
                )
                if kf_path:
                    keyframe_list.append(kf_path)

            context.keyframe_paths[idx] = keyframe_list

        incremental_cache.save_keyframes(context.video_path, context.keyframe_paths)

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.KEYFRAME_EXTRACTION,
                len(scenes), len(scenes),
                f"提取了 {sum(len(v) for v in context.keyframe_paths.values())} 个关键帧"
            )

        logger.info(f"关键帧提取完成: {len(context.keyframe_paths)} 个场景")
        return context

    def _extract_single_frame(
        self,
        video_path: Path,
        position: float,
        video_md5: str,
        scene_idx: int,
    ) -> str | None:
        import cv2

        output_dir = config.KEYFRAME_DIR / video_md5
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(position * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        filename = f"scene_{scene_idx + 1:04d}_{position:.2f}s.jpg"
        filepath = output_dir / filename

        cv2.imwrite(str(filepath), frame)
        return str(filepath)


class AudioExtractorPlugin(AnalyzerPlugin):
    """音频提取插件"""

    name = "audio_extractor"
    priority = 3
    is_parallel_safe = True

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.audio_analyzer import extract_audio

        if context.progress_callback:
            context.progress_callback(PipelineStage.AUDIO_EXTRACTION, 0, 1, "提取音频...")

        audio_path = extract_audio(context.video_path, config.AUDIO_DIR / f"{context.video_md5}.wav")
        context.audio_path = audio_path

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.AUDIO_EXTRACTION,
                1, 1,
                "音频提取完成" if audio_path else "音频提取失败"
            )

        return context


class VLMAnalyzerPlugin(AnalyzerPlugin):
    """VLM分析插件"""

    name = "vlm_analyzer"
    priority = 4

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.vlm_analyzer_concurrent import ConcurrentVLMAnalyzer

        if not context.keyframe_paths:
            return context

        analyzer = ConcurrentVLMAnalyzer()

        total_frames = sum(len(frames) for frames in context.keyframe_paths.values())

        if context.progress_callback:
            context.progress_callback(PipelineStage.VLM_ANALYSIS, 0, total_frames, "VLM分析...")

        all_results = {}

        async def progress_wrapper(current, total):
            if context.progress_callback:
                context.progress_callback(PipelineStage.VLM_ANALYSIS, current, total, "VLM分析中...")

        scene_frame_results = await analyzer.analyze_batch(
            [path for paths in context.keyframe_paths.values() for path in paths],
            progress_callback=progress_wrapper,
        )

        idx = 0
        for scene_idx, paths in context.keyframe_paths.items():
            scene_results = []
            for _ in paths:
                if idx < len(scene_frame_results):
                    result = scene_frame_results[idx]
                    if result:
                        scene_results.append(result)
                idx += 1
            all_results[scene_idx] = scene_results

        context.scene_analyses = all_results

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.VLM_ANALYSIS,
                total_frames, total_frames,
                f"VLM分析完成"
            )

        return context


class WhisperPlugin(AnalyzerPlugin):
    """Whisper语音转写插件"""

    name = "whisper_transcriber"
    priority = 5
    is_parallel_safe = True

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.whisper_transcriber import transcribe_video

        if context.progress_callback:
            context.progress_callback(PipelineStage.WHISPER_TRANSCRIPTION, 0, 1, "语音转写中...")

        transcript = transcribe_video(context.video_path)

        if transcript:
            context.transcript = {
                "full_text": transcript.full_text,
                "segments": [s.__dict__ for s in transcript.segments],
                "language": transcript.language,
            }
            incremental_cache.save_transcript(context.video_path, context.transcript)

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.WHISPER_TRANSCRIPTION,
                1, 1,
                "语音转写完成" if transcript else "语音转写失败"
            )

        return context


class AudioAnalyzerPlugin(AnalyzerPlugin):
    """音频能量分析插件"""

    name = "audio_analyzer"
    priority = 6
    is_parallel_safe = True

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.audio_analyzer import extract_audio_features_for_scenes
        from app.services.scene_detector import SceneSegment

        if not context.audio_path:
            audio_path = extract_audio_features_for_scenes(
                context.video_path,
                [SceneSegment(**s) for s in context.scenes],
                config.AUDIO_DIR,
            )
            if not audio_path:
                return context
            context.audio_path = list(audio_path.values())[0].__dict__ if audio_path else None

        if context.progress_callback:
            context.progress_callback(PipelineStage.AUDIO_ANALYSIS, 0, len(context.scenes), "音频分析...")

        scenes = [SceneSegment(**s) for s in context.scenes]
        audio_features = extract_audio_features_for_scenes(
            context.video_path,
            scenes,
            config.AUDIO_DIR,
        )

        for idx, result in audio_features.items():
            context.audio_features[idx] = result.__dict__

        if context.progress_callback:
            context.progress_callback(
                PipelineStage.AUDIO_ANALYSIS,
                len(context.scenes), len(context.scenes),
                f"分析了 {len(audio_features)} 个片段"
            )

        return context


class HighlightGeneratorPlugin(AnalyzerPlugin):
    """高光生成插件"""

    name = "highlight_generator"
    priority = 7

    async def analyze(self, context: AnalysisContext) -> AnalysisContext:
        from app.services.highlight_pipeline import (
            _compute_weighted_score,
            _merge_consecutive_highlights,
            _filter_similar_scenes,
            _filter_by_type,
            _filter_by_scores,
        )
        from app.services.scene_detector import SceneSegment

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 0, 5, "生成高光...")

        highlights = []
        for idx, scene in enumerate(context.scenes):
            scene_seg = SceneSegment(**scene)
            scene_results = context.scene_analyses.get(idx, [])

            if not scene_results:
                continue

            fused = self._fuse_scene_results(scene_results)

            audio_energy = 5.0
            if idx in context.audio_features:
                audio_energy = context.audio_features[idx].get("energy_score", 5.0)

            highlight = {
                "id": idx + 1,
                "start_time": scene_seg.start_timecode,
                "end_time": scene_seg.end_timecode,
                "start_seconds": scene_seg.start_seconds,
                "end_seconds": scene_seg.end_seconds,
                "keyframe_url": context.keyframe_paths.get(idx, [""])[0] if context.keyframe_paths.get(idx) else "",
                "scores": {
                    "visual_impact": fused.visual_impact,
                    "cinematography": fused.cinematography,
                    "emotion_intensity": fused.emotion_intensity,
                    "facial_expression": fused.facial_expression,
                    "plot_importance": fused.plot_importance,
                    "action_intensity": fused.action_intensity,
                    "audio_energy": audio_energy,
                    "memorability": fused.memorability,
                },
                "type": fused.type,
                "description": fused.description,
            }

            highlight["scores"]["total"] = _compute_weighted_score(highlight["scores"])
            highlights.append(highlight)

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 1, 5, f"初选 {len(highlights)} 个")

        highlights = _merge_consecutive_highlights(highlights)

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 2, 5, f"合并后 {len(highlights)} 个")

        highlights = _filter_similar_scenes(highlights)

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 3, 5, f"去重后 {len(highlights)} 个")

        threshold = context.metadata.get("score_threshold", config.SCORE_THRESHOLD)
        highlights = [h for h in highlights if h["scores"]["total"] >= threshold]

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 4, 5, f"阈值过滤后 {len(highlights)} 个")

        if context.metadata.get("type_filter"):
            highlights = _filter_by_type(highlights, context.metadata["type_filter"])

        highlights.sort(key=lambda h: h["scores"]["total"], reverse=True)

        if context.metadata.get("count_limit"):
            highlights = highlights[: context.metadata["count_limit"]]

        for i, hl in enumerate(highlights):
            hl["id"] = i + 1

        context.highlights = highlights

        if context.progress_callback:
            context.progress_callback(PipelineStage.MERGING_FILTERING, 5, 5, f"最终 {len(highlights)} 个高光")

        return context

    def _fuse_scene_results(self, results: list) -> Any:
        """融合场景分析结果"""
        if len(results) == 1:
            return results[0]

        dimensions = [
            "visual_impact", "cinematography", "emotion_intensity",
            "facial_expression", "plot_importance", "action_intensity",
            "audio_energy", "memorability",
        ]

        fused = {}
        for dim in dimensions:
            values = [getattr(r, dim) for r in results]
            fused[dim] = round(sum(values) / len(values), 1)

        from collections import Counter
        type_counts = Counter([r.type for r in results])
        fused["type"] = type_counts.most_common(1)[0][0]

        descriptions = [r.description for r in results if r.description]
        fused["description"] = descriptions[0] if descriptions else ""

        return type("FusedResult", (), fused)()


class HighlightPipeline:
    """高光检测Pipeline"""

    def __init__(self):
        self.plugins: list[AnalyzerPlugin] = []
        self._tracker = ProgressTracker()

    def register(self, plugin: AnalyzerPlugin):
        """注册插件"""
        self.plugins.append(plugin)
        self.plugins.sort(key=lambda p: p.priority)
        logger.info(f"注册插件: {plugin.name} (优先级 {plugin.priority})")

    def unregister(self, name: str):
        """取消注册插件"""
        self.plugins = [p for p in self.plugins if p.name != name]

    def get_plugin(self, name: str) -> AnalyzerPlugin | None:
        """获取插件"""
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None

    def enable_plugin(self, name: str, enabled: bool = True):
        """启用/禁用插件"""
        plugin = self.get_plugin(name)
        if plugin:
            plugin.enabled = enabled

    async def run(
        self,
        video_path: str | Path,
        progress_callback: StreamingProgressCallback | None = None,
        **metadata,
    ) -> AnalysisContext:
        """运行Pipeline"""
        video_path = Path(video_path)

        if smart_cache.is_analyzing(video_path):
            in_progress = smart_cache.get_in_progress_info(video_path)
            raise RuntimeError(f"视频正在分析中: {in_progress}")

        analysis_id = smart_cache.start_analysis(video_path)

        try:
            context = AnalysisContext(
                video_path=video_path,
                video_name=video_path.name,
                video_md5=smart_cache._smart_cache._compute_file_md5(video_path),
                progress_callback=progress_callback,
                metadata=metadata,
            )

            self._tracker.start()

            for plugin in self.plugins:
                if not plugin.enabled:
                    continue

                self._tracker.begin_stage(PipelineStage[plugin.name.upper().replace("_", "_")] if plugin.name.upper().replace("_", "_") in PipelineStage.__members__ else PipelineStage.MERGING_FILTERING)

                context = await plugin.run(context)

                self._tracker.end_stage()

            context.metadata["analysis_id"] = analysis_id
            context.metadata["performance"] = self._tracker.get_summary()

            if progress_callback:
                progress_callback(PipelineStage.COMPLETED, 1, 1, "分析完成")

            return context

        finally:
            smart_cache.finish_analysis(video_path)

    async def run_parallel(self, video_path: str | Path, **metadata) -> AnalysisContext:
        """并行运行Pipeline（部分插件可并行）"""
        video_path = Path(video_path)

        parallel_plugins = [p for p in self.plugins if p.is_parallel_safe and p.enabled]
        sequential_plugins = [p for p in self.plugins if not p.is_parallel_safe and p.enabled]

        context = AnalysisContext(
            video_path=video_path,
            video_name=video_path.name,
            video_md5=smart_cache._smart_cache._compute_file_md5(video_path),
            metadata=metadata,
        )

        for plugin in sequential_plugins:
            context = await plugin.run(context)

        if parallel_plugins:
            tasks = [plugin.run(context) for plugin in parallel_plugins]
            results = await asyncio.gather(*tasks)
            context = results[-1] if results else context

        return context

    def get_pipeline_info(self) -> dict:
        """获取Pipeline信息"""
        return {
            "total_plugins": len(self.plugins),
            "plugins": [
                {
                    "name": p.name,
                    "priority": p.priority,
                    "enabled": p.enabled,
                    "parallel_safe": p.is_parallel_safe,
                }
                for p in self.plugins
            ],
        }


_default_pipeline: HighlightPipeline | None = None


def get_default_pipeline() -> HighlightPipeline:
    """获取默认Pipeline"""
    global _default_pipeline

    if _default_pipeline is None:
        _default_pipeline = HighlightPipeline()

        _default_pipeline.register(SceneDetectorPlugin())
        _default_pipeline.register(KeyframeExtractorPlugin())
        _default_pipeline.register(AudioExtractorPlugin())
        _default_pipeline.register(VLMAnalyzerPlugin())
        _default_pipeline.register(WhisperPlugin())
        _default_pipeline.register(AudioAnalyzerPlugin())
        _default_pipeline.register(HighlightGeneratorPlugin())

    return _default_pipeline
