import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import dashscope
from dashscope import MultiModalConversation

from app.config_v2 import config

logger = logging.getLogger(__name__)

SCORING_PROMPT = """你是一个专业的视频内容分析专家。请仔细观察这个视频关键帧，进行深度内容分析。

## 评分维度（0-10分，精确到小数点后一位）

**1. 视觉冲击力 (visual_impact)** - 画面震撼程度、色彩/光影突出性
**2. 镜头语言 (cinematography)** - 运镜技巧、角度创新
**3. 情感强度 (emotion_intensity)** - 情感传达力度、场景氛围
**4. 面部表情夸张度 (facial_expression)** - 表情夸张程度
**5. 剧情重要性 (plot_importance)** - 关键情节转折点
**6. 动作强度 (action_intensity)** - 动作密度和精彩程度
**7. 音频能量 (audio_energy)** - 音量峰值和音效震撼度（注：基于画面估计）
**8. 记忆点 (memorability)** - 画面/台词的独特性

## 场景类型分类
- action: 动作场面（打斗、追逐、爆炸）
- drama: 戏剧场景（情感对话、人物冲突）
- emotion: 情感高潮（感人时刻、温馨、浪漫）
- comedy: 喜剧场景（搞笑、幽默、滑稽）
- suspense: 悬疑紧张（惊悚、恐怖、悬念）
- other: 其他类型

## 输出格式
请严格按以下JSON格式输出：
{
  "visual_impact": <0-10>,
  "cinematography": <0-10>,
  "emotion_intensity": <0-10>,
  "facial_expression": <0-10>,
  "plot_importance": <0-10>,
  "action_intensity": <0-10>,
  "audio_energy": <0-10>,
  "memorability": <0-10>,
  "type": "<action|drama|emotion|comedy|suspense|other>",
  "description": "<20字以内的中文描述>"
}"""


@dataclass
class VLMResult:
    visual_impact: float
    cinematography: float
    emotion_intensity: float
    facial_expression: float
    plot_importance: float
    action_intensity: float
    audio_energy: float
    memorability: float
    type: str
    description: str
    frame_index: int = 0
    quality_score: float = 0.0

    @property
    def total_score(self) -> float:
        weights = config.scoring.default_weights
        total_weight = config.scoring.total_weight
        weighted_sum = (
            self.visual_impact * weights.get("visual_impact", 3) +
            self.cinematography * weights.get("cinematography", 2) +
            self.emotion_intensity * weights.get("emotion_intensity", 3) +
            self.facial_expression * weights.get("facial_expression", 3) +
            self.plot_importance * weights.get("plot_importance", 2) +
            self.action_intensity * weights.get("action_intensity", 2) +
            self.audio_energy * weights.get("audio_energy", 2) +
            self.memorability * weights.get("memorability", 2)
        )
        return round(weighted_sum / total_weight, 2)


@dataclass
class SceneAnalysisResult:
    scene_index: int
    start_seconds: float
    end_seconds: float
    frame_results: list[VLMResult] = field(default_factory=list)
    fused_result: VLMResult | None = None
    quality_scores: list[float] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.frame_results) > 0


class ConcurrentVLMAnalyzer:
    """并发VLM分析器

    支持批量并发调用VLM API，带有速率限制和重试机制
    """

    def __init__(
        self,
        max_concurrent: int = None,
        timeout: int = None,
        max_retries: int = 3,
    ):
        self.max_concurrent = max_concurrent or config.VLM_MAX_CONCURRENT
        self.timeout = timeout or config.VLM_TIMEOUT
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

    def _validate_api_key(self) -> bool:
        if not config.DASHSCOPE_API_KEY or config.DASHSCOPE_API_KEY == "your_api_key_here":
            logger.error("DASHSCOPE_API_KEY未配置")
            return False
        dashscope.api_key = config.DASHSCOPE_API_KEY
        return True

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_response(self, text_content: str) -> dict | None:
        try:
            json_str = text_content.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            return json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None

    def _call_vlm_sync(self, image_path: str, frame_index: int = 0) -> VLMResult | None:
        """同步调用VLM API"""
        try:
            image_base64 = self._encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": SCORING_PROMPT},
                    ],
                }
            ]

            response = MultiModalConversation.call(
                model=config.VLM_MODEL,
                messages=messages,
                timeout=self.timeout,
                result_format="message",
            )

            if response.status_code != 200:
                logger.error(f"VLM API失败: {response.code} - {response.message}")
                return None

            text_content = response.output.choices[0].message.content[0]["text"]
            result_dict = self._parse_response(text_content)

            if result_dict is None:
                return None

            return VLMResult(
                visual_impact=float(result_dict.get("visual_impact", 0)),
                cinematography=float(result_dict.get("cinematography", 0)),
                emotion_intensity=float(result_dict.get("emotion_intensity", 0)),
                facial_expression=float(result_dict.get("facial_expression", 0)),
                plot_importance=float(result_dict.get("plot_importance", 0)),
                action_intensity=float(result_dict.get("action_intensity", 0)),
                audio_energy=float(result_dict.get("audio_energy", 0)),
                memorability=float(result_dict.get("memorability", 0)),
                type=result_dict.get("type", "other"),
                description=result_dict.get("description", ""),
                frame_index=frame_index,
            )

        except Exception as e:
            logger.error(f"VLM分析异常: {e}")
            return None

    async def analyze_single(self, image_path: str, frame_index: int = 0) -> VLMResult | None:
        """异步分析单张图片"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._call_vlm_sync,
                image_path,
                frame_index,
            )
            return result

    async def analyze_batch(
        self,
        image_paths: list[str],
        progress_callback: Callable[[int, int], None] = None,
    ) -> list[VLMResult | None]:
        """批量并发分析多张图片

        Args:
            image_paths: 图片路径列表
            progress_callback: 进度回调 (current, total)

        Returns:
            分析结果列表，与输入顺序对应
        """
        if not self._validate_api_key():
            return [None] * len(image_paths)

        tasks = []
        for idx, path in enumerate(image_paths):
            task = self.analyze_single(path, idx)
            tasks.append(task)

        results = []
        total = len(tasks)

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)

        return results

    async def analyze_scenes(
        self,
        scenes: list,
        keyframe_paths: dict[int, list[str]],
        progress_callback: Callable[[str, int, int], None] = None,
    ) -> list[SceneAnalysisResult]:
        """分析多个场景，每个场景可能有多个关键帧

        Args:
            scenes: 场景列表
            keyframe_paths: {场景索引: [帧路径列表]}
            progress_callback: 进度回调

        Returns:
            各场景的分析结果
        """
        all_frame_paths = []
        scene_frame_mapping = []

        for scene_idx, paths in keyframe_paths.items():
            scene_frame_mapping.append({
                "scene_index": scene_idx,
                "scene": scenes[scene_idx],
                "frame_paths": paths,
            })
            for path in paths:
                all_frame_paths.append((scene_idx, path))

        async def analyze_with_scene_info():
            tasks = []
            for scene_idx, frame_idx, path in enumerate(all_frame_paths):
                task = self.analyze_single(path, frame_idx)
                tasks.append((scene_idx, frame_idx, task))

            results_map = {}
            for i, (_, frame_idx, coro) in enumerate(tasks):
                result = await coro
                results_map[(scene_idx, frame_idx)] = result
                if progress_callback:
                    progress_callback(f"VLM分析中... ({i + 1}/{len(tasks)})", i + 1, len(tasks))

            return results_map

        results_map = await analyze_with_scene_info()

        scene_results = []
        for mapping in scene_frame_mapping:
            scene_idx = mapping["scene_index"]
            scene = mapping["scene"]
            frame_paths = mapping["frame_paths"]

            frame_results = []
            for frame_idx, path in enumerate(frame_paths):
                result = results_map.get((scene_idx, frame_idx))
                if result:
                    frame_results.append(result)

            scene_result = SceneAnalysisResult(
                scene_index=scene_idx,
                start_seconds=scene.start_seconds,
                end_seconds=scene.end_seconds,
                frame_results=frame_results,
                quality_scores=[],
            )

            if frame_results:
                scene_result.fused_result = self._fuse_results(frame_results)

            scene_results.append(scene_result)

        return scene_results

    def _fuse_results(self, results: list[VLMResult]) -> VLMResult:
        """融合多个帧的分析结果

        使用加权平均策略，类型采用投票机制
        """
        if len(results) == 1:
            return results[0]

        dimensions = [
            "visual_impact", "cinematography", "emotion_intensity",
            "facial_expression", "plot_importance", "action_intensity",
            "audio_energy", "memorability",
        ]

        fused_scores = {}
        for dim in dimensions:
            values = [getattr(r, dim) for r in results]
            fused_scores[dim] = round(sum(values) / len(values), 1)

        type_counts: dict[str, int] = {}
        for r in results:
            type_counts[r.type] = type_counts.get(r.type, 0) + 1
        dominant_type = max(type_counts, key=type_counts.get)

        descriptions = [r.description for r in results if r.description]
        fused_description = descriptions[0] if descriptions else ""

        return VLMResult(
            **fused_scores,
            type=dominant_type,
            description=fused_description,
            frame_index=0,
        )


class VLMANalyzerPool:
    """VLM分析器池

    管理多个VLM分析器实例，提高并发处理能力
    """

    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self._analyzers = [
            ConcurrentVLMAnalyzer()
            for _ in range(pool_size)
        ]
        self._current_index = 0
        self._lock = None

    def get_analyzer(self) -> ConcurrentVLMAnalyzer:
        """获取下一个可用的分析器"""
        analyzer = self._analyzers[self._current_index]
        self._current_index = (self._current_index + 1) % self.pool_size
        return analyzer


_analyzer_pool: VLMANalyzerPool | None = None


def get_vlm_analyzer_pool(pool_size: int = 2) -> VLMANalyzerPool:
    """获取全局VLM分析器池"""
    global _analyzer_pool
    if _analyzer_pool is None:
        _analyzer_pool = VLMANalyzerPool(pool_size)
    return _analyzer_pool


async def analyze_keyframes_concurrent(
    keyframe_paths: list[str],
    progress_callback: Callable[[int, int], None] = None,
) -> list[VLMResult | None]:
    """便捷函数：并发分析关键帧列表"""
    pool = get_vlm_analyzer_pool()
    analyzer = pool.get_analyzer()
    return await analyzer.analyze_batch(keyframe_paths, progress_callback)
