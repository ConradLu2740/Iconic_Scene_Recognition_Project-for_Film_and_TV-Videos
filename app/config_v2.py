import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DimensionConfig:
    name: str
    weight: float
    min_score: float = 0.0
    max_score: float = 10.0
    description: str = ""


@dataclass
class ScoringConfig:
    dimensions: dict[str, DimensionConfig] = field(default_factory=dict)
    scene_dimension_boosts: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dimensions:
            self.dimensions = {
                "visual_impact": DimensionConfig(
                    name="visual_impact",
                    weight=3.0,
                    description="视觉冲击力 - 画面震撼程度、色彩/光影突出性"
                ),
                "cinematography": DimensionConfig(
                    name="cinematography",
                    weight=2.0,
                    description="镜头语言 - 运镜技巧、角度创新"
                ),
                "emotion_intensity": DimensionConfig(
                    name="emotion_intensity",
                    weight=3.0,
                    description="情感强度 - 情感传达力度、场景氛围"
                ),
                "facial_expression": DimensionConfig(
                    name="facial_expression",
                    weight=3.0,
                    description="面部表情夸张度 - 表情夸张程度"
                ),
                "plot_importance": DimensionConfig(
                    name="plot_importance",
                    weight=2.0,
                    description="剧情重要性 - 关键情节转折点"
                ),
                "action_intensity": DimensionConfig(
                    name="action_intensity",
                    weight=2.0,
                    description="动作强度 - 动作密度和精彩程度"
                ),
                "audio_energy": DimensionConfig(
                    name="audio_energy",
                    weight=2.0,
                    description="音频能量 - 音量峰值和音效震撼度"
                ),
                "memorability": DimensionConfig(
                    name="memorability",
                    weight=2.0,
                    description="记忆点 - 画面/台词的独特性"
                ),
            }

        if not self.scene_dimension_boosts:
            self.scene_dimension_boosts = {
                "action": ["action_intensity", "visual_impact"],
                "drama": ["plot_importance", "emotion_intensity"],
                "emotion": ["emotion_intensity", "facial_expression"],
                "comedy": ["facial_expression", "memorability"],
                "suspense": ["emotion_intensity", "plot_importance"],
                "other": [],
            }

    @property
    def default_weights(self) -> dict[str, float]:
        return {name: dim.weight for name, dim in self.dimensions.items()}

    @property
    def total_weight(self) -> float:
        return sum(dim.weight for dim in self.dimensions.values())

    def get_boosted_weights(self, dimension_boosts: dict[str, float] | None) -> dict[str, float]:
        weights = self.default_weights.copy()
        if dimension_boosts:
            for dim, boost in dimension_boosts.items():
                if dim in weights:
                    weights[dim] = round(weights[dim] * boost, 2)
        return weights


@dataclass
class PipelineConfig:
    merge_time_threshold: float = 3.0
    merge_score_threshold: float = 1.5
    similarity_threshold: float = 0.6
    max_concurrent_vlm: int = 5
    keyframe_samples_per_scene: int = 3
    temporal_context_window: int = 2
    enable_quality_filter: bool = True
    quality_threshold: float = 50.0


@dataclass
class CacheConfig:
    enable_versioning: bool = True
    cache_version: str = "v2"
    max_age_days: int = 30
    enable_incremental: bool = True


class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
        self.VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "30"))
        self.VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus")
        self.VLM_MAX_CONCURRENT = int(os.getenv("VLM_MAX_CONCURRENT", "5"))

        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

        self.SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "6.0"))

        self.UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
        self.KEYFRAME_DIR = BASE_DIR / os.getenv("KEYFRAME_DIR", "keyframes")
        self.CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "cache")
        self.CLIPS_DIR = BASE_DIR / os.getenv("CLIPS_DIR", "clips")
        self.AUDIO_DIR = BASE_DIR / os.getenv("AUDIO_DIR", "audio")
        self.VISUALIZATION_DIR = BASE_DIR / os.getenv("VISUALIZATION_DIR", "visualizations")

        self.MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))

        self.ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

        self.ALLOWED_TYPES = ["action", "drama", "emotion", "comedy", "suspense", "other"]

        self.SCORE_FILTERS = {
            "visual_impact": float(os.getenv("SCORE_FILTER_VISUAL_IMPACT", "0")),
            "cinematography": float(os.getenv("SCORE_FILTER_CINEMATOGRAPHY", "0")),
            "emotion_intensity": float(os.getenv("SCORE_FILTER_EMOTION_INTENSITY", "0")),
            "facial_expression": float(os.getenv("SCORE_FILTER_FACIAL_EXPRESSION", "0")),
            "plot_importance": float(os.getenv("SCORE_FILTER_PLOT_IMPORTANCE", "0")),
            "action_intensity": float(os.getenv("SCORE_FILTER_ACTION_INTENSITY", "0")),
            "audio_energy": float(os.getenv("SCORE_FILTER_AUDIO_ENERGY", "0")),
            "memorability": float(os.getenv("SCORE_FILTER_MEMORABILITY", "0")),
        }

        self.TYPE_FILTER = os.getenv("TYPE_FILTER", "").split(",") if os.getenv("TYPE_FILTER") else []

        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.KEYFRAME_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.CLIPS_DIR.mkdir(exist_ok=True)
        self.AUDIO_DIR.mkdir(exist_ok=True)
        self.VISUALIZATION_DIR.mkdir(exist_ok=True)

        self.scoring = ScoringConfig()
        self.pipeline = PipelineConfig()
        self.cache = CacheConfig()

    def update_from_env(self):
        self.SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", str(self.SCORE_THRESHOLD)))
        self.VLM_MAX_CONCURRENT = int(os.getenv("VLM_MAX_CONCURRENT", str(self.VLM_MAX_CONCURRENT)))
        self.pipeline.max_concurrent_vlm = self.VLM_MAX_CONCURRENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "vlm_model": self.VLM_MODEL,
            "vlm_timeout": self.VLM_TIMEOUT,
            "vlm_max_concurrent": self.VLM_MAX_CONCURRENT,
            "whisper_model": self.WHISPER_MODEL,
            "score_threshold": self.SCORE_THRESHOLD,
            "allowed_types": self.ALLOWED_TYPES,
            "max_video_size_mb": self.MAX_VIDEO_SIZE_MB,
            "scoring_weights": self.scoring.default_weights,
            "scene_boosts": self.scoring.scene_dimension_boosts,
            "pipeline_config": {
                "merge_time_threshold": self.pipeline.merge_time_threshold,
                "merge_score_threshold": self.pipeline.merge_score_threshold,
                "similarity_threshold": self.pipeline.similarity_threshold,
                "max_concurrent_vlm": self.pipeline.max_concurrent_vlm,
                "keyframe_samples_per_scene": self.pipeline.keyframe_samples_per_scene,
                "temporal_context_window": self.pipeline.temporal_context_window,
                "enable_quality_filter": self.pipeline.enable_quality_filter,
                "quality_threshold": self.pipeline.quality_threshold,
            },
        }


config = Config()
