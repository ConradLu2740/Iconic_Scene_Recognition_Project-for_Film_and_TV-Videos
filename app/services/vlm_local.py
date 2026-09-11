import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LocalVLMResult:
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

    @property
    def total_score(self) -> float:
        weights = {
            "visual_impact": 3, "cinematography": 2, "emotion_intensity": 3,
            "facial_expression": 3, "plot_importance": 2, "action_intensity": 2,
            "audio_energy": 2, "memorability": 2,
        }
        total = sum(weights.values())
        return round(
            self.visual_impact * 3 + self.cinematography * 2 +
            self.emotion_intensity * 3 + self.facial_expression * 3 +
            self.plot_importance * 2 + self.action_intensity * 2 +
            self.audio_energy * 2 + self.memorability * 2
        ) / total, 2


class VLMBackend(ABC):
    """VLM后端抽象基类"""

    @abstractmethod
    def analyze(self, image_path: str) -> LocalVLMResult | None:
        """分析单张图片"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class RuleBasedVLM(VLMBackend):
    """基于规则的VLM（无需模型）

    使用传统计算机视觉方法分析图片特征
    """

    def __init__(self):
        self.face_cascade = None
        self._load_cascade()

    def _load_cascade(self):
        """加载OpenCV级联分类器"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("人脸检测分类器加载成功")
            else:
                logger.warning("人脸检测分类器未找到")
        except Exception as e:
            logger.warning(f"加载级联分类器失败: {e}")

    def is_available(self) -> bool:
        return True

    def analyze(self, image_path: str) -> LocalVLMResult | None:
        """基于规则分析图片"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            visual_impact = self._analyze_visual_impact(image)
            cinematography = self._analyze_cinematography(image)
            emotion_intensity = self._analyze_emotion_intensity(image, gray)
            facial_expression = self._analyze_facial_expression(gray)
            plot_importance = self._analyze_plot_importance(image, gray)
            action_intensity = self._analyze_action_intensity(image, gray)
            audio_energy = self._estimate_audio_energy(image)
            memorability = self._analyze_memorability(image, gray)

            scene_type = self._classify_scene_type(
                action_intensity, emotion_intensity, facial_expression
            )

            description = self._generate_description(
                scene_type, facial_expression, action_intensity
            )

            return LocalVLMResult(
                visual_impact=round(visual_impact, 1),
                cinematography=round(cinematography, 1),
                emotion_intensity=round(emotion_intensity, 1),
                facial_expression=round(facial_expression, 1),
                plot_importance=round(plot_importance, 1),
                action_intensity=round(action_intensity, 1),
                audio_energy=round(audio_energy, 1),
                memorability=round(memorability, 1),
                type=scene_type,
                description=description,
            )

        except Exception as e:
            logger.error(f"规则分析失败: {e}")
            return None

    def _analyze_visual_impact(self, image: np.ndarray) -> float:
        """分析视觉冲击力"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = np.mean(hsv[:, :, 1])
        brightness = np.mean(hsv[:, :, 2])

        saturation_score = min(10, saturation / 25.5)
        brightness_score = min(10, abs(brightness - 128) / 12.8)

        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(10, edge_density * 500)

        color_variance = np.var(image)
        color_score = min(10, color_variance / 1000)

        return (saturation_score * 0.3 + brightness_score * 0.3 +
                edge_score * 0.2 + color_score * 0.2)

    def _analyze_cinematography(self, image: np.ndarray) -> float:
        """分析镜头语言"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        line_score = 5.0
        if lines is not None and len(lines) > 0:
            line_score = min(10, 5 + len(lines) * 0.1)

        brightness_gradient = np.gradient(gray.mean(axis=1))
        gradient_score = min(10, np.std(gradient_gradient) * 2)

        aspect = image.shape[1] / image.shape[0] if image.shape[0] > 0 else 1.0
        composition_score = 5.0 if 0.5 <= aspect <= 2.0 else 7.0

        return (line_score * 0.4 + gradient_score * 0.3 + composition_score * 0.3)

    def _analyze_emotion_intensity(self, image: np.ndarray, gray: np.ndarray) -> float:
        """分析情感强度"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        warm_colors = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([45, 255, 255]))
        warm_ratio = np.sum(warm_colors > 0) / warm_colors.size

        cool_colors = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([150, 255, 255]))
        cool_ratio = np.sum(cool_colors > 0) / cool_colors.size

        warm_score = warm_ratio * 20
        cool_score = cool_ratio * 15

        brightness_var = np.var(gray)
        contrast = np.max(gray) - np.min(gray)
        contrast_score = min(10, contrast / 25.5)

        return (warm_score + cool_score + contrast_score) / 3

    def _analyze_facial_expression(self, gray: np.ndarray) -> float:
        """分析面部表情"""
        if self.face_cascade is None:
            return 5.0

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return 5.0

        face_scores = []
        for x, y, w, h in faces:
            face_roi = gray[y:y + h, x:x + w]

            face_size_ratio = (w * h) / (gray.shape[0] * gray.shape[1])
            size_score = min(10, face_size_ratio * 200)

            face_roi_resized = cv2.resize(face_roi, (50, 50))
            brightness_var = np.var(face_roi_resized)
            expression_score = min(10, brightness_var / 50)

            face_scores.append((size_score + expression_score) / 2)

        return max(face_scores) if face_scores else 5.0

    def _analyze_plot_importance(self, image: np.ndarray, gray: np.ndarray) -> float:
        """分析剧情重要性"""
        brightness = np.mean(gray)

        center_region = gray[
            gray.shape[0] // 4:3 * gray.shape[0] // 4,
            gray.shape[1] // 4:3 * gray.shape[1] // 4
        ]
        center_brightness = np.mean(center_region)

        if brightness < 50:
            return 6.0 + (50 - brightness) / 10

        if abs(center_brightness - brightness) > 20:
            return 7.5

        return 5.0

    def _analyze_action_intensity(self, image: np.ndarray, gray: np.ndarray) -> float:
        """分析动作强度"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        optical_flow = self._estimate_motion(image)
        motion_score = min(10, optical_flow * 50)

        brightness_changes = np.abs(np.diff(gray.astype(float), axis=0))
        change_score = min(10, np.mean(brightness_changes) * 2)

        return (edge_density * 300 + motion_score + change_score) / 3

    def _estimate_motion(self, image: np.ndarray) -> float:
        """估计运动程度"""
        blur = cv2.GaussianBlur(image, (15, 15), 0)
        diff = cv2.absdiff(image, blur)
        motion = np.mean(diff)
        return min(1.0, motion / 30)

    def _estimate_audio_energy(self, image: np.ndarray) -> float:
        """估计音频能量（基于画面）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        saturation = np.mean(hsv[:, :, 1])
        brightness = np.mean(hsv[:, :, 2])

        saturation_score = min(10, saturation / 25.5)
        brightness_score = min(10, abs(brightness - 128) / 12.8)

        return (saturation_score + brightness_score) / 2 + 3

    def _analyze_memorability(self, image: np.ndarray, gray: np.ndarray) -> float:
        """分析记忆点"""
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        color_diversity_score = min(10, unique_colors / 1000)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(10, edge_density * 300)

        brightness = np.mean(gray)
        unusual_brightness = abs(brightness - 128) > 50
        novelty_score = 8.0 if unusual_brightness else 5.0

        return (color_diversity_score * 0.3 + edge_score * 0.4 + novelty_score * 0.3)

    def _classify_scene_type(
        self,
        action: float,
        emotion: float,
        facial: float,
    ) -> str:
        """分类场景类型"""
        scores = {
            "action": action,
            "emotion": emotion + facial,
            "comedy": facial * 0.8,
            "suspense": action * 0.7 + emotion * 0.3,
            "drama": emotion + facial + 2,
        }

        max_type = max(scores, key=scores.get)
        if scores[max_type] < 5:
            return "other"
        return max_type

    def _generate_description(
        self,
        scene_type: str,
        facial: float,
        action: float,
    ) -> str:
        """生成描述"""
        descriptions = {
            "action": "动作场面",
            "emotion": "情感场景",
            "comedy": "喜剧场景",
            "suspense": "悬疑紧张",
            "drama": "戏剧场景",
            "other": "一般场景",
        }

        base = descriptions.get(scene_type, "一般场景")

        if facial > 7:
            base = f"夸张表情的{base}"
        elif action > 7:
            base = f"激烈的{base}"

        return base


class HuggingFaceVLM(VLMBackend):
    """HuggingFace VLM后端

    使用 transformers 库的视觉语言模型
    """

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering

            logger.info(f"加载 HuggingFace 模型: {self.model_name}")
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForQuestionAnswering.from_pretrained(self.model_name)
            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model = None
            self._processor = None

    def analyze(self, image_path: str) -> LocalVLMResult | None:
        if not self.is_available():
            return None

        self._load_model()

        if self._model is None:
            return None

        try:
            from PIL import Image
            import torch

            raw_image = Image.open(image_path).convert("rgb")

            questions = {
                "visual_impact": "Rate the visual impact from 0-10",
                "emotion": "Rate the emotion intensity from 0-10",
                "action": "Rate the action intensity from 0-10",
            }

            results = {}
            for key, question in questions.items():
                inputs = self._processor(raw_image, question, return_tensors="pt")
                output = self._model.generate(**inputs)
                answer = self._processor.decode(output[0], skip_special_tokens=True)

                try:
                    score = float(''.join(filter(lambda x: x.isdigit() or x == '.', answer)))
                    results[key] = min(10, max(0, score))
                except:
                    results[key] = 5.0

            return LocalVLMResult(
                visual_impact=results.get("visual_impact", 5.0),
                cinematography=5.0,
                emotion_intensity=results.get("emotion", 5.0),
                facial_expression=5.0,
                plot_importance=5.0,
                action_intensity=results.get("action", 5.0),
                audio_energy=5.0,
                memorability=5.0,
                type="other",
                description="AI分析场景",
            )

        except Exception as e:
            logger.error(f"HuggingFace 分析失败: {e}")
            return None


class VLMFactory:
    """VLM工厂类"""

    _backends = {}

    @classmethod
    def register(cls, name: str, backend_class: type):
        cls._backends[name] = backend_class

    @classmethod
    def create(cls, backend: Literal["dashscope", "rule", "huggingface"] = "rule") -> VLMBackend:
        if backend == "dashscope":
            from app.services.vlm_analyzer import VLMResult

            class DashScopeBackend(VLMBackend):
                def __init__(self):
                    self._analyzer = None

                def is_available(self) -> bool:
                    from app.config import DASHSCOPE_API_KEY
                    return bool(DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "your_api_key_here")

                def analyze(self, image_path: str):
                    from app.services.vlm_analyzer import analyze_keyframe
                    result = analyze_keyframe(image_path)
                    if result:
                        return LocalVLMResult(
                            visual_impact=result.visual_impact,
                            cinematography=result.cinematography,
                            emotion_intensity=result.emotion_intensity,
                            facial_expression=result.facial_expression,
                            plot_importance=result.plot_importance,
                            action_intensity=result.action_intensity,
                            audio_energy=result.audio_energy,
                            memorability=result.memorability,
                            type=result.type,
                            description=result.description,
                        )
                    return None

            return DashScopeBackend()

        elif backend == "rule":
            return RuleBasedVLM()

        elif backend == "huggingface":
            return HuggingFaceVLM()

        else:
            raise ValueError(f"未知的 VLM 后端: {backend}")


VLMFactory.register("dashscope", None)
VLMFactory.register("rule", RuleBasedVLM)
VLMFactory.register("huggingface", HuggingFaceVLM)


def get_vlm_backend() -> VLMBackend:
    """获取可用的VLM后端

    优先使用DashScope，失败则降级到规则引擎
    """
    backend = os.getenv("VLM_BACKEND", "auto")

    if backend == "auto":
        dashscope = VLMFactory.create("dashscope")
        if dashscope.is_available():
            return dashscope

        logger.warning("DashScope 不可用，降级到规则引擎")
        return VLMFactory.create("rule")

    return VLMFactory.create(backend)
