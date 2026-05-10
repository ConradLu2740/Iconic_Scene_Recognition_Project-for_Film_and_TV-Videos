import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameQualityResult:
    frame_position: float
    quality_score: float
    sharpness: float
    contrast: float
    brightness: float
    noise_level: float
    is_blurry: bool
    is_too_dark: bool
    is_too_bright: bool


def calculate_sharpness(image: np.ndarray) -> float:
    """使用拉普拉斯算子计算图像清晰度

    Laplacian算子的方差可以反映图像的边缘清晰程度

    Args:
        image: BGR格式图像

    Returns:
        清晰度分数，越高表示越清晰
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def calculate_contrast(image: np.ndarray) -> float:
    """计算图像对比度

    使用RMS对比度（灰度值标准差）

    Args:
        image: BGR格式图像

    Returns:
        对比度分数
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    return float(std_dev)


def calculate_brightness(image: np.ndarray) -> float:
    """计算图像亮度

    Args:
        image: BGR格式图像

    Returns:
        平均亮度值（0-255）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def calculate_noise_level(image: np.ndarray) -> float:
    """估计图像噪声水平

    使用高频成分的能量作为噪声的近似

    Args:
        image: BGR格式图像

    Returns:
        噪声水平估计值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    noise = np.abs(cv2.filter2D(gray, -1, kernel))
    noise_level = np.mean(noise)

    return float(noise_level)


def analyze_frame_quality(
    frame: np.ndarray,
    frame_position: float = 0.0,
    sharpness_threshold: float = 100.0,
    brightness_low_threshold: float = 30.0,
    brightness_high_threshold: float = 220.0,
) -> FrameQualityResult:
    """综合分析单帧图像质量

    Args:
        frame: BGR格式图像
        frame_position: 帧在视频中的时间位置（秒）
        sharpness_threshold: 清晰度阈值，低于此值判定为模糊
        brightness_low_threshold: 亮度下限，低于此值判定为过暗
        brightness_high_threshold: 亮度上限，高于此值判定为过曝

    Returns:
        帧质量分析结果
    """
    sharpness = calculate_sharpness(frame)
    contrast = calculate_contrast(frame)
    brightness = calculate_brightness(frame)
    noise_level = calculate_noise_level(frame)

    quality_score = _compute_quality_score(
        sharpness, contrast, brightness, noise_level
    )

    return FrameQualityResult(
        frame_position=frame_position,
        quality_score=round(quality_score, 2),
        sharpness=round(sharpness, 2),
        contrast=round(contrast, 2),
        brightness=round(brightness, 2),
        noise_level=round(noise_level, 2),
        is_blurry=(sharpness < sharpness_threshold),
        is_too_dark=(brightness < brightness_low_threshold),
        is_too_bright=(brightness > brightness_high_threshold),
    )


def _compute_quality_score(
    sharpness: float,
    contrast: float,
    brightness: float,
    noise_level: float,
) -> float:
    """综合计算图像质量分数（0-100）

    Args:
        sharpness: 清晰度
        contrast: 对比度
        brightness: 亮度
        noise_level: 噪声水平

    Returns:
        综合质量分数
    """
    sharpness_score = min(100, sharpness / 2)
    contrast_score = min(100, contrast * 2)
    brightness_score = 100 if 50 <= brightness <= 200 else max(0, 100 - abs(brightness - 125) * 2)
    noise_score = max(0, 100 - noise_level * 10)

    quality = (
        sharpness_score * 0.35 +
        contrast_score * 0.25 +
        brightness_score * 0.15 +
        noise_score * 0.25
    )

    return min(100, quality)


def extract_best_keyframes(
    video_path: str | Path,
    scene: object,
    num_samples: int = 3,
    quality_threshold: float = 50.0,
) -> list[tuple[float, str]]:
    """从场景中提取质量最好的关键帧

    不只是取中间帧，而是采样多个帧并选择质量最佳的

    Args:
        video_path: 视频文件路径
        scene: 场景对象，需要有 start_seconds, end_seconds, start_frame, end_frame 属性
        num_samples: 采样数量
        quality_threshold: 最低质量阈值

    Returns:
        [(帧位置秒数, 质量分数), ...] 按质量降序排列
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, min(scene.start_frame, total_frames - 1))
    end_frame = max(0, min(scene.end_frame, total_frames - 1))

    if end_frame <= start_frame:
        cap.release()
        return []

    frame_positions = np.linspace(start_frame, end_frame, num_samples, dtype=int)

    frame_qualities = []

    for frame_idx in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            position_seconds = frame_idx / fps
            quality = analyze_frame_quality(frame, position_seconds)
            frame_qualities.append((position_seconds, quality.quality_score, frame.copy()))

    cap.release()

    frame_qualities.sort(key=lambda x: x[1], reverse=True)

    results = []
    for position, score, frame in frame_qualities:
        if score >= quality_threshold:
            results.append((position, score))

    if not results:
        middle_position = (start_frame + end_frame) / 2 / fps
        results.append((middle_position, 0.0))

    return results[:num_samples]


def smart_sample_scene_frames(
    scene: object,
    num_frames: int = 3,
) -> list[float]:
    """智能采样场景帧位置

    采用分层采样策略，确保覆盖场景的开始、中间、结束

    Args:
        scene: 场景对象，需要有 start_seconds, end_seconds 属性
        num_frames: 采样帧数

    Returns:
        各帧的时间位置列表（秒）
    """
    start = scene.start_seconds
    end = scene.end_seconds
    duration = end - start

    if duration <= 0 or num_frames <= 0:
        return [start]

    if num_frames == 1:
        return [(start + end) / 2]

    if num_frames == 2:
        return [
            start + duration * 0.25,
            start + duration * 0.75,
        ]

    positions = []
    for i in range(num_frames):
        if i == 0:
            pos = start + duration * 0.15
        elif i == num_frames - 1:
            pos = start + duration * 0.85
        else:
            ratio = 0.25 + 0.5 * (i - 1) / (num_frames - 2)
            pos = start + duration * ratio
        positions.append(pos)

    return positions


def filter_frames_by_quality(
    frames: list[tuple[float, np.ndarray]],
    min_quality: float = 30.0,
) -> list[tuple[float, np.ndarray, FrameQualityResult]]:
    """根据质量筛选关键帧

    Args:
        frames: [(时间位置, 帧图像), ...]
        min_quality: 最低质量分数

    Returns:
        满足质量要求的帧及其质量信息
    """
    filtered = []

    for position, frame in frames:
        quality = analyze_frame_quality(frame, position)

        if quality.quality_score >= min_quality:
            filtered.append((position, frame, quality))

    if not filtered:
        best_frame = max(frames, key=lambda x: analyze_frame_quality(x[1], x[0]).quality_score)
        filtered.append((best_frame[0], best_frame[1], analyze_frame_quality(best_frame[1], best_frame[0])))

    return filtered
