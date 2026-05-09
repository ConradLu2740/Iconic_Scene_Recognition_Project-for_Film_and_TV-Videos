import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioEnergyResult:
    start_seconds: float
    end_seconds: float
    energy_score: float
    rms_mean: float
    rms_peak: float
    spectral_centroid_mean: float
    has_speech: bool
    speech_ratio: float


def extract_audio(video_path: str | Path, output_path: str | Path = None) -> str | None:
    """从视频中提取音频流

    Args:
        video_path: 视频文件路径
        output_path: 音频输出路径，默认为视频同目录的同名.wav文件

    Returns:
        音频文件路径，失败返回None
    """
    video_path = Path(video_path)

    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"音频提取成功: {output_path}")
            return str(output_path)
        else:
            logger.error(f"音频提取失败: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("音频提取超时")
        return None
    except Exception as e:
        logger.error(f"音频提取异常: {e}")
        return None


def analyze_audio_segment(
    audio_path: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> AudioEnergyResult:
    """分析指定时间段的音频能量特征

    使用librosa提取音频特征：
    - RMS能量（整体响度）
    - 峰值因子（瞬时强度）
    - 频谱质心（音色亮度）
    - 语音检测（是否有对话）

    Args:
        audio_path: 音频文件路径
        start_seconds: 开始时间
        end_seconds: 结束时间

    Returns:
        音频能量分析结果
    """
    try:
        duration = end_seconds - start_seconds
        y, sr = librosa.load(
            str(audio_path),
            offset=start_seconds,
            duration=duration,
            sr=16000,
        )

        if len(y) == 0:
            return AudioEnergyResult(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                energy_score=0.0,
                rms_mean=0.0,
                rms_peak=0.0,
                spectral_centroid_mean=0.0,
                has_speech=False,
                speech_ratio=0.0,
            )

        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_peak = float(np.max(rms))

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = float(np.mean(spectral_centroids))

        hop_length = 512
        energy = np.array([
            sum(abs(y[i * hop_length:(i + 1) * hop_length] ** 2))
            for i in range(len(y) // hop_length)
        ])

        speech_ratio = _detect_speech_ratio(y, sr)

        energy_score = _compute_energy_score(rms_mean, rms_peak, spectral_centroid_mean, speech_ratio)

        return AudioEnergyResult(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            energy_score=round(energy_score, 2),
            rms_mean=round(rms_mean, 4),
            rms_peak=round(rms_peak, 4),
            spectral_centroid_mean=round(spectral_centroid_mean, 2),
            has_speech=(speech_ratio > 0.3),
            speech_ratio=round(speech_ratio, 2),
        )

    except Exception as e:
        logger.error(f"音频分析异常 [{start_seconds}-{end_seconds}s]: {e}")
        return AudioEnergyResult(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            energy_score=0.0,
            rms_mean=0.0,
            rms_peak=0.0,
            spectral_centroid_mean=0.0,
            has_speech=False,
            speech_ratio=0.0,
        )


def _compute_energy_score(
    rms_mean: float,
    rms_peak: float,
    spectral_centroid: float,
    speech_ratio: float,
) -> float:
    """综合计算音频能量分数（0-10）

    Args:
        rms_mean: 平均RMS能量
        rms_peak: 峰值RMS
        spectral_centroid: 平均频谱质心
        speech_ratio: 语音占比

    Returns:
        归一化的能量分数
    """
    rms_score = min(10, rms_mean * 100)
    peak_score = min(10, rms_peak * 150)
    centroid_score = min(10, spectral_centroid / 500)

    speech_bonus = 2.0 if speech_ratio > 0.5 else 0.0

    raw_score = (rms_score * 0.4 + peak_score * 0.4 + centroid_score * 0.2 + speech_bonus)
    return min(10, raw_score)


def _detect_speech_ratio(y: np.ndarray, sr: int) -> float:
    """简单语音检测：基于能量和频谱特征

    简化实现，不依赖语音识别模型

    Args:
        y: 音频波形
        sr: 采样率

    Returns:
        语音占比（0-1）
    """
    try:
        frame_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)

        energy = np.array([
            np.sum(y[i * hop_length:i * hop_length + frame_length] ** 2)
            for i in range(len(y) // hop_length - 1)
        ])

        if len(energy) == 0:
            return 0.0

        energy_threshold = np.percentile(energy, 30)

        speech_frames = np.sum(energy > energy_threshold)
        total_frames = len(energy)

        return speech_frames / total_frames if total_frames > 0 else 0.0

    except Exception:
        return 0.5


def batch_analyze_audio_segments(
    audio_path: str | Path,
    segments: list[tuple[float, float]],
) -> list[AudioEnergyResult]:
    """批量分析多个音频片段

    Args:
        audio_path: 音频文件路径
        segments: [(start1, end1), (start2, end2), ...]

    Returns:
        各片段的分析结果列表
    """
    results = []
    for start, end in segments:
        result = analyze_audio_segment(audio_path, start, end)
        results.append(result)
    return results


def extract_audio_features_for_scenes(
    video_path: str | Path,
    scenes: list,
    audio_cache_dir: Path | None = None,
) -> dict[int, AudioEnergyResult]:
    """为每个场景提取音频特征

    Args:
        video_path: 视频文件路径
        scenes: 场景列表，每个元素有 start_seconds, end_seconds 属性
        audio_cache_dir: 音频缓存目录

    Returns:
        {场景索引: AudioEnergyResult} 的字典
    """
    audio_path = _get_audio_path(video_path, audio_cache_dir)

    if audio_path is None or not Path(audio_path).exists():
        logger.warning("音频文件不可用，跳过音频能量分析")
        return {}

    results = {}
    for idx, scene in enumerate(scenes):
        result = analyze_audio_segment(
            audio_path,
            scene.start_seconds,
            scene.end_seconds,
        )
        results[idx] = result

    return results


def _get_audio_path(video_path: Path, cache_dir: Path | None = None) -> str | None:
    """获取音频文件路径

    如果指定了缓存目录，优先从缓存目录查找同名文件
    """
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_audio = cache_dir / f"{video_path.stem}.wav"

        if cached_audio.exists():
            return str(cached_audio)

        audio_path = extract_audio(video_path, cached_audio)
        return audio_path
    else:
        audio_path = extract_audio(video_path)
        return audio_path
