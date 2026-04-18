import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import WHISPER_MODEL

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = WHISPER_MODEL
WHISPER_DEVICE = "auto"

_whisper_model = None


def get_whisper_model(model_size: str = None):
    """获取或初始化Whisper模型单例

    Args:
        model_size: 可选的模型大小，默认使用配置中的值
                   可选值: tiny, base, small, medium, large
    """
    global _whisper_model
    target_size = model_size or WHISPER_MODEL_SIZE

    if _whisper_model is None or target_size != WHISPER_MODEL_SIZE:
        logger.info(f"加载Whisper模型: {target_size}")
        _whisper_model = WhisperModel(
            target_size,
            device=WHISPER_DEVICE,
            compute_type="int8",
        )
        logger.info("Whisper模型加载完成")
    return _whisper_model


@dataclass
class SegmentTranscript:
    """语音片段转写结果"""
    start_seconds: float
    end_seconds: float
    text: str
    words: list


@dataclass
class FullTranscript:
    """完整视频语音转写结果"""
    full_text: str
    segments: list[SegmentTranscript]
    language: str | None


def transcribe_video(video_path: str | Path, progress_callback=None) -> FullTranscript | None:
    """使用Faster-Whisper对视频进行语音转写

    Args:
        video_path: 视频文件路径
        progress_callback: 进度回调函数

    Returns:
        完整转写结果，包含各时间段文字
    """
    try:
        if progress_callback:
            progress_callback("语音转写中...", 0, 1)

        model = get_whisper_model()
        logger.info(f"开始语音转写: {video_path}")

        segments, info = model.transcribe(
            str(video_path),
            language="zh",
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        segment_list = []
        full_text_parts = []

        for seg in segments:
            seg_text = seg.text.strip()
            if seg_text:
                segment_list.append(SegmentTranscript(
                    start_seconds=seg.start,
                    end_seconds=seg.end,
                    text=seg_text,
                    words=[],
                ))
                full_text_parts.append(seg_text)

        full_text = " ".join(full_text_parts)
        language = info.language if hasattr(info, 'language') else None

        logger.info(f"语音转写完成: {len(segment_list)}个片段, 语言: {language}")

        if progress_callback:
            progress_callback("语音转写完成", 1, 1)

        return FullTranscript(
            full_text=full_text,
            segments=segment_list,
            language=language,
        )

    except Exception as e:
        logger.error(f"语音转写失败: {e}")
        return None


def get_transcript_for_time_range(
    transcript: FullTranscript,
    start_seconds: float,
    end_seconds: float
) -> str:
    """获取指定时间范围内的语音文字

    Args:
        transcript: 完整转写结果
        start_seconds: 开始时间
        end_seconds: 结束时间

    Returns:
        时间范围内的语音文字
    """
    if not transcript or not transcript.segments:
        return ""

    relevant_texts = []
    for seg in transcript.segments:
        if seg.start_seconds >= start_seconds - 0.5 and seg.end_seconds <= end_seconds + 0.5:
            relevant_texts.append(seg.text)

    return " ".join(relevant_texts)
