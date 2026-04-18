import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from app.config import SCORE_THRESHOLD, KEYFRAME_DIR
from app.services.scene_detector import detect_scenes
from app.services.keyframe_extractor import extract_keyframes
from app.services.vlm_analyzer import analyze_keyframe
from app.services.whisper_transcriber import transcribe_video, get_transcript_for_time_range
from app.utils.cache import compute_file_md5, get_cached_result, save_cached_result
from app.utils.video_utils import format_seconds

logger = logging.getLogger(__name__)

MERGE_TIME_THRESHOLD = 3.0
MERGE_SCORE_THRESHOLD = 1.5
SIMILARITY_THRESHOLD = 0.6


def _merge_consecutive_highlights(highlights: list[dict]) -> list[dict]:
    """合并时间上连续且评分相近的高分片段

    Args:
        highlights: 原始高分片段列表，按时间排序

    Returns:
        合并后的片段列表
    """
    if not highlights:
        return []

    merged = []
    current = highlights[0].copy()

    for next_hl in highlights[1:]:
        time_gap = next_hl["start_seconds"] - current["end_seconds"]
        score_diff = abs(next_hl["scores"]["total"] - current["scores"]["total"])

        if time_gap <= MERGE_TIME_THRESHOLD and score_diff <= MERGE_SCORE_THRESHOLD:
            current["end_seconds"] = next_hl["end_seconds"]
            current["end_time"] = next_hl["end_time"]
            current["scores"]["total"] = round(
                (current["scores"]["total"] + next_hl["scores"]["total"]) / 2, 2
            )
            if next_hl["scores"]["total"] > current["scores"]["total"]:
                current["keyframe_url"] = next_hl["keyframe_url"]
                current["description"] = next_hl["description"]
                current["type"] = next_hl["type"]
        else:
            merged.append(current)
            current = next_hl.copy()

    merged.append(current)
    return merged


def _filter_similar_scenes(highlights: list[dict]) -> list[dict]:
    """过滤掉描述和类型高度相似的重复场景

    Args:
        highlights: 合并后的片段列表

    Returns:
        去重后的片段列表
    """
    if not highlights:
        return []

    filtered = []
    for hl in highlights:
        is_duplicate = False
        for existing in filtered:
            if _is_similar(hl, existing):
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(hl)

    return filtered


def _is_similar(hl1: dict, hl2: dict) -> bool:
    """判断两个场景是否相似

    通过类型相同 + 描述关键词重叠判断
    """
    if hl1["type"] != hl2["type"]:
        return False

    desc1 = set(hl1["description"])
    desc2 = set(hl2["description"])
    overlap = len(desc1 & desc2) / max(len(desc1 | desc2), 1)

    return overlap >= SIMILARITY_THRESHOLD


def _filter_by_type(highlights: list[dict], allowed_types: list[str]) -> list[dict]:
    """按场景类型过滤

    Args:
        highlights: 片段列表
        allowed_types: 允许的类型列表，空列表表示不过滤

    Returns:
        过滤后的片段列表
    """
    if not allowed_types:
        return highlights
    return [hl for hl in highlights if hl["type"] in allowed_types]


def _filter_by_scores(highlights: list[dict], score_filters: dict) -> list[dict]:
    """按各维度分数过滤

    Args:
        highlights: 片段列表
        score_filters: 各维度最低分数要求，格式 {dimension: min_score}

    Returns:
        满足所有分数要求的片段列表
    """
    if not score_filters or all(v == 0 for v in score_filters.values()):
        return highlights

    filtered = []
    for hl in highlights:
        scores = hl.get("scores", {})
        meets_all = True
        for dim, min_score in score_filters.items():
            if min_score > 0 and scores.get(dim, 0) < min_score:
                meets_all = False
                break
        if meets_all:
            filtered.append(hl)

    return filtered


def analyze_video(video_path: str, progress_callback=None, type_filter: list[str] = None, score_filters: dict = None) -> dict:
    """执行完整的视频"名场面"分析流水线

    流程: MD5去重检查 → 场景分割 → 关键帧提取 → VLM评分 → 语音转写 → 筛选结果 → 合并去重

    Args:
        video_path: 视频文件路径
        progress_callback: 进度回调函数，接收(step_name, current, total)参数

    Returns:
        包含名场面列表的完整分析结果字典
    """
    video_path = Path(video_path)
    video_name = video_path.name

    md5_hash = compute_file_md5(video_path)
    cached = get_cached_result(md5_hash)
    if cached:
        logger.info(f"命中缓存，跳过分析: {video_name}")
        if progress_callback:
            progress_callback("缓存命中", 1, 1)
        return cached

    if progress_callback:
        progress_callback("场景检测中...", 0, 4)

    logger.info(f"开始场景检测: {video_name}")
    scenes = detect_scenes(video_path)
    logger.info(f"检测到 {len(scenes)} 个场景片段")

    if not scenes:
        result = _build_empty_result(video_name)
        save_cached_result(md5_hash, result)
        return result

    if progress_callback:
        progress_callback("提取关键帧中...", 1, 4)

    logger.info("开始提取关键帧")
    keyframe_paths = extract_keyframes(video_path, scenes, md5_hash)
    logger.info(f"提取了 {len(keyframe_paths)} 个关键帧")

    if progress_callback:
        progress_callback("VLM智能评分中...", 2, 5)

    highlights = []
    total_keyframes = len(keyframe_paths)

    for idx, (scene, kf_path) in enumerate(zip(scenes, keyframe_paths)):
        if not kf_path:
            logger.warning(f"场景 {idx + 1} 关键帧提取失败，跳过")
            continue

        logger.info(f"分析关键帧 {idx + 1}/{total_keyframes}")
        vlm_result = analyze_keyframe(kf_path)

        if vlm_result is None:
            logger.warning(f"场景 {idx + 1} VLM分析失败，跳过")
            continue

        relative_kf_path = str(Path(kf_path).relative_to(KEYFRAME_DIR.parent))
        relative_kf_url = relative_kf_path.replace(os.sep, "/")

        highlight = {
            "id": idx + 1,
            "start_time": format_seconds(scene.start_seconds),
            "end_time": format_seconds(scene.end_seconds),
            "start_seconds": round(scene.start_seconds, 2),
            "end_seconds": round(scene.end_seconds, 2),
            "keyframe_url": f"/{relative_kf_url}",
            "scores": {
                "visual_impact": vlm_result.visual_impact,
                "cinematography": vlm_result.cinematography,
                "emotion_intensity": vlm_result.emotion_intensity,
                "facial_expression": vlm_result.facial_expression,
                "plot_importance": vlm_result.plot_importance,
                "action_intensity": vlm_result.action_intensity,
                "audio_energy": vlm_result.audio_energy,
                "memorability": vlm_result.memorability,
                "total": round(vlm_result.total_score, 2),
            },
            "type": vlm_result.type,
            "description": vlm_result.description,
        }

        if vlm_result.total_score >= SCORE_THRESHOLD:
            highlights.append(highlight)

    if progress_callback:
        progress_callback("语音转写中...", 3, 5)

    transcript = transcribe_video(video_path)
    if transcript:
        for hl in highlights:
            hl["transcript"] = get_transcript_for_time_range(
                transcript, hl["start_seconds"], hl["end_seconds"]
            )
        logger.info("语音转写完成，已匹配到各片段")
    else:
        for hl in highlights:
            hl["transcript"] = ""
        logger.warning("语音转写失败")

    if progress_callback:
        progress_callback("合并相邻片段中...", 4, 5)

    raw_highlights_count = len(highlights)
    logger.info(f"合并前共 {raw_highlights_count} 个高分片段")
    highlights = _merge_consecutive_highlights(highlights)
    logger.info(f"合并后共 {len(highlights)} 个片段")

    highlights = _filter_similar_scenes(highlights)
    logger.info(f"去重后共 {len(highlights)} 个片段")

    highlights = _filter_by_type(highlights, type_filter or [])
    logger.info(f"类型过滤后共 {len(highlights)} 个片段")

    highlights = _filter_by_scores(highlights, score_filters or {})
    logger.info(f"分数过滤后共 {len(highlights)} 个片段")

    for i, hl in enumerate(highlights):
        hl["id"] = i + 1

    if progress_callback:
        progress_callback("分析完成", 5, 5)

    result = {
        "input_video": video_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_config": {
            "vlm_model": "qwen-vl-plus",
            "score_threshold": SCORE_THRESHOLD,
            "total_scenes": len(scenes),
            "highlights_before_process": raw_highlights_count,
            "whisper_transcribed": transcript is not None,
            "type_filter": type_filter,
            "score_filters": score_filters,
        },
        "highlights": highlights,
    }

    save_cached_result(md5_hash, result)
    logger.info(f"分析完成，共 {len(highlights)} 个名场面")
    return result


def _build_empty_result(video_name: str) -> dict:
    """构建无场景检测结果时的空结果字典"""
    return {
        "input_video": video_name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_config": {
            "vlm_model": "qwen-vl-plus",
            "score_threshold": SCORE_THRESHOLD,
            "total_scenes": 0,
        },
        "highlights": [],
    }
