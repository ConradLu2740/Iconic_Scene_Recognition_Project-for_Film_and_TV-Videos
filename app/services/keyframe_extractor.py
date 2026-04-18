import os
from pathlib import Path

import cv2

from app.config import KEYFRAME_DIR
from app.services.scene_detector import SceneSegment


def extract_keyframes(video_path: str | Path, scenes: list[SceneSegment], video_md5: str) -> list[str]:
    """从每个场景片段中提取中间帧作为关键帧，保存为JPEG图片

    Args:
        video_path: 视频文件路径
        scenes: 场景片段列表
        video_md5: 视频MD5哈希，用于组织关键帧存储目录

    Returns:
        关键帧图片路径列表，与scenes列表一一对应
    """
    output_dir = KEYFRAME_DIR / video_md5
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframe_paths = []

    for idx, scene in enumerate(scenes):
        mid_frame = (scene.start_frame + scene.end_frame) // 2
        frame_position = mid_frame / fps

        cap.set(cv2.CAP_PROP_POS_MSEC, frame_position * 1000)
        ret, frame = cap.read()

        if ret:
            filename = f"scene_{idx + 1:04d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            keyframe_paths.append(str(filepath))
        else:
            keyframe_paths.append("")

    cap.release()
    return keyframe_paths
