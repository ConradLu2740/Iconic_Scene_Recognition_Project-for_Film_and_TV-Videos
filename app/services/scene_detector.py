from dataclasses import dataclass
from pathlib import Path

from scenedetect import open_video, SceneManager, ContentDetector

from app.config import KEYFRAME_DIR


@dataclass
class SceneSegment:
    """表示一个检测到的场景片段，包含起止时间码和帧号"""
    start_timecode: str
    end_timecode: str
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int


def detect_scenes(video_path: str | Path, threshold: float = 27.0) -> list[SceneSegment]:
    """使用PySceneDetect的ContentDetector检测视频中的镜头切换点，返回场景片段列表

    Args:
        video_path: 视频文件路径
        threshold: 内容变化检测阈值，值越小越敏感，默认27.0

    Returns:
        检测到的场景片段列表
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    segments = []
    for scene in scene_list:
        start_tc, end_tc = scene
        segments.append(SceneSegment(
            start_timecode=str(start_tc),
            end_timecode=str(end_tc),
            start_seconds=start_tc.get_seconds(),
            end_seconds=end_tc.get_seconds(),
            start_frame=start_tc.get_frames(),
            end_frame=end_tc.get_frames(),
        ))

    return segments
