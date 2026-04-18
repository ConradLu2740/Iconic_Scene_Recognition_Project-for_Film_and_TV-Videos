from pathlib import Path
from app.config import ALLOWED_VIDEO_EXTENSIONS


def validate_video_file(filename: str) -> bool:
    """检查视频文件扩展名是否在允许列表中"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS


def format_seconds(seconds: float) -> str:
    """将秒数格式化为HH:MM:SS时间字符串"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
