import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def find_ffmpeg_path() -> str:
    """自动检测系统中FFmpeg的路径

    优先检测常见安装位置，返回ffmpeg可执行文件路径
    """
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    common_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe",
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg/bin/ffmpeg.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "ffmpeg/bin/ffmpeg.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
    ]

    for path in common_paths:
        if path.exists():
            logger.info(f"找到FFmpeg: {path}")
            return str(path)

    logger.warning("未找到FFmpeg，请确保已安装并添加到PATH")
    return "ffmpeg"


def find_ffprobe_path() -> str:
    """自动检测系统中ffprobe的路径"""
    if shutil.which("ffprobe"):
        return "ffprobe"

    common_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffprobe.exe",
        Path(os.environ.get("ProgramFiles", "")) / "ffmpeg/bin/ffprobe.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "ffmpeg/bin/ffprobe.exe",
        Path("C:/ffmpeg/bin/ffprobe.exe"),
        Path("C:/Program Files/ffmpeg/bin/ffprobe.exe"),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return "ffprobe"


_ffmpeg_path = None
_ffprobe_path = None


def get_ffmpeg_cmd() -> tuple[str, str]:
    """获取FFmpeg和FFprobe的命令路径"""
    global _ffmpeg_path, _ffprobe_path
    if _ffmpeg_path is None:
        _ffmpeg_path = find_ffmpeg_path()
        _ffprobe_path = find_ffprobe_path()
    return _ffmpeg_path, _ffprobe_path


def extract_video_clip(
    video_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    output_path: str | Path,
) -> str | None:
    """使用FFmpeg从视频中提取指定时间范围的片段

    Args:
        video_path: 输入视频路径
        start_seconds: 开始时间（秒）
        end_seconds: 结束时间（秒）
        output_path: 输出视频路径

    Returns:
        成功返回输出路径，失败返回None
    """
    try:
        video_path = str(video_path)
        output_path = str(output_path)
        duration = end_seconds - start_seconds

        ffmpeg_cmd, _ = get_ffmpeg_cmd()

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-ss", str(start_seconds),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            output_path,
        ]

        logger.info(f"提取视频片段: {start_seconds}s - {end_seconds}s")
        logger.debug(f"FFmpeg命令: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(60, int(duration * 2)),
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg提取失败: {result.stderr}")
            return None

        logger.info(f"视频片段提取成功: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error("视频片段提取超时")
        return None
    except Exception as e:
        logger.error(f"视频片段提取异常: {e}")
        return None


def get_video_duration(video_path: str | Path) -> float | None:
    """获取视频时长（秒）"""
    try:
        _, ffprobe_cmd = get_ffmpeg_cmd()
        cmd = [
            ffprobe_cmd,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return float(result.stdout.strip())
        return None

    except Exception as e:
        logger.error(f"获取视频时长失败: {e}")
        return None


def extract_clips_from_highlights(
    video_path: str | Path,
    highlights: list[dict],
    output_dir: str | Path,
    prefix: str = "clip",
) -> list[dict]:
    """从名场面列表提取多个视频片段

    Args:
        video_path: 输入视频路径
        highlights: 名场面列表
        output_dir: 输出目录
        prefix: 输出文件名前缀

    Returns:
        提取结果列表，每项包含 {id, start, end, output_path, success}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for hl in highlights:
        clip_id = hl.get("id", 0)
        start = hl.get("start_seconds", 0)
        end = hl.get("end_seconds", 0)

        output_path = output_dir / f"{prefix}_{clip_id:03d}_{start:.0f}s_{end:.0f}s.mp4"

        success = extract_video_clip(video_path, start, end, output_path) is not None

        results.append({
            "id": clip_id,
            "start_time": hl.get("start_time", ""),
            "end_time": hl.get("end_time", ""),
            "start_seconds": start,
            "end_seconds": end,
            "output_path": str(output_path) if success else None,
            "success": success,
        })

    return results
