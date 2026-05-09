import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config_v2 import config

logger = logging.getLogger(__name__)


class SmartCache:
    """智能缓存系统

    支持版本控制、过期清理、增量更新和防重复分析
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._in_progress: dict[str, dict] = {}

    def _get_cache_key(self, video_path: str | Path, analysis_params: dict | None = None) -> str:
        """生成缓存键

        包含版本号、视频MD5和参数哈希
        """
        video_path = Path(video_path)
        video_md5 = self._compute_file_md5(video_path)

        params_hash = ""
        if analysis_params:
            params_str = json.dumps(analysis_params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        version = config.cache.cache_version
        return f"{version}_{video_md5}_{params_hash}"

    def _compute_file_md5(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _get_cache_metadata_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.meta.json"

    def get(self, video_path: str | Path, analysis_params: dict | None = None) -> dict | None:
        """获取缓存的分析结果

        Args:
            video_path: 视频文件路径
            analysis_params: 分析参数

        Returns:
            缓存的分析结果，不存在或过期返回None
        """
        cache_key = self._get_cache_key(video_path, analysis_params)
        cache_file = self._get_cache_path(cache_key)
        meta_file = self._get_cache_metadata_path(cache_key)

        if not cache_file.exists():
            return None

        if config.cache.enable_versioning and meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if self._is_expired(meta):
                    logger.info(f"缓存已过期: {cache_key}")
                    self._remove(cache_key)
                    return None
            except Exception as e:
                logger.warning(f"读取缓存元数据失败: {e}")

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            logger.info(f"命中缓存: {cache_key}")
            return result
        except Exception as e:
            logger.error(f"读取缓存失败: {e}")
            return None

    def _is_expired(self, meta: dict) -> bool:
        """检查缓存是否过期"""
        if not config.cache.max_age_days:
            return False

        created_at = meta.get("created_at")
        if not created_at:
            return True

        try:
            created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_time).days
            return age_days > config.cache.max_age_days
        except Exception:
            return True

    def save(self, video_path: str | Path, result: dict, analysis_params: dict | None = None) -> str:
        """保存分析结果到缓存

        Args:
            video_path: 视频文件路径
            result: 分析结果
            analysis_params: 分析参数

        Returns:
            缓存键
        """
        cache_key = self._get_cache_key(video_path, analysis_params)
        cache_file = self._get_cache_path(cache_key)
        meta_file = self._get_cache_metadata_path(cache_key)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if config.cache.enable_versioning:
                meta = {
                    "cache_key": cache_key,
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "video_path": str(video_path),
                    "analysis_params": analysis_params,
                    "version": config.cache.cache_version,
                }
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"缓存已保存: {cache_key}")
            return cache_key

        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return cache_key

    def _remove(self, cache_key: str):
        """删除缓存文件"""
        cache_file = self._get_cache_path(cache_key)
        meta_file = self._get_cache_metadata_path(cache_key)

        if cache_file.exists():
            cache_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

    def clear_expired(self) -> int:
        """清理过期缓存

        Returns:
            删除的缓存数量
        """
        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.stem.endswith(".meta"):
                continue

            meta_file = self._get_cache_metadata_path(cache_file.stem)
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if self._is_expired(meta):
                        self._remove(cache_file.stem)
                        removed += 1
                except Exception:
                    pass

        if removed > 0:
            logger.info(f"清理了 {removed} 个过期缓存")
        return removed

    def clear_all(self) -> int:
        """清空所有缓存

        Returns:
            删除的缓存数量
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        for meta_file in self.cache_dir.glob("*.meta.json"):
            meta_file.unlink()
            count += 1

        logger.info(f"清空了 {count} 个缓存文件")
        return count

    def start_analysis(self, video_path: str | Path) -> str:
        """标记分析开始，用于防重复

        Args:
            video_path: 视频文件路径

        Returns:
            分析ID
        """
        cache_key = self._get_cache_key(video_path)
        analysis_id = f"{cache_key}_{int(time.time() * 1000)}"

        with self._lock:
            self._in_progress[cache_key] = {
                "analysis_id": analysis_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "video_path": str(video_path),
            }

        logger.info(f"分析已开始: {analysis_id}")
        return analysis_id

    def is_analyzing(self, video_path: str | Path) -> bool:
        """检查视频是否正在分析中

        Args:
            video_path: 视频文件路径

        Returns:
            是否正在分析
        """
        cache_key = self._get_cache_key(video_path)
        with self._lock:
            return cache_key in self._in_progress

    def finish_analysis(self, video_path: str | Path):
        """标记分析完成

        Args:
            video_path: 视频文件路径
        """
        cache_key = self._get_cache_key(video_path)
        with self._lock:
            if cache_key in self._in_progress:
                del self._in_progress[cache_key]
        logger.info(f"分析已结束: {cache_key}")

    def get_in_progress_info(self, video_path: str | Path) -> dict | None:
        """获取正在进行的分析信息"""
        cache_key = self._get_cache_key(video_path)
        with self._lock:
            return self._in_progress.get(cache_key)

    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        cache_files = list(self.cache_dir.glob("*.json"))
        meta_files = list(self.cache_dir.glob("*.meta.json"))

        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

        return {
            "total_caches": len(cache_files),
            "total_meta": len(meta_files),
            "in_progress": len(self._in_progress),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }


class IncrementalCache:
    """增量缓存系统

    支持只缓存中间结果，后续可以增量更新
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (config.CACHE_DIR / "incremental")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._smart_cache = SmartCache(self.cache_dir)

    def save_scene_detection(
        self,
        video_path: str | Path,
        scenes: list[dict],
    ):
        """缓存场景检测结果"""
        cache_key = f"scenes_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "type": "scene_detection",
                "scenes": scenes,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, f, ensure_ascii=False, indent=2)

    def save_keyframes(
        self,
        video_path: str | Path,
        keyframe_paths: dict[int, list[str]],
    ):
        """缓存关键帧提取结果"""
        cache_key = f"keyframes_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "type": "keyframes",
                "keyframes": keyframe_paths,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, f, ensure_ascii=False, indent=2)

    def save_transcript(
        self,
        video_path: str | Path,
        transcript: dict,
    ):
        """缓存语音转写结果"""
        cache_key = f"transcript_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "type": "transcript",
                "transcript": transcript,
                "cached_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, f, ensure_ascii=False, indent=2)

    def get_cached_scenes(self, video_path: str | Path) -> list[dict] | None:
        """获取缓存的场景检测结果"""
        cache_key = f"scenes_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("scenes")
        return None

    def get_cached_keyframes(self, video_path: str | Path) -> dict | None:
        """获取缓存的关键帧结果"""
        cache_key = f"keyframes_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("keyframes")
        return None

    def get_cached_transcript(self, video_path: str | Path) -> dict | None:
        """获取缓存的语音转写结果"""
        cache_key = f"transcript_{self._smart_cache._get_cache_key(video_path)}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("transcript")
        return None


smart_cache = SmartCache()
incremental_cache = IncrementalCache()
