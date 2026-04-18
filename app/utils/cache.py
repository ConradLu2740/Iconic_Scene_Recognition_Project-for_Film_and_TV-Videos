import hashlib
import json
from pathlib import Path
from app.config import CACHE_DIR


def compute_file_md5(file_path: str | Path) -> str:
    """计算文件的MD5哈希值，用于视频去重缓存"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def get_cached_result(md5_hash: str) -> dict | None:
    """根据MD5哈希查找缓存的分析结果，命中则返回JSON数据，否则返回None"""
    cache_file = CACHE_DIR / f"{md5_hash}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_result(md5_hash: str, result: dict) -> None:
    """将分析结果保存为JSON缓存文件，键为视频MD5哈希"""
    cache_file = CACHE_DIR / f"{md5_hash}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
