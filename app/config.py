import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "6.0"))
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "30"))
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
KEYFRAME_DIR = BASE_DIR / os.getenv("KEYFRAME_DIR", "keyframes")
CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "cache")
CLIPS_DIR = BASE_DIR / os.getenv("CLIPS_DIR", "clips")
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))

SCORE_FILTERS = {
    "visual_impact": float(os.getenv("SCORE_FILTER_VISUAL_IMPACT", "0")),
    "cinematography": float(os.getenv("SCORE_FILTER_CINEMATOGRAPHY", "0")),
    "emotion_intensity": float(os.getenv("SCORE_FILTER_EMOTION_INTENSITY", "0")),
    "facial_expression": float(os.getenv("SCORE_FILTER_FACIAL_EXPRESSION", "0")),
    "plot_importance": float(os.getenv("SCORE_FILTER_PLOT_IMPORTANCE", "0")),
    "action_intensity": float(os.getenv("SCORE_FILTER_ACTION_INTENSITY", "0")),
    "audio_energy": float(os.getenv("SCORE_FILTER_AUDIO_ENERGY", "0")),
    "memorability": float(os.getenv("SCORE_FILTER_MEMORABILITY", "0")),
}

ALLOWED_TYPES = ["action", "drama", "emotion", "comedy", "suspense", "other"]
TYPE_FILTER = os.getenv("TYPE_FILTER", "").split(",") if os.getenv("TYPE_FILTER") else []

UPLOAD_DIR.mkdir(exist_ok=True)
KEYFRAME_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
