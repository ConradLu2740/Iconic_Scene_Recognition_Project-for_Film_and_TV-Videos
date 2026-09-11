from app.services.scene_detector import detect_scenes, SceneSegment
from app.services.keyframe_extractor import extract_keyframes
from app.services.vlm_analyzer import analyze_keyframe, VLMResult
from app.services.whisper_transcriber import transcribe_video, get_transcript_for_time_range, FullTranscript
from app.services.highlight_pipeline import analyze_video
from app.services.intent_recognizer import recognize_intent, IntentType, RecognizedIntent
from app.services.video_clipper import extract_clips_from_highlights, extract_video_clip

try:
    from app.services.audio_analyzer import (
        AudioEnergyResult,
        analyze_audio_segment,
        extract_audio,
        extract_audio_features_for_scenes,
    )
except ImportError:
    AudioEnergyResult = None
    analyze_audio_segment = None
    extract_audio = None
    extract_audio_features_for_scenes = None

try:
    from app.services.frame_quality import (
        FrameQualityResult,
        analyze_frame_quality,
        extract_best_keyframes,
        smart_sample_scene_frames,
        filter_frames_by_quality,
    )
except ImportError:
    FrameQualityResult = None
    analyze_frame_quality = None
    extract_best_keyframes = None
    smart_sample_scene_frames = None
    filter_frames_by_quality = None

try:
    from app.services.smart_cache import (
        SmartCache,
        IncrementalCache,
        smart_cache,
        incremental_cache,
    )
except ImportError:
    SmartCache = None
    IncrementalCache = None
    smart_cache = None
    incremental_cache = None

try:
    from app.services.progress_tracker import (
        StreamingProgressCallback,
        ProgressTracker,
        ProgressGenerator,
        PipelineStage,
        ProgressInfo,
    )
except ImportError:
    StreamingProgressCallback = None
    ProgressTracker = None
    ProgressGenerator = None
    PipelineStage = None
    ProgressInfo = None

try:
    from app.services.visualizer import (
        generate_timeline_visualization,
        generate_score_distribution_chart,
        generate_type_distribution_pie,
        generate_full_report_visualization,
    )
except ImportError:
    generate_timeline_visualization = None
    generate_score_distribution_chart = None
    generate_type_distribution_pie = None
    generate_full_report_visualization = None

try:
    from app.services.pipeline_plugins import (
        HighlightPipeline,
        AnalyzerPlugin,
        AnalysisContext,
        get_default_pipeline,
    )
except ImportError:
    HighlightPipeline = None
    AnalyzerPlugin = None
    AnalysisContext = None
    get_default_pipeline = None

try:
    from app.services.vlm_local import (
        VLMBackend,
        RuleBasedVLM,
        VLMFactory,
        get_vlm_backend,
    )
except ImportError:
    VLMBackend = None
    RuleBasedVLM = None
    VLMFactory = None
    get_vlm_backend = None

try:
    from app.services.vlm_analyzer_concurrent import (
        ConcurrentVLMAnalyzer,
        VLMAnalyzerPool,
        analyze_keyframes_concurrent,
    )
except ImportError:
    ConcurrentVLMAnalyzer = None
    VLMAnalyzerPool = None
    analyze_keyframes_concurrent = None

__all__ = [
    "detect_scenes",
    "SceneSegment",
    "extract_keyframes",
    "analyze_keyframe",
    "VLMResult",
    "transcribe_video",
    "get_transcript_for_time_range",
    "FullTranscript",
    "analyze_video",
    "recognize_intent",
    "IntentType",
    "RecognizedIntent",
    "extract_clips_from_highlights",
    "extract_video_clip",
    "AudioEnergyResult",
    "analyze_audio_segment",
    "extract_audio",
    "extract_audio_features_for_scenes",
    "FrameQualityResult",
    "analyze_frame_quality",
    "extract_best_keyframes",
    "smart_sample_scene_frames",
    "filter_frames_by_quality",
    "SmartCache",
    "IncrementalCache",
    "smart_cache",
    "incremental_cache",
    "StreamingProgressCallback",
    "ProgressTracker",
    "ProgressGenerator",
    "PipelineStage",
    "ProgressInfo",
    "generate_timeline_visualization",
    "generate_score_distribution_chart",
    "generate_type_distribution_pie",
    "generate_full_report_visualization",
    "HighlightPipeline",
    "AnalyzerPlugin",
    "AnalysisContext",
    "get_default_pipeline",
    "VLMBackend",
    "RuleBasedVLM",
    "VLMFactory",
    "get_vlm_backend",
    "ConcurrentVLMAnalyzer",
    "VLMAnalyzerPool",
    "analyze_keyframes_concurrent",
]
