import json
import logging
import shutil
from pathlib import Path

import gradio as gr

from app.config import (
    UPLOAD_DIR,
    MAX_VIDEO_SIZE_MB,
    SCORE_THRESHOLD,
    KEYFRAME_DIR,
    CLIPS_DIR,
    ALLOWED_TYPES,
)
from app.services.highlight_pipeline import analyze_video
from app.services.intent_recognizer import recognize_intent, IntentType
from app.services.video_clipper import extract_clips_from_highlights
from app.utils.video_utils import validate_video_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_current_video_path = None
_current_result = None


def _apply_intent_to_ui(intent_result):
    """将意图识别结果转换为 Gradio UI 组件的更新操作

    根据识别出的意图类型、场景类型、维度权重等，
    自动填充对应的阈值、类型选择和维度滑块。

    Args:
        intent_result: RecognizedIntent 意图识别结果

    Returns:
        tuple: (threshold, type_filter, facial, emotion, action, memorability, intent_info)
    """
    threshold = SCORE_THRESHOLD
    type_filter = intent_result.scene_types if intent_result.scene_types else []
    facial_min = 0.0
    emotion_min = 0.0
    action_min = 0.0
    memorability_min = 0.0

    if intent_result.score_threshold is not None:
        threshold = intent_result.score_threshold

    boosts = intent_result.dimension_boosts
    if boosts.get("facial_expression", 1.0) > 1.0:
        facial_min = 5.0
    if boosts.get("emotion_intensity", 1.0) > 1.0:
        emotion_min = 5.0
    if boosts.get("action_intensity", 1.0) > 1.0:
        action_min = 5.0
    if boosts.get("memorability", 1.0) > 1.0:
        memorability_min = 5.0

    intent_info = (
        f"🧠 意图识别: 类型={intent_result.intent_type.value}, "
        f"场景={intent_result.scene_types or '全部'}, "
        f"置信度={intent_result.confidence:.0%}"
    )
    if intent_result.dimension_boosts:
        boost_desc = ", ".join(f"{k}×{v}" for k, v in intent_result.dimension_boosts.items())
        intent_info += f"\n📊 维度增强: {boost_desc}"
    if intent_result.count_limit:
        intent_info += f"\n🔢 数量限制: 前{intent_result.count_limit}个"

    return threshold, type_filter, facial_min, emotion_min, action_min, memorability_min, intent_info


def parse_intent_query(query):
    """解析用户自然语言查询，返回意图识别结果和UI参数填充值

    Args:
        query: 用户输入的自然语言查询

    Returns:
        tuple: Gradio组件更新值列表
    """
    if not query or not query.strip():
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), "💡 请输入你的需求，如「找出最搞笑的5个片段」"
        )

    intent = recognize_intent(query)
    threshold, type_filter, facial, emotion, action, memorability, info = _apply_intent_to_ui(intent)

    return (
        gr.update(value=threshold),
        gr.update(value=type_filter),
        gr.update(value=facial),
        gr.update(value=emotion),
        gr.update(value=action),
        gr.update(value=memorability),
        info,
    )


def process_video(
    video_file,
    threshold,
    type_filter,
    whisper_model,
    facial_min,
    emotion_min,
    action_min,
    memorability_min,
    intent_query,
):
    """处理上传的视频文件，执行名场面分析并返回结果

    支持两种模式：
    1. 手动配置模式：通过滑块和复选框设置参数
    2. 意图驱动模式：通过自然语言输入，自动映射参数并应用维度权重增强

    Args:
        video_file: Gradio上传的视频文件路径
        threshold: 名场面评分阈值
        type_filter: 场景类型过滤列表
        whisper_model: Whisper模型大小
        facial_min: 面部表情最低分数
        emotion_min: 情感强度最低分数
        action_min: 动作强度最低分数
        memorability_min: 记忆点最低分数
        intent_query: 用户自然语言意图查询

    Returns:
        tuple: (结果JSON文本, 关键帧图片列表, 状态消息, 导出按钮状态)
    """
    global _current_video_path, _current_result

    if video_file is None:
        return "请上传视频文件", [], "❌ 未上传文件", None

    video_path = Path(video_file)
    if not validate_video_file(video_path.name):
        return "不支持的视频格式", [], f"❌ 仅支持 {', '.join(['mp4','avi','mov','mkv','flv','wmv','webm'])}", None

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        return "文件过大", [], f"❌ 文件大小 {file_size_mb:.1f}MB 超过限制 {MAX_VIDEO_SIZE_MB}MB", None

    dest_path = UPLOAD_DIR / video_path.name
    if video_path != dest_path:
        shutil.copy2(video_path, dest_path)

    import app.config as config
    config.SCORE_THRESHOLD = float(threshold)
    config.WHISPER_MODEL = whisper_model

    score_filters = {
        "facial_expression": float(facial_min),
        "emotion_intensity": float(emotion_min),
        "action_intensity": float(action_min),
        "memorability": float(memorability_min),
    }

    dimension_boosts = None
    count_limit = None
    effective_threshold = float(threshold)

    if intent_query and intent_query.strip():
        intent = recognize_intent(intent_query.strip())
        if intent.intent_type != IntentType.UNKNOWN:
            dimension_boosts = intent.dimension_boosts if intent.dimension_boosts else None
            count_limit = intent.count_limit
            if intent.score_threshold is not None:
                effective_threshold = intent.score_threshold

    try:
        result = analyze_video(
            str(dest_path),
            type_filter=type_filter if type_filter else None,
            score_filters=score_filters if any(v > 0 for v in score_filters.values()) else None,
            dimension_boosts=dimension_boosts,
            count_limit=count_limit,
            score_threshold=effective_threshold,
        )
        _current_video_path = str(dest_path)
        _current_result = result
    except Exception as e:
        logger.exception("分析过程异常")
        return f"分析失败: {str(e)}", [], f"❌ 分析失败: {str(e)}", None

    json_output = json.dumps(result, ensure_ascii=False, indent=2)

    keyframe_images = []
    for hl in result.get("highlights", []):
        kf_url = hl.get("keyframe_url", "")
        if kf_url:
            kf_full_path = Path(str(KEYFRAME_DIR.parent) + kf_url)
            if kf_full_path.exists():
                label = f"场景{hl['id']} | {hl['type']} | 评分{hl['scores']['total']}"
                if hl.get("transcript"):
                    label += f"\n\"{hl['transcript'][:30]}...\""
                keyframe_images.append((str(kf_full_path), label))

    highlight_count = len(result.get("highlights", []))
    total_scenes = result.get("analysis_config", {}).get("total_scenes", 0)
    whisper_ok = result.get("analysis_config", {}).get("whisper_transcribed", False)
    status = f"✅ 分析完成: 共{total_scenes}个场景, 检测到{highlight_count}个名场面"
    if whisper_ok:
        status += " (含语音字幕)"
    if dimension_boosts:
        boost_desc = ", ".join(f"{k}×{v}" for k, v in dimension_boosts.items())
        status += f"\n🧠 意图增强: {boost_desc}"

    clip_btn = gr.update(visible=True) if highlight_count > 0 else gr.update(visible=False)

    return json_output, keyframe_images, status, clip_btn


def export_clips():
    """导出名场面视频片段"""
    global _current_video_path, _current_result

    if not _current_video_path or not _current_result:
        return "❌ 请先分析视频", None

    try:
        highlights = _current_result.get("highlights", [])
        if not highlights:
            return "❌ 没有名场面可导出", None

        video_name = Path(_current_video_path).stem
        output_dir = CLIPS_DIR / video_name

        clips = extract_clips_from_highlights(
            _current_video_path,
            highlights,
            output_dir,
            prefix=video_name,
        )

        success_count = sum(1 for c in clips if c["success"])
        if success_count == 0:
            return "❌ 视频片段导出失败（请确保已安装FFmpeg）", None

        clip_files = [c["output_path"] for c in clips if c["success"]]
        return f"✅ 成功导出 {success_count}/{len(clips)} 个片段", clip_files

    except Exception as e:
        logger.exception("导出片段异常")
        return f"❌ 导出失败: {str(e)}", None


def build_ui() -> gr.Blocks:
    """构建Gradio界面，包含意图识别、视频上传、参数配置、结果展示和视频剪辑导出功能"""
    with gr.Blocks(title="视频名场面检测器") as demo:
        gr.Markdown(
            "# 🎬 视频名场面检测器\n"
            "上传视频 → 自然语言描述需求 → 自动检测精彩片段 → 输出名场面数据 + 视频片段导出\n"
            "基于 PySceneDetect + 通义千问VL + Faster-Whisper + 意图识别 多模态智能分析"
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📹 上传视频",
                    sources=["upload"],
                )

                gr.Markdown("### 🧠 智能意图输入")
                intent_input = gr.Textbox(
                    label="💬 用自然语言描述你的需求",
                    placeholder="例如：找出最搞笑的5个片段 / 只要8分以上的动作戏 / 感人的场景",
                    lines=2,
                    info="输入后点击「理解意图」，系统自动填充下方参数",
                )
                intent_parse_btn = gr.Button("🔍 理解意图", variant="secondary")
                intent_info = gr.Textbox(
                    label="意图识别结果",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("---")
                threshold_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=SCORE_THRESHOLD,
                    step=0.5,
                    label="🎯 名场面总分阈值",
                    info="分数高于此阈值的场景将被标记为名场面",
                )

                gr.Markdown("### 🔍 精细过滤")
                type_checkbox = gr.CheckboxGroup(
                    choices=ALLOWED_TYPES,
                    label="场景类型",
                    info="选择要保留的场景类型（不选则包含所有类型）",
                )
                whisper_dropdown = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="🎤 Whisper模型",
                    info="越大越准确，但处理速度更慢",
                )

                gr.Markdown("### 📊 维度分数过滤（设置0则不限制）")
                with gr.Row():
                    facial_slider = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="😱 面部表情", info="最低分数要求")
                    emotion_slider = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="❤️ 情感强度", info="最低分数要求")
                with gr.Row():
                    action_slider = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="💥 动作强度", info="最低分数要求")
                    memorability_slider = gr.Slider(minimum=0, maximum=10, value=0, step=0.5, label="🧠 记忆点", info="最低分数要求")

                analyze_btn = gr.Button(
                    "🚀 开始分析",
                    variant="primary",
                    size="lg",
                )
                status_text = gr.Textbox(
                    label="状态",
                    interactive=False,
                )
                export_btn = gr.Button(
                    "✂️ 导出视频片段",
                    variant="secondary",
                    size="lg",
                    visible=False,
                )
                clip_output = gr.File(
                    label="📁 导出的视频片段",
                    file_count="multiple",
                    visible=False,
                )

            with gr.Column(scale=1):
                json_output = gr.Code(
                    label="📋 分析结果 (JSON)",
                    language="json",
                    interactive=False,
                )
                gallery = gr.Gallery(
                    label="🖼️ 名场面关键帧",
                    columns=2,
                    height="auto",
                    object_fit="contain",
                )

        intent_parse_btn.click(
            fn=parse_intent_query,
            inputs=[intent_input],
            outputs=[
                threshold_slider, type_checkbox,
                facial_slider, emotion_slider,
                action_slider, memorability_slider,
                intent_info,
            ],
        )

        analyze_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                threshold_slider,
                type_checkbox,
                whisper_dropdown,
                facial_slider,
                emotion_slider,
                action_slider,
                memorability_slider,
                intent_input,
            ],
            outputs=[json_output, gallery, status_text, export_btn],
        )

        export_btn.click(
            fn=export_clips,
            inputs=[],
            outputs=[status_text, clip_output],
        )

        gr.Markdown(
            "---\n"
            "### 使用说明\n"
            "**🧠 智能模式（推荐）**：在「智能意图输入」框中用自然语言描述需求，点击「理解意图」自动配置参数\n\n"
            "**手动模式**：直接调整下方滑块和选项\n\n"
            "1. 上传视频文件（支持 mp4/avi/mov/mkv 等格式）\n"
            "2. 输入需求或手动设置参数\n"
            "3. 点击「开始分析」，等待2-3分钟\n"
            "4. 查看JSON结果和关键帧缩略图\n"
            "5. 可选：点击「导出视频片段」下载名场面\n\n"
            "⚠️ 需要安装FFmpeg才能导出视频片段"
        )

    return demo


def main():
    """启动Gradio应用服务器"""
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
