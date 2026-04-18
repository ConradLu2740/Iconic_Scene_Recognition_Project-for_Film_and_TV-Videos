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
from app.services.video_clipper import extract_clips_from_highlights
from app.utils.video_utils import validate_video_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_current_video_path = None
_current_result = None


def process_video(
    video_file,
    threshold,
    type_filter,
    whisper_model,
    facial_min,
    emotion_min,
    action_min,
    memorability_min,
):
    """处理上传的视频文件，执行名场面分析并返回结果

    Args:
        video_file: Gradio上传的视频文件路径
        threshold: 名场面评分阈值
        type_filter: 场景类型过滤列表
        whisper_model: Whisper模型大小
        facial_min: 面部表情最低分数
        emotion_min: 情感强度最低分数
        action_min: 动作强度最低分数
        memorability_min: 记忆点最低分数

    Returns:
        tuple: (结果JSON文本, 关键帧图片列表, 状态消息)
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

    try:
        result = analyze_video(
            str(dest_path),
            type_filter=type_filter if type_filter else None,
            score_filters=score_filters if any(v > 0 for v in score_filters.values()) else None,
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
    """构建Gradio界面，包含视频上传、参数配置、结果展示和视频剪辑导出功能"""
    with gr.Blocks(title="视频名场面检测器") as demo:
        gr.Markdown(
            "# 🎬 视频名场面检测器\n"
            "上传视频 → 自动检测精彩片段 → 输出名场面JSON数据 + 视频片段导出\n"
            "基于 PySceneDetect + 通义千问VL + Faster-Whisper 多模态分析"
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📹 上传视频",
                    sources=["upload"],
                )
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
            "1. 上传视频文件（支持 mp4/avi/mov/mkv 等格式）\n"
            "2. 设置总分阈值（默认6.0，值越高筛选越严格）\n"
            "3. 可选：设置类型过滤和维度分数过滤\n"
            "4. 选择Whisper模型大小（越大越准确但越慢）\n"
            "5. 点击「开始分析」，等待2-3分钟\n"
            "6. 查看JSON结果和关键帧缩略图\n"
            "7. 可选：点击「导出视频片段」下载名场面\n\n"
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
