# 视频名场面检测器 (Video Highlight Detector)

## 项目简介

基于多模态AI的视频精彩片段自动检测与导出系统。用户上传视频后，系统自动分析并输出"名场面"时间戳、关键帧、评分及语音字幕，支持一键导出视频片段。

## 核心技术栈

| 层级 | 技术 |
|------|------|
| **前端** | Gradio (Python Web UI) |
| **后端** | FastAPI + Python |
| **场景检测** | PySceneDetect (ContentDetector) |
| **视觉理解** | 通义千问VL (qwen-vl-plus) |
| **语音转写** | Faster-Whisper |
| **视频处理** | FFmpeg + OpenCV |
| **缓存** | MD5去重 + JSON本地缓存 |

## 系统架构

```
视频上传 → 场景分割 → 关键帧提取 → VLM多维度评分 → 语音转写 → 片段合并去重 → 结果输出
     ↓
  MD5缓存检查（命中则跳过）
```

## 核心功能

- [x] PySceneDetect 镜头切换检测
- [x] OpenCV 关键帧提取
- [x] 通义千问VL 8维度智能评分
- [x] Faster-Whisper 语音转写
- [x] 多维度分数过滤 + 类型过滤
- [x] FFmpeg 视频片段裁剪导出
- [x] MD5缓存去重

## 评分维度 (8维)

| 维度 | 权重 | 说明 |
|------|------|------|
| visual_impact | 3 | 视觉冲击力 |
| cinematography | 2 | 镜头语言 |
| emotion_intensity | 3 | 情感强度 |
| facial_expression | 3 | 面部表情夸张度 |
| plot_importance | 2 | 剧情重要性 |
| action_intensity | 2 | 动作强度 |
| audio_energy | 2 | 音频能量 |
| memorability | 2 | 记忆点 |

## 项目结构

```
video-analyzer/
├── app/
│   ├── main.py                    # Gradio Web UI 入口
│   ├── config.py                  # 配置管理 (dotenv)
│   ├── services/
│   │   ├── highlight_pipeline.py  # 分析流水线编排
│   │   ├── scene_detector.py     # PySceneDetect 封装
│   │   ├── keyframe_extractor.py # OpenCV 关键帧提取
│   │   ├── vlm_analyzer.py        # 通义千问VL 调用
│   │   ├── whisper_transcriber.py # Faster-Whisper 转写
│   │   └── video_clipper.py       # FFmpeg 片段导出
│   └── utils/
│       ├── cache.py               # MD5 缓存管理
│       └── video_utils.py         # 视频工具函数
├── requirements.txt
└── .env.example
```

## 技术亮点

1. **多模态融合分析**：结合视觉理解(VLM)和语音转写(Whisper)，多维度评估视频精彩程度
2. **智能后处理**：片段合并算法（时间连续性+评分相似度）+ 语义去重
3. **FFmpeg路径自动检测**：兼容多种安装方式的FFmpeg路径探测
4. **零外部依赖的异步**：使用Python生成器实现轻量级进度回调

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API Key
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 3. 启动应用
python -m app.main
# 访问 http://localhost:7860
```

## 输出示例

```json
{
  "input_video": "example.mp4",
  "generated_at": "2026-04-18T12:00:00Z",
  "analysis_config": {
    "vlm_model": "qwen-vl-plus",
    "score_threshold": 6.0,
    "total_scenes": 51,
    "whisper_transcribed": true
  },
  "highlights": [
    {
      "id": 1,
      "start_time": "00:01:23",
      "end_time": "00:02:45",
      "scores": {
        "visual_impact": 8.5,
        "emotion_intensity": 9.0,
        "facial_expression": 9.5,
        "memorability": 8.0,
        "total": 8.42
      },
      "type": "drama",
      "description": "主角表情极度震惊，情感张力十足",
      "transcript": "不...这不可能"
    }
  ]
}
```

## 适用场景

- 视频剪辑师快速筛选精彩片段
- 内容创作者高效定位高光时刻
- AI数据集构建（精彩片段标注）
