import base64
import json
import logging
from dataclasses import dataclass

import dashscope
from dashscope import MultiModalConversation

from app.config import DASHSCOPE_API_KEY, VLM_MODEL, VLM_TIMEOUT

logger = logging.getLogger(__name__)

SCORING_PROMPT = """你是一个专业的视频内容分析专家。请仔细观察这个视频关键帧，进行深度内容分析。

## 评分维度（0-10分，精确到小数点后一位）

**1. 视觉冲击力 (visual_impact)**
- 画面震撼程度、色彩/光影突出性
- 强烈的视觉元素（爆炸、特殊效果、大场面）
- 评分标准：0几乎没有→5中等→10极其震撼

**2. 镜头语言 (cinematography)**
- 运镜技巧（推拉摇移跟等）
- 角度创新（航拍、俯拍、仰拍、鱼眼等）
- 特殊技巧（慢动作、倒放等）
- 评分标准：0普通→5有一定技巧→10极具创意

**3. 情感强度 (emotion_intensity)**
- 情感传达力度（紧张、兴奋、感动、愤怒、恐惧）
- 场景氛围营造
- 人物表情和肢体语言情感
- 评分标准：0平静→5中等感染→10极度强烈

**4. 面部表情夸张度 (facial_expression)**
- 表情夸张程度（惊讶张大嘴、愤怒扭曲、极度恐惧、狂喜）
- 表情占据画面比例和醒目程度
- 情绪清晰可辨认
- 评分标准：0几乎没有表情→5中等夸张→10极其夸张明显

**5. 剧情重要性 (plot_importance)**
- 关键情节转折点
- 重大信息揭露或决定性时刻
- 角色命运转折、故事高潮
- 评分标准：0无关紧要→5一般重要→10极其关键

**6. 动作强度 (action_intensity)**
- 动作密度和精彩程度
- 打斗编排、追逐戏、爆炸场面
- 速度感和节奏感
- 评分标准：0几乎没有动作→5中等→10极其激烈

**7. 音频能量 (audio_energy)**
- 音量峰值和音效震撼度
- 爆炸声、撞击声、欢呼声等
- 配乐强度和节奏感
- 评分标准：0安静→5中等→10极其震撼

**8. 记忆点 (memorability)**
- 画面/台词的独特性和可模仿性
- 标志性程度（成为经典名场面的潜力）
- 令人印象深刻的程度
- 评分标准：0平淡无奇→5有一定记忆点→10极具标志性

## 场景类型分类

- **action**: 动作场面（打斗、追逐、爆炸、战争、格斗）
- **drama**: 戏剧场景（情感对话、人物冲突、内心挣扎）
- **emotion**: 情感高潮（感人时刻、泪点、温馨、浪漫）
- **comedy**: 喜剧场景（搞笑、幽默、滑稽、夸张表情）
- **suspense**: 悬疑紧张（惊悚、恐怖、悬念、紧张刺激）
- **other**: 其他类型

## 重要提示

1. 每个维度独立评分，不要互相影响
2. 总分由各维度加权平均得出（视觉3、镜头2、情感3、表情3、剧情2、动作2、音频2、记忆2，共19权重）
3. facial_expression高分场景即使其他维度一般也应给较高总分
4. 描述字段不超过20字，聚焦画面核心内容

请严格按以下JSON格式输出：
{
  "visual_impact": <0-10>,
  "cinematography": <0-10>,
  "emotion_intensity": <0-10>,
  "facial_expression": <0-10>,
  "plot_importance": <0-10>,
  "action_intensity": <0-10>,
  "audio_energy": <0-10>,
  "memorability": <0-10>,
  "type": "<action|drama|emotion|comedy|suspense|other>",
  "description": "<20字以内的中文描述>"
}"""


@dataclass
class VLMResult:
    """VLM分析结果，包含8维度评分和场景描述"""
    visual_impact: float
    cinematography: float
    emotion_intensity: float
    facial_expression: float
    plot_importance: float
    action_intensity: float
    audio_energy: float
    memorability: float
    type: str
    description: str

    @property
    def total_score(self) -> float:
        """计算加权总分"""
        weights = {
            "visual_impact": 3,
            "cinematography": 2,
            "emotion_intensity": 3,
            "facial_expression": 3,
            "plot_importance": 2,
            "action_intensity": 2,
            "audio_energy": 2,
            "memorability": 2,
        }
        total_weight = sum(weights.values())
        weighted_sum = (
            self.visual_impact * weights["visual_impact"] +
            self.cinematography * weights["cinematography"] +
            self.emotion_intensity * weights["emotion_intensity"] +
            self.facial_expression * weights["facial_expression"] +
            self.plot_importance * weights["plot_importance"] +
            self.action_intensity * weights["action_intensity"] +
            self.audio_energy * weights["audio_energy"] +
            self.memorability * weights["memorability"]
        )
        return round(weighted_sum / total_weight, 2)


def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为base64字符串，用于API传输"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_keyframe(image_path: str) -> VLMResult | None:
    """调用通义千问VL模型对单个关键帧进行8维度评分

    Args:
        image_path: 关键帧图片路径

    Returns:
        VLM分析结果，失败时返回None
    """
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_api_key_here":
        logger.error("DASHSCOPE_API_KEY未配置，请在.env文件中设置")
        return None

    dashscope.api_key = DASHSCOPE_API_KEY

    try:
        image_base64 = encode_image_to_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/jpeg;base64,{image_base64}"},
                    {"text": SCORING_PROMPT},
                ],
            }
        ]

        response = MultiModalConversation.call(
            model=VLM_MODEL,
            messages=messages,
            timeout=VLM_TIMEOUT,
            result_format="message",
        )

        if response.status_code != 200:
            logger.error(f"VLM API调用失败: {response.code} - {response.message}")
            return None

        text_content = response.output.choices[0].message.content[0]["text"]

        json_str = text_content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        result_dict = json.loads(json_str)

        return VLMResult(
            visual_impact=float(result_dict.get("visual_impact", 0)),
            cinematography=float(result_dict.get("cinematography", 0)),
            emotion_intensity=float(result_dict.get("emotion_intensity", 0)),
            facial_expression=float(result_dict.get("facial_expression", 0)),
            plot_importance=float(result_dict.get("plot_importance", 0)),
            action_intensity=float(result_dict.get("action_intensity", 0)),
            audio_energy=float(result_dict.get("audio_energy", 0)),
            memorability=float(result_dict.get("memorability", 0)),
            type=result_dict.get("type", "other"),
            description=result_dict.get("description", ""),
        )

    except json.JSONDecodeError as e:
        logger.error(f"VLM返回结果JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"VLM分析异常: {e}")
        return None
