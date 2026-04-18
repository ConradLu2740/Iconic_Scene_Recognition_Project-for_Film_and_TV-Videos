import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """用户意图类型枚举

    HIGHLIGHT_SEARCH: 搜索精彩片段，如"找出最搞笑的片段"
    FILTER_REQUEST: 筛选请求，如"只要8分以上的动作戏"
    SUMMARY_REQUEST: 视频概要，如"总结一下这个视频"
    EXPORT_REQUEST: 导出请求，如"把这些片段导出"
    COMPARISON_QUERY: 对比查询，如"哪个片段更感人"
    UNKNOWN: 无法识别的意图
    """

    HIGHLIGHT_SEARCH = "highlight_search"
    FILTER_REQUEST = "filter_request"
    SUMMARY_REQUEST = "summary_request"
    EXPORT_REQUEST = "export_request"
    COMPARISON_QUERY = "comparison_query"
    UNKNOWN = "unknown"


SCENE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "action": [
        "动作", "打斗", "追逐", "爆炸", "战争", "格斗", "武打",
        "fight", "chase", "explosion", "battle", "action",
    ],
    "drama": [
        "戏剧", "对话", "冲突", "争吵", "对峙", "谈判",
        "drama", "dialogue", "conflict", "argument",
    ],
    "emotion": [
        "感人", "感动", "泪点", "温馨", "浪漫", "爱情", "催泪",
        "touching", "emotional", "romantic", "heartfelt", "tear",
    ],
    "comedy": [
        "搞笑", "幽默", "滑稽", "喜剧", "笑点", "逗", "好玩",
        "funny", "comedy", "humor", "hilarious", "laugh",
    ],
    "suspense": [
        "悬疑", "紧张", "惊悚", "恐怖", "悬念", "刺激", "吓人",
        "suspense", "thriller", "horror", "tense", "scary",
    ],
}

DIMENSION_KEYWORDS: dict[str, list[str]] = {
    "visual_impact": ["视觉", "画面", "震撼", "特效", "大场面", "visual", "spectacular"],
    "cinematography": ["镜头", "运镜", "拍摄", "航拍", "慢动作", "cinematography", "camera"],
    "emotion_intensity": ["情感", "情绪", "感染力", "氛围", "emotion", "atmosphere"],
    "facial_expression": ["表情", "面部", "夸张", "脸", "facial", "expression"],
    "plot_importance": ["剧情", "情节", "转折", "关键", "plot", "turning point"],
    "action_intensity": ["动作", "打斗", "激烈", "节奏", "action", "intensity"],
    "audio_energy": ["音效", "配乐", "声音", "音量", "audio", "sound"],
    "memorability": ["经典", "记忆", "印象", "标志", "memorable", "iconic"],
}

INTENT_KEYWORDS: dict[IntentType, list[tuple[str, float]]] = {
    IntentType.HIGHLIGHT_SEARCH: [
        ("找出", 1.0), ("找到", 1.0), ("搜索", 1.0), ("寻找", 1.0), ("看看", 0.8), ("有没有", 0.8),
        ("精彩", 1.0), ("名场面", 1.2), ("高光", 1.0), ("亮点", 1.0), ("好看", 0.8),
        ("哪些", 0.8), ("有什么", 0.8), ("场景", 0.6), ("镜头", 0.5),
        ("find", 1.0), ("search", 1.0), ("highlight", 1.2), ("best", 1.0), ("moment", 1.0),
        ("show", 0.8), ("scenes", 0.8), ("moments", 1.0),
    ],
    IntentType.FILTER_REQUEST: [
        ("只要", 1.5), ("筛选", 1.5), ("过滤", 1.5), ("排除", 1.5), ("保留", 1.5),
        ("以上", 1.0), ("超过", 1.0), ("大于", 1.0), ("高于", 1.0), ("至少", 1.0),
        ("filter", 1.5), ("only", 1.5), ("above", 1.0), ("minimum", 1.0), ("exclude", 1.5),
    ],
    IntentType.SUMMARY_REQUEST: [
        ("总结", 2.0), ("概要", 2.0), ("概述", 2.0), ("简介", 2.0), ("主要内容", 2.0), ("大概", 1.5),
        ("讲了什么", 2.0), ("说了什么", 2.0), ("关于什么", 2.0), ("什么内容", 2.0),
        ("summary", 2.0), ("overview", 2.0), ("summarize", 2.0), ("brief", 1.5), ("about", 1.0),
    ],
    IntentType.EXPORT_REQUEST: [
        ("导出", 2.0), ("下载", 2.0), ("保存", 1.5), ("剪辑", 1.5), ("裁剪", 1.5), ("提取", 1.5),
        ("export", 2.0), ("download", 2.0), ("save", 1.5), ("clip", 1.5), ("extract", 1.5),
    ],
    IntentType.COMPARISON_QUERY: [
        ("对比", 2.0), ("比较", 2.0), ("哪个更", 2.0), ("谁更", 2.0), ("区别", 2.0),
        ("compare", 2.0), ("versus", 2.0), ("which", 1.5), ("difference", 2.0),
    ],
}


@dataclass
class RecognizedIntent:
    """意图识别结果，包含意图类型和提取的槽位信息

    Attributes:
        intent_type: 识别出的意图类型
        scene_types: 目标场景类型列表
        dimension_boosts: 维度权重提升映射，key为维度名，value为提升倍数
        score_threshold: 识别出的分数阈值
        count_limit: 期望的片段数量限制
        confidence: 识别置信度 0-1
        raw_query: 原始用户输入
    """

    intent_type: IntentType
    scene_types: list[str] = field(default_factory=list)
    dimension_boosts: dict[str, float] = field(default_factory=dict)
    score_threshold: float | None = None
    count_limit: int | None = None
    confidence: float = 0.0
    raw_query: str = ""

    def to_pipeline_params(self) -> dict:
        """将意图识别结果转换为 highlight_pipeline 可用的参数字典

        Returns:
            包含 type_filter, score_filters, dimension_weights 的参数字典
        """
        params: dict = {}

        if self.scene_types:
            params["type_filter"] = self.scene_types

        if self.score_threshold is not None:
            params["score_threshold"] = self.score_threshold

        if self.dimension_boosts:
            params["dimension_boosts"] = self.dimension_boosts

        if self.count_limit is not None:
            params["count_limit"] = self.count_limit

        return params


def _classify_intent(query: str) -> tuple[IntentType, float]:
    """基于关键词匹配对用户查询进行意图分类

    采用加权投票策略：直接匹配词权重1.0，模糊匹配词权重0.6，
    同时对 FILTER_REQUEST 做数值检测增强。
    当场景类型关键词出现但无其他强意图信号时，默认为 HIGHLIGHT_SEARCH。

    Args:
        query: 用户输入的自然语言查询

    Returns:
        (意图类型, 置信度) 元组
    """
    query_lower = query.lower()
    scores: dict[IntentType, float] = {}

    for intent_type, keywords in INTENT_KEYWORDS.items():
        score = 0.0
        for kw, weight in keywords:
            if kw in query_lower:
                score += weight
        scores[intent_type] = score

    has_number_threshold = bool(re.search(r"\d+分|[0-9.]+以上|超过\d|至少\d|above\s*\d", query_lower))
    if has_number_threshold:
        scores[IntentType.FILTER_REQUEST] += 1.5

    has_scene_type = False
    for scene_type, keywords in SCENE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                has_scene_type = True
                break
        if has_scene_type:
            break

    if has_scene_type and scores.get(IntentType.HIGHLIGHT_SEARCH, 0) == 0 and max(scores.values()) == 0:
        scores[IntentType.HIGHLIGHT_SEARCH] = 0.8

    if not scores or max(scores.values()) == 0:
        return IntentType.UNKNOWN, 0.0

    best_intent = max(scores, key=scores.get)
    total_score = sum(scores.values())
    confidence = scores[best_intent] / total_score if total_score > 0 else 0.0

    return best_intent, round(min(confidence, 1.0), 2)


def _extract_scene_types(query: str) -> list[str]:
    """从用户查询中提取目标场景类型

    Args:
        query: 用户输入的自然语言查询

    Returns:
        匹配到的场景类型列表
    """
    query_lower = query.lower()
    matched: list[str] = []

    for scene_type, keywords in SCENE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                matched.append(scene_type)
                break

    return matched


def _extract_dimension_boosts(query: str, scene_types: list[str]) -> dict[str, float]:
    """从用户查询和场景类型推断维度权重提升

    当用户明确提到某个维度关键词时，该维度权重提升1.5倍；
    场景类型关联的维度提升1.3倍。

    Args:
        query: 用户输入的自然语言查询
        scene_types: 已识别的场景类型列表

    Returns:
        维度名到提升倍数的映射
    """
    query_lower = query.lower()
    boosts: dict[str, float] = {}

    for dim, keywords in DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                boosts[dim] = 1.5
                break

    scene_dimension_map: dict[str, list[str]] = {
        "action": ["action_intensity", "visual_impact"],
        "drama": ["plot_importance", "emotion_intensity"],
        "emotion": ["emotion_intensity", "facial_expression"],
        "comedy": ["facial_expression", "memorability"],
        "suspense": ["emotion_intensity", "plot_importance"],
    }

    for st in scene_types:
        for dim in scene_dimension_map.get(st, []):
            if dim not in boosts:
                boosts[dim] = 1.3

    return boosts


def _extract_score_threshold(query: str) -> float | None:
    """从用户查询中提取分数阈值

    支持的模式："8分以上", "超过7分", "7.5以上", "above 8"

    Args:
        query: 用户输入的自然语言查询

    Returns:
        提取到的分数阈值，未找到则返回None
    """
    patterns = [
        r"(\d+\.?\d*)\s*分以上",
        r"(\d+\.?\d*)\s*分及以上",
        r"超过\s*(\d+\.?\d*)\s*分",
        r"至少\s*(\d+\.?\d*)\s*分",
        r"大于\s*(\d+\.?\d*)\s*分",
        r"高于\s*(\d+\.?\d*)\s*分",
        r"above\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*分(的|左右)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            value = float(match.group(1))
            if 0 < value <= 10:
                return value

    return None


def _extract_count_limit(query: str) -> int | None:
    """从用户查询中提取数量限制

    支持的模式："前3个", "5个片段", "top 10"

    Args:
        query: 用户输入的自然语言查询

    Returns:
        提取到的数量限制，未找到则返回None
    """
    patterns = [
        r"前\s*(\d+)\s*个",
        r"(\d+)\s*个片段",
        r"(\d+)\s*个名场面",
        r"top\s*(\d+)",
        r"前\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            count = int(match.group(1))
            if 0 < count <= 100:
                return count

    return None


def recognize_intent(query: str) -> RecognizedIntent:
    """对用户自然语言查询进行意图识别和槽位提取

    这是意图识别模块的主入口函数。流程：
    1. 意图分类 → 确定用户想做什么
    2. 场景类型提取 → 确定关注哪些类型
    3. 维度权重推断 → 根据意图动态调整评分权重
    4. 阈值/数量提取 → 提取具体数值约束

    Args:
        query: 用户输入的自然语言查询，如"找出最搞笑的5个片段"

    Returns:
        RecognizedIntent 包含完整的意图识别结果
    """
    if not query or not query.strip():
        return RecognizedIntent(
            intent_type=IntentType.UNKNOWN,
            confidence=0.0,
            raw_query=query or "",
        )

    query = query.strip()
    intent_type, confidence = _classify_intent(query)
    scene_types = _extract_scene_types(query)
    dimension_boosts = _extract_dimension_boosts(query, scene_types)
    score_threshold = _extract_score_threshold(query)
    count_limit = _extract_count_limit(query)

    result = RecognizedIntent(
        intent_type=intent_type,
        scene_types=scene_types,
        dimension_boosts=dimension_boosts,
        score_threshold=score_threshold,
        count_limit=count_limit,
        confidence=confidence,
        raw_query=query,
    )

    logger.info(
        f"意图识别: query='{query}', intent={intent_type.value}, "
        f"types={scene_types}, boosts={dimension_boosts}, "
        f"threshold={score_threshold}, count={count_limit}, conf={confidence}"
    )

    return result
