import math
from dataclasses import dataclass


@dataclass
class IntentMetrics:
    """意图识别评测指标

    Attributes:
        total: 测试用例总数
        correct: 正确预测数
        precision: 精确率
        recall: 召回率
        f1: F1分数
        per_class_stats: 各意图类型的详细统计
    """

    total: int
    correct: int
    precision: float
    recall: float
    f1: float
    per_class_stats: dict


def compute_intent_metrics(predictions: list[dict]) -> IntentMetrics:
    """计算意图识别的精确率、召回率和F1分数

    采用 macro-average 策略，先计算每个类别的指标再取平均，
    避免类别不平衡导致指标偏差。

    Args:
        predictions: 预测结果列表，每项包含 expected 和 predicted 字段
            [{"expected": "highlight_search", "predicted": "highlight_search"}, ...]

    Returns:
        IntentMetrics 包含完整的评测指标
    """
    if not predictions:
        return IntentMetrics(
            total=0, correct=0, precision=0.0, recall=0.0, f1=0.0,
            per_class_stats={}
        )

    label_set = set()
    for p in predictions:
        label_set.add(p["expected"])
        label_set.add(p["predicted"])

    per_class_stats = {}
    precisions = []
    recalls = []

    for label in sorted(label_set):
        tp = sum(1 for p in predictions if p["expected"] == label and p["predicted"] == label)
        fp = sum(1 for p in predictions if p["expected"] != label and p["predicted"] == label)
        fn = sum(1 for p in predictions if p["expected"] == label and p["predicted"] != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_class_stats[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": tp + fn,
        }
        precisions.append(precision)
        recalls.append(recall)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    correct = sum(1 for p in predictions if p["expected"] == p["predicted"])

    return IntentMetrics(
        total=len(predictions),
        correct=correct,
        precision=round(macro_precision, 4),
        recall=round(macro_recall, 4),
        f1=round(macro_f1, 4),
        per_class_stats=per_class_stats,
    )


def compute_weighted_score_correlation(
    system_scores: list[float],
    human_scores: list[float],
) -> dict:
    """计算系统评分与人工评分的一致性指标

    使用 Pearson 相关系数衡量系统评分与人工评分的线性相关性，
    用于评估 VLM 评分模块的准确性。

    Args:
        system_scores: 系统输出的评分列表
        human_scores: 人工标注的评分列表

    Returns:
        包含 pearson_r, rmse, mae 的指标字典
    """
    if len(system_scores) != len(human_scores) or len(system_scores) < 2:
        return {"pearson_r": 0.0, "rmse": 0.0, "mae": 0.0, "sample_count": 0}

    n = len(system_scores)
    mean_sys = sum(system_scores) / n
    mean_human = sum(human_scores) / n

    cov = sum((s - mean_sys) * (h - mean_human) for s, h in zip(system_scores, human_scores))
    std_sys = math.sqrt(sum((s - mean_sys) ** 2 for s in system_scores))
    std_human = math.sqrt(sum((h - mean_human) ** 2 for h in human_scores))

    pearson_r = cov / (std_sys * std_human) if std_sys > 0 and std_human > 0 else 0.0

    rmse = math.sqrt(sum((s - h) ** 2 for s, h in zip(system_scores, human_scores)) / n)
    mae = sum(abs(s - h) for s, h in zip(system_scores, human_scores)) / n

    return {
        "pearson_r": round(pearson_r, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "sample_count": n,
    }


def compute_highlight_detection_metrics(
    detected_segments: list[dict],
    ground_truth_segments: list[dict],
    iou_threshold: float = 0.3,
) -> dict:
    """计算名场面检测的精确率、召回率和F1

    使用时间区间 IoU (Intersection over Union) 判断检测是否命中，
    IoU 超过阈值视为正确检测。

    Args:
        detected_segments: 系统检测的片段列表，每项含 start_seconds, end_seconds
        ground_truth_segments: 人工标注的片段列表，格式同上
        iou_threshold: IoU 判定阈值，默认0.3

    Returns:
        包含 precision, recall, f1, iou_threshold 的指标字典
    """
    if not ground_truth_segments:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou_threshold": iou_threshold}

    matched_gt = set()
    true_positives = 0

    for det in detected_segments:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth_segments):
            if gt_idx in matched_gt:
                continue

            iou = _compute_iou(
                det["start_seconds"], det["end_seconds"],
                gt["start_seconds"], gt["end_seconds"],
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    false_positives = len(detected_segments) - true_positives
    false_negatives = len(ground_truth_segments) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _compute_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """计算两个时间区间的 IoU (Intersection over Union)

    Args:
        start1, end1: 第一个时间区间的起止秒数
        start2, end2: 第二个时间区间的起止秒数

    Returns:
        IoU 值，0到1之间
    """
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0.0, intersection_end - intersection_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = max(0.0, union_end - union_start)

    return intersection / union if union > 0 else 0.0
