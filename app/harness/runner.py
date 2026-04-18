import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.services.intent_recognizer import recognize_intent, IntentType
from app.services.highlight_pipeline import _compute_weighted_score, _merge_consecutive_highlights, _filter_similar_scenes, DEFAULT_WEIGHTS
from app.harness.metrics import compute_intent_metrics, compute_weighted_score_correlation, compute_highlight_detection_metrics
from app.harness.cases import INTENT_TEST_CASES, SCORING_TEST_CASES, PIPELINE_SMOKE_TESTS

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """单个测试用例的执行结果

    Attributes:
        name: 测试名称
        passed: 是否通过
        detail: 详细信息
        duration_ms: 执行耗时(毫秒)
    """

    name: str
    passed: bool
    detail: str
    duration_ms: float


@dataclass
class HarnessReport:
    """Harness 完整评测报告

    Attributes:
        timestamp: 报告生成时间
        total_tests: 总测试数
        passed_tests: 通过数
        failed_tests: 失败数
        intent_metrics: 意图识别评测指标
        scoring_metrics: 评分一致性指标
        smoke_results: 冒烟测试结果列表
        benchmark_results: 性能基准测试结果
    """

    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    intent_metrics: dict = field(default_factory=dict)
    scoring_metrics: dict = field(default_factory=dict)
    smoke_results: list[TestResult] = field(default_factory=list)
    benchmark_results: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """计算测试通过率"""
        return round(self.passed_tests / self.total_tests, 4) if self.total_tests > 0 else 0.0

    def to_dict(self) -> dict:
        """将报告序列化为字典"""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "pass_rate": f"{self.pass_rate:.1%}",
            },
            "intent_recognition": self.intent_metrics,
            "scoring_consistency": self.scoring_metrics,
            "smoke_tests": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, "duration_ms": r.duration_ms}
                for r in self.smoke_results
            ],
            "benchmarks": self.benchmark_results,
        }


def run_intent_tests() -> tuple[list[TestResult], dict]:
    """运行意图识别评测套件

    对每条测试用例调用 recognize_intent，比对预测意图与期望意图，
    同时检查场景类型、阈值、数量等槽位的提取准确性。

    Returns:
        (测试结果列表, 意图识别指标字典)
    """
    results: list[TestResult] = []
    predictions: list[dict] = []

    for case in INTENT_TEST_CASES:
        start = time.perf_counter()
        query = case["query"]
        expected_intent = case["expected_intent"]

        intent = recognize_intent(query)
        duration_ms = (time.perf_counter() - start) * 1000

        predicted = intent.intent_type.value
        intent_match = predicted == expected_intent

        type_match = True
        if "expected_types" in case:
            expected_types = case["expected_types"]
            type_match = set(intent.scene_types) == set(expected_types)

        threshold_match = True
        if "expected_threshold" in case:
            threshold_match = intent.score_threshold == case["expected_threshold"]

        count_match = True
        if "expected_count" in case:
            count_match = intent.count_limit == case["expected_count"]

        boost_match = True
        if "expected_boosts" in case:
            for dim in case["expected_boosts"]:
                if dim not in intent.dimension_boosts:
                    boost_match = False

        all_match = intent_match and type_match and threshold_match and count_match and boost_match

        details = []
        if not intent_match:
            details.append(f"意图: 期望={expected_intent}, 实际={predicted}")
        if not type_match:
            details.append(f"类型: 期望={case.get('expected_types', [])}, 实际={intent.scene_types}")
        if not threshold_match:
            details.append(f"阈值: 期望={case.get('expected_threshold')}, 实际={intent.score_threshold}")
        if not count_match:
            details.append(f"数量: 期望={case.get('expected_count')}, 实际={intent.count_limit}")
        if not boost_match:
            details.append(f"增强: 期望包含={case.get('expected_boosts', [])}, 实际={list(intent.dimension_boosts.keys())}")

        detail_str = "; ".join(details) if details else "全部匹配"

        results.append(TestResult(
            name=f"intent: {query}",
            passed=all_match,
            detail=detail_str,
            duration_ms=round(duration_ms, 2),
        ))

        predictions.append({"expected": expected_intent, "predicted": predicted})

    metrics = compute_intent_metrics(predictions)
    intent_metrics = {
        "total": metrics.total,
        "correct": metrics.correct,
        "accuracy": round(metrics.correct / metrics.total, 4) if metrics.total > 0 else 0.0,
        "macro_precision": metrics.precision,
        "macro_recall": metrics.recall,
        "macro_f1": metrics.f1,
        "per_class": metrics.per_class_stats,
    }

    return results, intent_metrics


def run_scoring_tests() -> tuple[list[TestResult], dict]:
    """运行评分一致性评测

    使用人工标注的测试用例，计算系统加权评分与人工评分的相关性，
    评估 VLM 评分模块的准确性。

    Returns:
        (测试结果列表, 评分一致性指标字典)
    """
    results: list[TestResult] = []
    system_scores: list[float] = []
    human_scores: list[float] = []

    for case in SCORING_TEST_CASES:
        start = time.perf_counter()
        computed = _compute_weighted_score(case["system_scores"])
        duration_ms = (time.perf_counter() - start) * 1000

        diff = abs(computed - case["human_score"])
        passed = diff <= 2.0

        system_scores.append(computed)
        human_scores.append(case["human_score"])

        results.append(TestResult(
            name=f"scoring: {case['description']}",
            passed=passed,
            detail=f"系统={computed:.2f}, 人工={case['human_score']:.2f}, 差值={diff:.2f}",
            duration_ms=round(duration_ms, 2),
        ))

    correlation = compute_weighted_score_correlation(system_scores, human_scores)

    return results, correlation


def run_smoke_tests() -> list[TestResult]:
    """运行核心模块冒烟测试

    不依赖外部资源（视频文件、API），仅验证内部逻辑正确性。
    包括加权评分计算、片段合并、相似去重等核心函数。

    Returns:
        测试结果列表
    """
    results: list[TestResult] = []

    start = time.perf_counter()
    scores = {"visual_impact": 8, "cinematography": 6, "emotion_intensity": 7,
              "facial_expression": 5, "plot_importance": 6, "action_intensity": 4,
              "audio_energy": 5, "memorability": 7}
    base_score = _compute_weighted_score(scores)
    boosted_score = _compute_weighted_score(scores, {"visual_impact": 1.5})
    passed = boosted_score > base_score
    results.append(TestResult(
        name="pipeline_weighted_score",
        passed=passed,
        detail=f"基础分={base_score:.2f}, 增强visual_impact后={boosted_score:.2f}, 增强高分维度后应更高",
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    ))

    start = time.perf_counter()
    highlights = [
        {"start_seconds": 0, "end_seconds": 5, "start_time": "00:00", "end_time": "00:05",
         "scores": {"total": 7.0}, "keyframe_url": "/a.jpg", "description": "场景A", "type": "action"},
        {"start_seconds": 6, "end_seconds": 10, "start_time": "00:06", "end_time": "00:10",
         "scores": {"total": 7.2}, "keyframe_url": "/b.jpg", "description": "场景B", "type": "action"},
        {"start_seconds": 20, "end_seconds": 25, "start_time": "00:20", "end_time": "00:25",
         "scores": {"total": 8.0}, "keyframe_url": "/c.jpg", "description": "场景C", "type": "drama"},
    ]
    merged = _merge_consecutive_highlights(highlights)
    passed = len(merged) == 2 and merged[0]["end_seconds"] == 10
    results.append(TestResult(
        name="pipeline_merge_consecutive",
        passed=passed,
        detail=f"输入3个片段, 合并后{len(merged)}个, 期望2个",
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    ))

    start = time.perf_counter()
    similar_highlights = [
        {"start_seconds": 0, "end_seconds": 5, "description": "激烈打斗场面", "type": "action",
         "scores": {"total": 7.0}, "keyframe_url": "/a.jpg", "start_time": "00:00", "end_time": "00:05"},
        {"start_seconds": 30, "end_seconds": 35, "description": "激烈打斗场景", "type": "action",
         "scores": {"total": 7.5}, "keyframe_url": "/b.jpg", "start_time": "00:30", "end_time": "00:35"},
        {"start_seconds": 60, "end_seconds": 65, "description": "温馨感人时刻", "type": "emotion",
         "scores": {"total": 8.0}, "keyframe_url": "/c.jpg", "start_time": "01:00", "end_time": "01:05"},
    ]
    filtered = _filter_similar_scenes(similar_highlights)
    passed = len(filtered) == 2
    results.append(TestResult(
        name="pipeline_filter_similar",
        passed=passed,
        detail=f"输入3个片段, 去重后{len(filtered)}个, 期望2个",
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    ))

    start = time.perf_counter()
    intent = recognize_intent("找出最搞笑的5个片段")
    passed = (
        intent.intent_type == IntentType.HIGHLIGHT_SEARCH
        and "comedy" in intent.scene_types
        and intent.count_limit == 5
    )
    results.append(TestResult(
        name="intent_recognizer_basic",
        passed=passed,
        detail=f"意图={intent.intent_type.value}, 类型={intent.scene_types}, 数量={intent.count_limit}",
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    ))

    start = time.perf_counter()
    det_metrics = compute_highlight_detection_metrics(
        detected_segments=[
            {"start_seconds": 1, "end_seconds": 5},
            {"start_seconds": 20, "end_seconds": 25},
            {"start_seconds": 50, "end_seconds": 55},
        ],
        ground_truth_segments=[
            {"start_seconds": 2, "end_seconds": 6},
            {"start_seconds": 21, "end_seconds": 24},
            {"start_seconds": 80, "end_seconds": 85},
        ],
        iou_threshold=0.3,
    )
    passed = det_metrics["precision"] > 0 and det_metrics["recall"] > 0
    results.append(TestResult(
        name="highlight_detection_metrics",
        passed=passed,
        detail=f"P={det_metrics['precision']}, R={det_metrics['recall']}, F1={det_metrics['f1']}",
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    ))

    return results


def run_benchmarks() -> dict:
    """运行性能基准测试

    测量核心函数的执行耗时，用于发现性能瓶颈和追踪优化效果。
    每个测试重复执行多次取平均值。

    Returns:
        基准测试结果字典
    """
    iterations = 1000
    results = {}

    start = time.perf_counter()
    for _ in range(iterations):
        recognize_intent("找出最搞笑的5个片段")
    results["intent_recognition_per_call_ms"] = round(
        (time.perf_counter() - start) * 1000 / iterations, 3
    )

    start = time.perf_counter()
    test_scores = {"visual_impact": 8, "cinematography": 6, "emotion_intensity": 7,
                   "facial_expression": 5, "plot_importance": 6, "action_intensity": 4,
                   "audio_energy": 5, "memorability": 7}
    for _ in range(iterations):
        _compute_weighted_score(test_scores)
    results["weighted_score_per_call_ms"] = round(
        (time.perf_counter() - start) * 1000 / iterations, 3
    )

    start = time.perf_counter()
    test_highlights = [
        {"start_seconds": i * 10, "end_seconds": i * 10 + 5, "start_time": f"00:{i:02d}",
         "end_time": f"00:{i:02d}", "scores": {"total": 7.0 + i * 0.1},
         "keyframe_url": f"/{i}.jpg", "description": f"场景{i}", "type": "action"}
        for i in range(50)
    ]
    for _ in range(iterations):
        _merge_consecutive_highlights(test_highlights)
    results["merge_50_highlights_per_call_ms"] = round(
        (time.perf_counter() - start) * 1000 / iterations, 3
    )

    start = time.perf_counter()
    for _ in range(iterations):
        _filter_similar_scenes(test_highlights)
    results["filter_similar_50_per_call_ms"] = round(
        (time.perf_counter() - start) * 1000 / iterations, 3
    )

    results["iterations"] = iterations

    return results


def run_full_harness() -> HarnessReport:
    """运行完整的 Harness 评测套件

    执行顺序：
    1. 意图识别评测 → 验证 NLU 准确性
    2. 评分一致性评测 → 验证 VLM 评分质量
    3. 冒烟测试 → 验证核心逻辑正确性
    4. 性能基准 → 量化执行效率

    Returns:
        HarnessReport 完整评测报告
    """
    from datetime import datetime, timezone

    logger.info("=" * 60)
    logger.info("开始运行 Harness 评测套件")
    logger.info("=" * 60)

    all_results: list[TestResult] = []

    logger.info("\n[1/4] 意图识别评测...")
    intent_results, intent_metrics = run_intent_tests()
    all_results.extend(intent_results)
    logger.info(f"  意图识别完成: {sum(1 for r in intent_results if r.passed)}/{len(intent_results)} 通过")

    logger.info("\n[2/4] 评分一致性评测...")
    scoring_results, scoring_metrics = run_scoring_tests()
    all_results.extend(scoring_results)
    logger.info(f"  评分评测完成: {sum(1 for r in scoring_results if r.passed)}/{len(scoring_results)} 通过")

    logger.info("\n[3/4] 冒烟测试...")
    smoke_results = run_smoke_tests()
    all_results.extend(smoke_results)
    logger.info(f"  冒烟测试完成: {sum(1 for r in smoke_results if r.passed)}/{len(smoke_results)} 通过")

    logger.info("\n[4/4] 性能基准测试...")
    benchmark_results = run_benchmarks()
    logger.info(f"  基准测试完成:")
    for k, v in benchmark_results.items():
        if k != "iterations":
            logger.info(f"    {k}: {v}ms")

    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed

    report = HarnessReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_tests=len(all_results),
        passed_tests=passed,
        failed_tests=failed,
        intent_metrics=intent_metrics,
        scoring_metrics=scoring_metrics,
        smoke_results=smoke_results,
        benchmark_results=benchmark_results,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"评测完成: {passed}/{len(all_results)} 通过 ({report.pass_rate:.1%})")
    logger.info("=" * 60)

    return report


def print_report(report: HarnessReport):
    """将评测报告以可读格式输出到控制台

    Args:
        report: HarnessReport 评测报告
    """
    data = report.to_dict()

    print("\n" + "=" * 60)
    print("  📊 Video Analyzer Harness 评测报告")
    print("=" * 60)

    print(f"\n📅 时间: {data['timestamp']}")
    print(f"📈 总计: {data['summary']['total']} | ✅ 通过: {data['summary']['passed']} | ❌ 失败: {data['summary']['failed']} | 通过率: {data['summary']['pass_rate']}")

    im = data["intent_recognition"]
    print(f"\n🧠 意图识别指标:")
    print(f"   准确率: {im.get('accuracy', 0):.1%} | Macro-P: {im.get('macro_precision', 0):.4f} | Macro-R: {im.get('macro_recall', 0):.4f} | Macro-F1: {im.get('macro_f1', 0):.4f}")
    if "per_class" in im:
        print("   各类别详情:")
        for label, stats in im["per_class"].items():
            print(f"     {label}: P={stats['precision']:.4f} R={stats['recall']:.4f} support={stats['support']}")

    sm = data["scoring_consistency"]
    print(f"\n📊 评分一致性:")
    print(f"   Pearson-r: {sm.get('pearson_r', 0):.4f} | RMSE: {sm.get('rmse', 0):.4f} | MAE: {sm.get('mae', 0):.4f} | 样本数: {sm.get('sample_count', 0)}")

    print(f"\n🔬 冒烟测试:")
    for t in data["smoke_tests"]:
        icon = "✅" if t["passed"] else "❌"
        print(f"   {icon} {t['name']}: {t['detail']} ({t['duration_ms']:.2f}ms)")

    bm = data["benchmarks"]
    print(f"\n⚡ 性能基准 ({bm.get('iterations', 0)}次迭代):")
    for k, v in bm.items():
        if k != "iterations":
            print(f"   {k}: {v}ms")

    print("\n" + "=" * 60)


def main():
    """Harness 入口函数"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_full_harness()
    print_report(report)

    report_path = Path(__file__).resolve().parent.parent.parent / "harness_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
