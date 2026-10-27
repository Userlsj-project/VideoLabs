"""
Step 9: 평가 지표 계산
- 핵심: 테스트 피드백 반영률 (reflected / constructive)
- 보조: 자동화 vs 수동 반영 소요 시간 단축 효과
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricsResult:
    total_comments: int
    constructive_count: int
    spam_count: int
    inappropriate_count: int
    non_constructive_count: int
    reflected_count: int
    reflection_rate: float

    auto_pipeline_seconds: float | None
    manual_estimate_seconds: float | None
    time_savings_percent: float | None

    category_distribution: dict[str, float]


def calculate_metrics(
    video_round,
    manual_time_per_feedback_minutes: float = 3.0,
) -> MetricsResult:
    """
    VideoRound 객체로부터 모든 평가 지표를 계산한다.

    반영률 = (로그에 is_reflected=True인 건설적 피드백 수) / (전체 건설적 피드백 수)
    수동 시간 추정 = constructive_count * manual_time_per_feedback_minutes * 60
    시간 단축 효과 = (manual - auto) / manual * 100
    """
    feedbacks = video_round.feedbacks

    total = len(feedbacks)
    constructive = [f for f in feedbacks if f.category == "constructive"]
    spam = [f for f in feedbacks if f.category == "spam"]
    inappropriate = [f for f in feedbacks if f.category == "inappropriate"]
    non_constructive = [f for f in feedbacks if f.category == "non_constructive"]
    reflected = [f for f in constructive if f.is_reflected]

    constructive_count = len(constructive)
    reflected_count = len(reflected)
    reflection_rate = (
        reflected_count / constructive_count if constructive_count > 0 else 0.0
    )

    # 자동 파이프라인 소요 시간
    auto_seconds: float | None = None
    if video_round.pipeline_start_at and video_round.prompt_updated_at:
        delta = video_round.prompt_updated_at - video_round.pipeline_start_at
        auto_seconds = delta.total_seconds()

    # 수동 추정 시간 (건설적 피드백 개수 × 수동 처리 시간)
    manual_seconds: float | None = None
    time_savings: float | None = None
    if constructive_count > 0:
        manual_seconds = constructive_count * manual_time_per_feedback_minutes * 60
        if auto_seconds is not None and manual_seconds > 0:
            time_savings = max(0.0, (manual_seconds - auto_seconds) / manual_seconds * 100)

    distribution = {}
    if total > 0:
        distribution = {
            "constructive": round(constructive_count / total * 100, 1),
            "non_constructive": round(len(non_constructive) / total * 100, 1),
            "spam": round(len(spam) / total * 100, 1),
            "inappropriate": round(len(inappropriate) / total * 100, 1),
        }

    return MetricsResult(
        total_comments=total,
        constructive_count=constructive_count,
        spam_count=len(spam),
        inappropriate_count=len(inappropriate),
        non_constructive_count=len(non_constructive),
        reflected_count=reflected_count,
        reflection_rate=round(reflection_rate, 4),
        auto_pipeline_seconds=auto_seconds,
        manual_estimate_seconds=manual_seconds,
        time_savings_percent=round(time_savings, 1) if time_savings is not None else None,
        category_distribution=distribution,
    )
