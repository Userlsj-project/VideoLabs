"""
[팀 직접 구현] F-3: 승인 전 참여도 향상 예측 (Pre-Approval Engagement Lift).

과거 라운드 데이터 기반으로 "이 클러스터를 반영하면 얼마나 개선될까?"를 예측.
경쟁사(Runway, Pika, YouTube Studio) 모두 없는 사전 예측 기능.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def estimate_engagement_lift(
    session_id: int,
    approved_cluster_count: int,
    avg_cluster_priority: float,
    has_recurring_issues: bool,
    db_session,
) -> dict:
    """
    과거 라운드 데이터 기반 참여도 향상 예측.

    Args:
        session_id: 현재 세션 ID
        approved_cluster_count: 이번에 승인할 클러스터 수
        avg_cluster_priority: 승인 클러스터 평균 우선순위 점수
        has_recurring_issues: 재발 이슈 포함 여부
        db_session: SQLAlchemy db.session

    Returns:
        {estimated_lift_pct, confidence, based_on_rounds, details}
    """
    try:
        from ..models import VideoRound
        past_rounds = (
            db_session.query(VideoRound)
            .filter(
                VideoRound.session_id == session_id,
                VideoRound.round_number > 1,
                VideoRound.positive_feedback_ratio > 0,
            )
            .order_by(VideoRound.round_number)
            .all()
        )
    except Exception as e:
        logger.warning("[EngagementPredictor] DB 조회 실패: %s", e)
        return {"estimated_lift_pct": None, "confidence": "error"}

    if len(past_rounds) < 2:
        return {
            "estimated_lift_pct": None,
            "confidence": "insufficient_data",
            "based_on_rounds": len(past_rounds),
            "message": "예측을 위해 최소 2개 라운드의 피드백 데이터가 필요합니다.",
        }

    # 연속 라운드 간 향상률 계산
    lifts = []
    for i in range(1, len(past_rounds)):
        prev_ratio = past_rounds[i - 1].positive_feedback_ratio or 0.0
        curr_ratio = past_rounds[i].positive_feedback_ratio or 0.0
        if prev_ratio > 0:
            lift = (curr_ratio - prev_ratio) / prev_ratio * 100
            lifts.append(lift)

    if not lifts:
        return {
            "estimated_lift_pct": None,
            "confidence": "insufficient_data",
            "based_on_rounds": len(past_rounds),
        }

    base_lift = sum(lifts) / len(lifts)

    # 보정 요소
    # 승인 클러스터 수 (더 많이 반영할수록 향상 기대)
    cluster_bonus = approved_cluster_count * 0.03
    # 우선순위가 높은 클러스터 반영 (긴급·심각 이슈 해결)
    priority_bonus = avg_cluster_priority * 0.05
    # 재발 이슈 해결 시 추가 부스트
    recurrence_bonus = 0.15 if has_recurring_issues else 0.0

    adjusted_lift = base_lift * (1 + cluster_bonus + priority_bonus) + recurrence_bonus * abs(base_lift)
    adjusted_lift = max(-50.0, min(100.0, adjusted_lift))  # -50%~+100% 클램프

    confidence = "high" if len(lifts) >= 4 else ("medium" if len(lifts) >= 2 else "low")

    return {
        "estimated_lift_pct": round(adjusted_lift, 1),
        "confidence": confidence,
        "based_on_rounds": len(lifts),
        "avg_historical_lift_pct": round(base_lift, 1),
        "bonuses": {
            "cluster_count": round(cluster_bonus * 100, 1),
            "priority": round(priority_bonus * 100, 1),
            "recurring": round(recurrence_bonus * 100, 1),
        },
    }


def update_round_feedback_stats(video_round, db_session) -> None:
    """
    라운드 완료 후 positive_feedback_ratio와 total_feedback_count를 업데이트.
    approve_clusters 이후 호출해 다음 예측에 사용.
    """
    try:
        feedbacks = video_round.feedbacks
        total = len(feedbacks)
        if total == 0:
            return

        # constructive = positive로 간주
        positive = sum(1 for f in feedbacks if f.category == "constructive")
        ratio = positive / total

        video_round.positive_feedback_ratio = ratio
        video_round.total_feedback_count = total
        db_session.commit()
        logger.debug(
            "[EngagementPredictor] 라운드 %d 통계 업데이트: ratio=%.3f, total=%d",
            video_round.round_number, ratio, total,
        )
    except Exception as e:
        logger.warning("[EngagementPredictor] 통계 업데이트 실패: %s", e)


def get_prompt_performance_history(session_id: int, db_session, top_k: int = 3) -> list[dict]:
    """
    F-1: 현재 세션의 과거 프롬프트 성능 이력 반환.
    프롬프트 입력 시 유사한 과거 성능 데이터를 보여주는 데 사용.
    """
    try:
        from ..models import VideoRound
        rounds = (
            db_session.query(VideoRound)
            .filter(
                VideoRound.session_id == session_id,
                VideoRound.video_gen_prompt != None,  # noqa: E711
                VideoRound.positive_feedback_ratio > 0,
            )
            .order_by(VideoRound.positive_feedback_ratio.desc())
            .limit(top_k)
            .all()
        )

        return [
            {
                "round_number": r.round_number,
                "positive_ratio_pct": round(r.positive_feedback_ratio * 100, 1),
                "total_feedbacks": r.total_feedback_count,
                "prompt_preview": (r.video_gen_prompt or "")[:100] + "...",
                "adapter": r.generator_adapter or "mock",
            }
            for r in rounds
        ]
    except Exception as e:
        logger.warning("[EngagementPredictor] 성능 이력 조회 실패: %s", e)
        return []
