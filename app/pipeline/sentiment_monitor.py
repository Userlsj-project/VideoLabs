"""
[팀 직접 구현] F-5: 실시간 감정 트렌드 감지 (Sentiment Drift Alert).

CUSUM 변형 알고리즘으로 업로드 후 댓글 부정 비율 급증을 탐지한다.
YouTube Analytics는 배치 분석만 제공; 이 모듈은 실시간 변화점을 탐지한다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def detect_sentiment_drift(
    feedbacks: list,
    window_minutes: int = 120,
    swing_threshold: float = 0.25,
) -> dict:
    """
    시간 창 기반 감정 변화점 감지.

    Args:
        feedbacks: Feedback 객체 목록 (category, collected_at 필드 필요)
        window_minutes: 분석 윈도우 (기본 2시간)
        swing_threshold: 부정 비율 급증 임계값 (기본 25%p)

    Returns:
        {drift_detected, swing_pct, baseline_neg_pct, latest_neg_pct, alert_level}
    """
    if len(feedbacks) < 20:
        return {"drift_detected": False, "reason": "insufficient_data", "swing_pct": 0}

    def is_negative(fb) -> bool:
        return fb.category not in ("constructive",) and fb.category is not None

    # 시간순 정렬
    try:
        sorted_fb = sorted(
            feedbacks,
            key=lambda x: x.collected_at or datetime.min
        )
    except Exception:
        return {"drift_detected": False, "reason": "sort_error", "swing_pct": 0}

    # 최근 8시간 내 댓글만 분석
    if sorted_fb[-1].collected_at:
        cutoff = sorted_fb[-1].collected_at - timedelta(hours=8)
        recent = [f for f in sorted_fb if f.collected_at and f.collected_at >= cutoff]
    else:
        recent = sorted_fb[-40:]

    if len(recent) < 10:
        return {"drift_detected": False, "reason": "insufficient_recent", "swing_pct": 0}

    # 4분할 윈도우로 나눠 부정 비율 계산
    window_size = max(5, len(recent) // 4)
    windows = []
    for i in range(0, len(recent) - window_size + 1, max(1, window_size // 2)):
        window = recent[i:i + window_size]
        if not window:
            continue
        neg_ratio = sum(1 for f in window if is_negative(f)) / len(window)
        windows.append(neg_ratio)

    if len(windows) < 2:
        return {"drift_detected": False, "reason": "insufficient_windows", "swing_pct": 0}

    baseline = windows[0]
    latest = windows[-1]
    swing = latest - baseline

    result = {
        "drift_detected": swing > swing_threshold,
        "swing_pct": round(swing * 100, 1),
        "baseline_neg_pct": round(baseline * 100, 1),
        "latest_neg_pct": round(latest * 100, 1),
        "total_analyzed": len(recent),
        "windows": len(windows),
    }

    if result["drift_detected"]:
        result["alert_level"] = "critical" if swing > 0.40 else "warning"
    else:
        result["alert_level"] = "normal"

    return result


def get_trending_topics(feedbacks: list, top_k: int = 3) -> list[dict]:
    """
    최근 댓글에서 급증하는 토픽을 감지한다.
    전체 대비 최근 절반의 토픽 비율 변화를 계산.
    """
    if len(feedbacks) < 10:
        return []

    from collections import Counter

    mid = len(feedbacks) // 2
    early = feedbacks[:mid]
    recent = feedbacks[mid:]

    def count_topics(fbs):
        c = Counter()
        for fb in fbs:
            if fb.topic_category and fb.topic_category != "other":
                c[fb.topic_category] += 1
        return c

    early_counts = count_topics(early)
    recent_counts = count_topics(recent)

    trending = []
    all_topics = set(list(early_counts.keys()) + list(recent_counts.keys()))
    for topic in all_topics:
        e = early_counts.get(topic, 0)
        r = recent_counts.get(topic, 0)
        if e == 0 and r > 0:
            change_pct = 100.0
        elif e > 0:
            change_pct = (r - e) / e * 100
        else:
            continue
        if change_pct > 20:
            trending.append({
                "topic": topic,
                "change_pct": round(change_pct, 1),
                "early_count": e,
                "recent_count": r,
            })

    return sorted(trending, key=lambda x: x["change_pct"], reverse=True)[:top_k]
