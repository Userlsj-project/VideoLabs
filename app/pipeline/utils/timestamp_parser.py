"""
[팀 직접 구현] 시청자 댓글에서 타임스탬프를 추출하고 샷 번호로 매핑.

YouTube 댓글에는 "0:15에서 카메라가 어색해요" 같은 타임스탬프가 자주 포함됨.
이를 파싱해 해당 시간대의 샷을 특정하는 것이 핵심.
"""

from __future__ import annotations
import re


# HH:MM:SS 또는 MM:SS 형식 — 끝 \b 제거: 한글은 Unicode \w라 word boundary 불일치
_TIMESTAMP_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?=\D|$)")


def extract_timestamp(text: str) -> float | None:
    """
    댓글 텍스트에서 첫 번째 타임스탬프를 추출해 초(float)로 반환.
    타임스탬프가 없으면 None 반환.

    Examples:
        "0:15에서 카메라가 어색해요"  → 15.0
        "1:30 부분이 너무 빠릅니다"   → 90.0
        "좋아요!"                     → None
    """
    m = _TIMESTAMP_RE.search(text)
    if not m:
        return None

    # MM:SS 형식
    if m.group(3) is None:
        minutes = int(m.group(1))
        seconds = int(m.group(2))
        return float(minutes * 60 + seconds)

    # HH:MM:SS 형식
    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    return float(hours * 3600 + minutes * 60 + seconds)


def map_timestamp_to_shot_number(
    timestamp_seconds: float,
    shots: list,  # list[ShotRound]
) -> int | None:
    """
    타임스탬프(초)가 어느 샷의 구간에 속하는지 반환.
    구간: shot.start_time_seconds <= timestamp < shot.end_time_seconds

    Args:
        timestamp_seconds: 댓글에서 추출한 시간(초)
        shots: ShotRound 객체 목록 (start/end_time_seconds 속성 필요)

    Returns:
        shot_number (1-based) 또는 None
    """
    for shot in shots:
        start = getattr(shot, "start_time_seconds", None)
        end = getattr(shot, "end_time_seconds", None)
        if start is None or end is None:
            continue
        if start <= timestamp_seconds < end:
            return shot.shot_number
    # 마지막 샷 범위 밖이면 가장 가까운 샷에 매핑
    if shots:
        last = max(shots, key=lambda s: getattr(s, "end_time_seconds", 0))
        if timestamp_seconds >= getattr(last, "start_time_seconds", 0):
            return last.shot_number
    return None
