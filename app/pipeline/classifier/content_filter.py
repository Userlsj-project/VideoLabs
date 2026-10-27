"""
[팀 직접 구현] 저작권·부적절 콘텐츠 필터.

6축 분류와 독립적으로 동작하며, 감지된 피드백은
클러스터링·프롬프트 반영 대상에서 자동 제외한다.
rejected_reason = "copyright_or_inappropriate"

통과 기준: 저작권 침해 소지 / 부적절한 콘텐츠 요구 패턴이 없을 것.
"""

from __future__ import annotations

import re

# ── 저작권 침해 패턴 ──────────────────────────────────────────────────────────
_COPYRIGHT_PATTERNS: list[str] = [
    r"저작권\s*(침해|위반)",
    r"copyright\s*infringement",
    r"표절",
    r"무단\s*(사용|도용|복제|배포|전재)",
    r"\bdmca\b",
    r"이\s*영상\s*(훔쳤|도용했|베꼈|복사했)",
    r"license\s*violation",
    r"불법\s*(복제|다운)",
]

# ── 부적절 콘텐츠 패턴 ────────────────────────────────────────────────────────
_INAPPROPRIATE_PATTERNS: list[str] = [
    r"성인\s*(콘텐츠|영상|물)",
    r"adult\s*content",
    r"\b19금\b",
    r"\bnsfw\b",
    r"\bnude\b",
    r"\bexplicit\b",
    r"폭력적(인|으로)\s*(장면|내용|묘사)",
    r"혐오\s*(발언|표현|발언)",
    r"hate\s*speech",
    r"성적\s*(묘사|표현|장면)",
    r"선정적",
]

_COPYRIGHT_RE = [re.compile(p, re.IGNORECASE) for p in _COPYRIGHT_PATTERNS]
_INAPPROPRIATE_RE = [re.compile(p, re.IGNORECASE) for p in _INAPPROPRIATE_PATTERNS]


def detect_content_violation(text: str) -> tuple[bool, str]:
    """
    [팀 직접 구현] 저작권·부적절 콘텐츠 감지.

    Args:
        text: 피드백 댓글 원문

    Returns:
        (violated, reason)
        violated=True  → 클러스터링 제외, rejected_reason 설정
        violated=False → 정상 파이프라인 진행
    """
    for pattern in _COPYRIGHT_RE:
        if pattern.search(text):
            return True, "copyright_or_inappropriate"
    for pattern in _INAPPROPRIATE_RE:
        if pattern.search(text):
            return True, "copyright_or_inappropriate"
    return False, ""
