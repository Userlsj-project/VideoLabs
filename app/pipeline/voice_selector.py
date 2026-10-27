"""
[팀 직접 설계] 피드백 텍스트 분석 → 최적 ElevenLabs 보이스 자동 선택.

피드백 키워드를 분석해 원하는 보이스 특성을 도출하고,
voice_catalog에서 가장 적합한 보이스 ID를 반환한다.
"""

from __future__ import annotations

import logging

from .voice_catalog import get_active_voices, get_voice_by_id

logger = logging.getLogger(__name__)

# ── 피드백 → 감정 프로파일 매핑 ──────────────────────────────────────────────
# 각 큐 위치(첫/중간/마지막)에 삽입할 ElevenLabs v4 감정 태그 + 속도 결정
# opening/closing이 None이면 해당 위치에 태그 없음

_EMOTION_RULES: list[tuple[list[str], dict]] = [
    # 단조롭다 → 활기차게
    (["단조롭", "감정없", "밋밋", "지루", "심심"],
     {"opening": "excited",   "middle": None, "closing": "inspired", "speed": 1.05}),

    # 과장/느끼 → 태그 제거, 속도 살짝 느리게
    (["느끼", "과장", "진부", "억지"],
     {"opening": None,        "middle": None, "closing": None,       "speed": 0.95}),

    # AI/로봇 느낌 → 자연체 유지, 약간 느리게
    (["ai같", "로봇", "인공", "어색", "부자연", "티가나", "티나"],
     {"opening": None,        "middle": None, "closing": None,       "speed": 0.93}),

    # 너무 빠르다
    (["빠르다", "빠른", "너무 빨리", "급하게"],
     {"speed": 0.85}),

    # 너무 느리다
    (["느리다", "느린", "너무 느려", "지루하게 느"],
     {"speed": 1.1}),
]

# 피드백 없을 때 기본 프로파일 (2라운드부터 적용)
_DEFAULT_EMOTION_PROFILE = {
    "opening": "proud",   # 첫 큐: 자신감 있는 선언
    "middle":  None,      # 중간 큐: 중립 (명확하게)
    "closing": "proud",   # 마지막 큐: 자부심
    "speed":   1.0,
}

def get_emotion_profile(feedback_texts: list[str]) -> dict:
    """
    [팀 직접 구현] 피드백 텍스트에서 큐 위치별 감정 태그 + 속도 프로파일 도출.

    2라운드 이후 generate_all_narrations()에 전달해 각 큐에 ElevenLabs v4
    감정 태그를 삽입한다. 피드백이 없어도 기본 프로파일을 반환한다.

    Returns:
        {
          "opening": str | None,  # 첫 큐 감정 태그
          "middle":  str | None,  # 중간 큐 감정 태그 (None = 태그 없음)
          "closing": str | None,  # 마지막 큐 감정 태그
          "speed":   float,       # 말하기 속도 (1.0 = 기본)
        }
    """
    profile: dict = dict(_DEFAULT_EMOTION_PROFILE)
    combined = " ".join(feedback_texts).lower()
    speed_overrides: list[float] = []

    for keywords, overrides in _EMOTION_RULES:
        if any(kw in combined for kw in keywords):
            for key, val in overrides.items():
                if key == "speed":
                    speed_overrides.append(val)
                else:
                    profile[key] = val  # None 포함 — 태그 명시적 제거

    if speed_overrides:
        profile["speed"] = round(sum(speed_overrides) / len(speed_overrides), 3)

    return profile


# ── 피드백 키워드 → 원하는 특성 매핑 ─────────────────────────────────────────
# (패턴, 원하는 태그, 피해야 할 스타일, 가중치)

_FEEDBACK_RULES: list[tuple[list[str], list[str], list[str], float]] = [
    # 키워드              원하는 태그              피해야 할 스타일     가중치
    (["ai같", "로봇", "인공", "어색", "부자연", "티가나", "티나"],
     ["자연스러운"],        ["dramatic"],           2.0),

    (["느끼", "과장", "진부", "억지"],
     ["공식적", "절제된"],  ["dramatic", "energetic"], 2.0),

    (["단조롭", "감정없", "밋밋", "지루"],
     ["활기찬", "감성적"],  ["formal"],             1.5),

    (["여성", "여자", "female"],
     ["여성"],             [],                     3.0),

    (["남성", "남자", "male"],
     ["남성"],             [],                     3.0),

    (["아나운서", "공식", "뉴스"],
     ["아나운서", "공식적"], [],                    1.5),

    (["친근", "따뜻", "편안"],
     ["친근한", "자연스러운"], ["formal"],          1.5),

    (["젊", "밝", "에너지", "활기"],
     ["활기찬", "밝은"],    [],                     1.5),

    (["중후", "무게감", "신뢰"],
     ["중후한", "신뢰감"],  [],                     1.5),

    (["감성", "감동", "스토리", "따뜻", "포근"],
     ["감성적", "따뜻한", "스토리텔링"], [],         1.5),
]


def analyze_feedback(feedback_texts: list[str]) -> dict:
    """
    [팀 직접 구현] 피드백 텍스트에서 원하는 보이스 특성 도출.

    Returns:
        {
          "desired_tags": [(tag, weight), ...],
          "avoid_styles": [style, ...],
          "detected_issues": [keyword, ...],
        }
    """
    combined = " ".join(feedback_texts).lower()
    desired: dict[str, float] = {}
    avoid_styles: set[str] = set()
    detected_issues: list[str] = []

    for keywords, want_tags, bad_styles, weight in _FEEDBACK_RULES:
        matched = [kw for kw in keywords if kw in combined]
        if matched:
            detected_issues.extend(matched)
            for tag in want_tags:
                desired[tag] = desired.get(tag, 0) + weight
            avoid_styles.update(bad_styles)

    return {
        "desired_tags": sorted(desired.items(), key=lambda x: x[1], reverse=True),
        "avoid_styles": list(avoid_styles),
        "detected_issues": list(set(detected_issues)),
    }


def score_voice(voice: dict, analysis: dict) -> float:
    """
    [팀 직접 구현] 보이스에 분석 결과를 적용해 적합도 점수 계산.

    - 원하는 태그 일치 시 가중치만큼 점수 상승
    - 피해야 할 스타일이면 -10 페널티 (사실상 제외)
    - 취약점(weakness)이 피드백 키워드에 포함되면 -5 페널티
    """
    score = 0.0

    # 피해야 할 스타일 → 강한 페널티
    if voice["style"] in analysis["avoid_styles"]:
        score -= 10.0

    # 원하는 태그 매칭
    voice_tags = [t.lower() for t in voice.get("tags", [])]
    for tag, weight in analysis["desired_tags"]:
        if tag.lower() in voice_tags:
            score += weight

    # 취약점 페널티: 피드백 키워드가 이 보이스의 weakness에 해당하면 감점
    combined_issues = " ".join(analysis["detected_issues"]).lower()
    for weak in voice.get("weakness", []):
        if weak.lower() in combined_issues:
            score -= 5.0

    return score


def select_voice(
    feedback_texts: list[str],
    current_voice_id: str,
) -> tuple[str, str, dict]:
    """
    [팀 직접 구현] 피드백 기반 최적 보이스 선택.

    Args:
        feedback_texts: 음성 관련 피드백 텍스트 목록
        current_voice_id: 현재 사용 중인 voice_id

    Returns:
        (선택된 voice_id, 변경 이유 요약, 분석 결과)
        변경 없으면 current_voice_id 그대로 반환
    """
    active_voices = get_active_voices()
    if not active_voices:
        logger.warning("[VoiceSelector] 활성화된 보이스 없음 — 현재 보이스 유지")
        return current_voice_id, "등록된 보이스 없음", {}

    analysis = analyze_feedback(feedback_texts)
    logger.info("[VoiceSelector] 피드백 분석: %s", analysis)

    if not analysis["desired_tags"] and not analysis["avoid_styles"]:
        return current_voice_id, "변경 필요 없음", analysis

    scores: list[tuple[float, dict]] = []
    for voice in active_voices:
        s = score_voice(voice, analysis)
        scores.append((s, voice))
        logger.debug("[VoiceSelector] %s → %.2f점", voice["name"], s)

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_voice = scores[0]

    # 현재 보이스 점수
    current_voice = get_voice_by_id(current_voice_id)
    if current_voice:
        current_score = score_voice(current_voice, analysis)
    else:
        current_score = -999

    # 현재 보이스가 이미 최적이면 유지
    if best_voice["voice_id"] == current_voice_id or best_score <= current_score:
        return current_voice_id, "현재 보이스가 최적", analysis

    issues = ", ".join(analysis["detected_issues"][:3])
    reason = (
        f"피드백 '{issues}' 감지 → "
        f"{best_voice['name']}으로 변경 (적합도 {best_score:.1f}점)"
    )
    logger.info("[VoiceSelector] 선택: %s (%s)", best_voice["name"], reason)
    return best_voice["voice_id"], reason, analysis
