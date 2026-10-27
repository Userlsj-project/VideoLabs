"""
[팀 직접 구현]
규칙 기반 폴백 분류기.
zero-shot 모델 로드 실패 시 오프라인에서도 동작하는 최후 폴백.
한국어·영어 키워드를 모두 포함.
"""

from __future__ import annotations

import re

from .base import BaseClassifier, ClassificationResult, topic_to_scope

# ── [팀 직접 설계] 키워드 사전 ────────────────────────────────────────────────

_CONSTRUCTIVE_KW = [
    # 한국어
    "줄여", "키워", "늘려", "바꿔", "넣어", "추가", "수정", "개선", "고쳐",
    "느리게", "빠르게", "크게", "작게", "밝게", "어둡게",
    "해주세요", "주세요", "좋을 것 같", "어떨까요",
    "힘들어요", "안 들려", "안 보여", "읽기 어렵", "이해하기 어렵",
    "아쉽", "부족", "짧아", "길어", "단조롭",
    # 영어
    "please", "should", "could", "would be better", "too loud", "too small",
    "too large", "too long", "too short", "too fast", "too slow",
    "lower", "higher", "increase", "decrease", "improve", "fix", "change",
    "cannot hear", "hard to read", "difficult to", "suggest", "recommend",
]

_SPAM_KW = [
    "구독", "좋아요 눌러", "방문해주세요", "맞구독", "팔로우",
    "subscribe", "follow me", "check out my", "visit my",
    "http://", "https://", ".com", ".kr",
]

_INAPPROPRIATE_KW = [
    "저작권", "표절", "불법", "야한", "성인", "19금", "폭력", "혐오",
    "copyright", "plagiarism", "illegal", "nsfw", "explicit",
]

# 주제별 키워드 매핑
_TOPIC_KW: dict[str, list[str]] = {
    "audio":    ["음악", "배경음", "볼륨", "효과음", "music", "sound", "volume", "audio", "bgm"],
    "subtitle": ["자막", "글씨", "폰트", "subtitle", "caption", "font", "text"],
    "pacing":   ["속도", "빠르", "느리", "편집", "템포", "pace", "speed", "tempo", "edit"],
    "visual":   ["색", "조명", "화질", "화면", "bright", "dark", "color", "visual", "light"],
    "story":    ["스토리", "내용", "이야기", "캐릭터", "story", "plot", "character", "narrative"],
    "content":  ["정보", "설명", "내용", "구성", "information", "content", "explain"],
    "technical":["화질", "품질", "기술", "quality", "technical", "production"],
}


class RuleBasedClassifier(BaseClassifier):
    """[팀 직접 구현] 키워드 기반 폴백 분류기."""

    def classify(self, texts: list[str]) -> list[ClassificationResult]:
        return [self._classify_one(t) for t in texts]

    def _classify_one(self, text: str) -> ClassificationResult:
        lower = text.lower()

        # 부적절 우선
        if any(kw in lower for kw in _INAPPROPRIATE_KW):
            label = "inappropriate"
            confidence = 0.75
        elif any(kw in lower for kw in _SPAM_KW):
            label = "spam"
            confidence = 0.75
        elif any(kw in text for kw in _CONSTRUCTIVE_KW):
            label = "constructive"
            confidence = 0.65
        else:
            label = "non_constructive"
            confidence = 0.60

        topic = self._detect_topic(lower)
        scope = topic_to_scope(topic)
        spec = 0.70 if label == "constructive" else 0.20
        val = 0.60 if label == "constructive" else 0.20

        severity = 0.3 if label == "constructive" else 0.0
        urgency = 0.2 if label == "constructive" else 0.0
        quality = round(0.30 * spec + 0.25 * val + 0.15 * severity, 4)

        return ClassificationResult(
            constructive_label=label,
            topic_category=topic,
            scope=scope,
            specificity_score=spec,
            validity_score=val,
            confidence=confidence,
            detected_language=_detect_lang(text),
            improvement_suggestion=text if label == "constructive" else None,
            severity_score=severity,
            urgency_score=urgency,
            quality_score=quality,
            topic_categories_multi=[(topic, 1.0)] if topic != "other" else [("other", 0.0)],
            needs_review=False,
        )

    def _detect_topic(self, lower: str) -> str:
        for topic, keywords in _TOPIC_KW.items():
            if any(kw in lower for kw in keywords):
                return topic
        return "other"


def _detect_lang(text: str) -> str:
    ko = len(re.findall(r"[가-힣]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    if ko > 3:
        return "ko"
    if en > ko * 2:
        return "en"
    return "auto"
