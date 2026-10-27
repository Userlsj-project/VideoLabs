"""
[오픈소스 모델 활용]
제로샷 NLI 기반 다국어 피드백 분류기.

사용 모델:
  MoritzLaurer/mDeBERTa-v3-base-zeroshot-v2.0
  - 라이선스: MIT
  - 파라미터: ~280M (DeBERTa-v3-base 기반)
  - CPU 실행 가능 (GPU 없이 팀 노트북에서 동작)
  - 지원 언어: 100+ (다국어 NLI)
  - https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-zeroshot-v2.0

[팀 직접 설계한 로직]
- 두 축 판별 기준 (구체성·타당성) 및 가설 레이블 설계
- 주제 항목 분류 체계 (8개 카테고리)
- 언어 무관 프롬프트: 입력 언어에 상관없이 동일한 영어 가설 레이블 적용
  → 다국어 NLI 모델이 한국어·영어·기타 입력과 영어 가설 간 함의 관계를 추론
- 배치 처리 및 모듈 레벨 캐싱
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from .base import (
    TOPIC_CATEGORIES,
    BaseClassifier,
    ClassificationResult,
    topic_to_scope,
)

logger = logging.getLogger(__name__)

# ── 모델 캐시 ─────────────────────────────────────────────────────────────
_pipeline_cache: dict[str, Any] = {}


def _get_zs_pipeline(model_name: str) -> Any:
    """[팀 직접 구현] 제로샷 파이프라인을 캐시해 재사용."""
    if model_name not in _pipeline_cache:
        from transformers import pipeline as hf_pipeline  # [오픈소스]

        logger.info("제로샷 분류 모델 로드 중: %s (최초 1회)", model_name)
        _pipeline_cache[model_name] = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU 고정
        )
    return _pipeline_cache[model_name]


# ── [팀 직접 설계] 가설 레이블 ─────────────────────────────────────────────
# 입력 텍스트가 어떤 언어든 영어 가설과 대조해 NLI 추론
# → 별도 언어별 학습 데이터 없이 다국어 분류 가능

# Pass 1: 건설적 여부 4-way
_CONSTRUCTIVE_LABELS = [
    "contains a specific and actionable improvement suggestion for the video",  # constructive
    "expresses a general impression or emotion without a specific suggestion",   # non_constructive
    "is spam, self-promotion, or unrelated to the video content",               # spam
    "requests inappropriate, harmful, or copyright-infringing content",         # inappropriate
]
_CONSTRUCTIVE_MAP = {
    _CONSTRUCTIVE_LABELS[0]: "constructive",
    _CONSTRUCTIVE_LABELS[1]: "non_constructive",
    _CONSTRUCTIVE_LABELS[2]: "spam",
    _CONSTRUCTIVE_LABELS[3]: "inappropriate",
}

# Pass 2: 주제 항목 8-way
# [팀 직접 설계] 영상 피드백에 특화된 분류 체계
_TOPIC_LABELS: dict[str, str] = {
    "story":     "video story, plot, narrative, or character development",
    "visual":    "visual aesthetics, colors, lighting, or cinematography",
    "audio":     "background music, sound effects, volume level, or audio quality",
    "subtitle":  "subtitles, captions, text font size, or text readability",
    "pacing":    "editing speed, video pacing, rhythm, or tempo",
    "content":   "information content, educational value, or topic coverage",
    "technical": "overall technical quality, production value, or equipment",
    "other":     "other aspects not covered above",
}
_TOPIC_LABEL_LIST = list(_TOPIC_LABELS.values())
_TOPIC_KEY_LIST = list(_TOPIC_LABELS.keys())

# Pass 3: 두 축 (구체성·타당성) — multi_label=True
_SPECIFICITY_LABEL = "contains a specific and actionable improvement suggestion"
_VAGUE_LABEL = "is a vague general comment without actionable advice"
_VALIDITY_LABEL = "provides logical reasoning, concrete examples, or evidence"
_EMOTION_LABEL = "expresses emotion or opinion without logical justification"


class ZeroShotClassifier(BaseClassifier):
    """
    [팀 직접 설계·구현]
    제로샷 NLI 모델을 사용한 다국어 피드백 분류기.
    모델 이름은 설정값으로 주입받아 교체 가능.
    """

    def __init__(
        self,
        model_name: str,
        specificity_threshold: float = 0.45,
        validity_threshold: float = 0.40,
    ) -> None:
        self._model_name = model_name
        self._spec_thr = specificity_threshold
        self._val_thr = validity_threshold

    # ── 공개 API ────────────────────────────────────────────────────────────

    def classify(self, texts: list[str]) -> list[ClassificationResult]:
        pipe = _get_zs_pipeline(self._model_name)
        results: list[ClassificationResult] = []

        for text in texts:
            try:
                result = self._classify_one(pipe, text)
            except Exception as e:
                logger.warning("제로샷 분류 실패 (%s…): %s", text[:30], e)
                result = _fallback_result(text)
            results.append(result)

        return results

    # ── 내부 로직 ────────────────────────────────────────────────────────────

    def _classify_one(self, pipe: Any, text: str) -> ClassificationResult:
        # [오픈소스 API] transformers zero-shot-classification pipeline
        # Pass 1: 건설적 여부
        c_out = pipe(text, candidate_labels=_CONSTRUCTIVE_LABELS, multi_label=False)
        top_label = c_out["labels"][0]
        confidence = float(c_out["scores"][0])
        constructive_label = _CONSTRUCTIVE_MAP[top_label]

        # Pass 2: 주제 항목
        t_out = pipe(text, candidate_labels=_TOPIC_LABEL_LIST, multi_label=False)
        topic_idx = _TOPIC_LABEL_LIST.index(t_out["labels"][0])
        topic_category = _TOPIC_KEY_LIST[topic_idx]
        scope = topic_to_scope(topic_category)

        # Pass 3: 두 축 점수 (모든 피드백에 대해 계산)
        sv_out = pipe(
            text,
            candidate_labels=[_SPECIFICITY_LABEL, _VAGUE_LABEL,
                               _VALIDITY_LABEL, _EMOTION_LABEL],
            multi_label=True,
        )
        score_map = dict(zip(sv_out["labels"], sv_out["scores"]))
        specificity_score = float(score_map.get(_SPECIFICITY_LABEL, 0.5))
        validity_score = float(score_map.get(_VALIDITY_LABEL, 0.5))

        # [팀 직접 설계] 두 축 기반 constructive 재확인
        # 모델이 constructive로 분류해도 두 축이 모두 미달이면 non_constructive로 보정
        if (
            constructive_label == "constructive"
            and specificity_score < self._spec_thr
            and validity_score < self._val_thr
        ):
            constructive_label = "non_constructive"
            confidence = min(confidence, 0.65)

        # 개선 제안 추출 (constructive인 경우)
        suggestion = _extract_suggestion(text) if constructive_label == "constructive" else None

        return ClassificationResult(
            constructive_label=constructive_label,
            topic_category=topic_category,
            scope=scope,
            specificity_score=round(specificity_score, 4),
            validity_score=round(validity_score, 4),
            confidence=round(confidence, 4),
            detected_language=_detect_language(text),
            improvement_suggestion=suggestion,
        )


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """
    [팀 직접 구현] 간단한 한국어/영어 판별.
    외부 라이브러리 없이 유니코드 범위로 판별.
    """
    korean_chars = len(re.findall(r"[가-힣]", text))
    ascii_alpha = len(re.findall(r"[a-zA-Z]", text))
    if korean_chars > 3:
        return "ko"
    if ascii_alpha > korean_chars * 2:
        return "en"
    return "auto"


def _extract_suggestion(text: str) -> str | None:
    """[팀 직접 구현] 개선 제안 문장 추출 (첫 번째 문장을 대표 텍스트로)."""
    sentences = re.split(r"[.!?。]\s*", text.strip())
    for s in sentences:
        if len(s) > 10:
            return s.strip()
    return text.strip() if text.strip() else None


def _fallback_result(text: str) -> ClassificationResult:
    """분류 실패 시 기본값 반환."""
    return ClassificationResult(
        constructive_label="non_constructive",
        topic_category="other",
        scope="content",
        specificity_score=0.0,
        validity_score=0.0,
        confidence=0.0,
        detected_language=_detect_language(text),
        improvement_suggestion=None,
    )
