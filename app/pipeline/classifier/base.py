"""
[팀 직접 설계한 인터페이스]
피드백 분류기 추상 베이스 클래스.
구체성(specificity)·타당성(validity) 두 축으로 건설적 피드백을 판별하고
주제 항목(topic_category)을 태깅한다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── 주제 항목 분류 체계 ────────────────────────────────────────────────────
# [팀 직접 설계]  스코프: content=프로젝트 한정, technical=전역 적용 가능

TOPIC_CATEGORIES: dict[str, str] = {
    "story":     "스토리/내러티브/캐릭터",
    "visual":    "영상미/색감/조명/구도",
    "audio":     "음악/효과음/볼륨",
    "subtitle":  "자막/폰트/가독성",
    "pacing":    "속도/템포/편집 리듬",
    "content":   "내용/정보/구성",
    "technical": "기타 기술적 요소",
    "other":     "기타",
}

# 기술적 항목 → 전역 설정 등록 가능
TECHNICAL_SCOPES: frozenset[str] = frozenset({"audio", "subtitle", "pacing", "technical"})
CONTENT_SCOPES: frozenset[str] = frozenset({"story", "visual", "content", "other"})


def topic_to_scope(topic: str) -> str:
    return "technical" if topic in TECHNICAL_SCOPES else "content"


@dataclass
class ClassificationResult:
    """단일 피드백 분류 결과."""

    # 건설적 여부: constructive | non_constructive | spam | inappropriate
    constructive_label: str

    # 주제 항목 (TOPIC_CATEGORIES 키) — 1차 카테고리
    topic_category: str

    # 항목 스코프: content(프로젝트 한정) | technical(전역 적용 가능)
    scope: str

    # 두 축 점수 (0~1)
    specificity_score: float   # 구체성: 실행 가능한 개선 제안이 있는가
    validity_score: float      # 타당성: 합리적 근거가 있는가

    # 종합 신뢰도
    confidence: float

    # 감지된 언어 (ISO 639-1, 'auto'=불명확, 'mixed_ko_en' 등 혼합 가능)
    detected_language: str = "auto"

    # 추출된 개선 제안 텍스트
    improvement_suggestion: str | None = None

    # ── [팀 직접 설계] 확장 분류 축 ────────────────────────────────────────

    # 심각도: 문제가 얼마나 치명적인가 (0=경미, 1=시청 불가 수준)
    severity_score: float = 0.0

    # 긴급도: 즉각 처리가 필요한가 (0=여유, 1=즉시 처리)
    urgency_score: float = 0.0

    # 피드백 품질: 구체성·타당성·심각도·길이를 종합한 전반적 품질 (0~1)
    quality_score: float = 0.0

    # 멀티레이블 토픽: [(category, score), ...] 내림차순 정렬 (최대 3개)
    # 하나의 댓글이 여러 카테고리에 걸칠 때를 위한 확장 태그
    topic_categories_multi: list = field(default_factory=list)

    # 분류 신뢰도가 낮아 수동 검토가 권장되는 경우 True
    needs_review: bool = False


class BaseClassifier(ABC):
    """
    [팀 직접 설계한 인터페이스]
    어떤 분류기 구현체든 이 인터페이스를 따라야 한다.
    특정 모델에 종속되지 않도록 교체 가능하게 설계.
    """

    @abstractmethod
    def classify(self, texts: list[str]) -> list[ClassificationResult]:
        """
        텍스트 목록을 분류한다.
        - 언어 무관(다국어) 처리
        - 두 축(구체성·타당성) 점수 포함
        - 주제 항목 태깅 포함
        """

    def classify_single(self, text: str) -> ClassificationResult:
        return self.classify([text])[0]
