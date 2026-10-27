"""
[팀 직접 설계한 로직]
1차 항목 태깅: 건설적 피드백을 주제 카테고리로 그룹화.
클러스터링 전에 수행하는 전처리 단계.

카테고리 체계 (TOPIC_CATEGORIES in base.py):
  story, visual, audio, subtitle, pacing, content, technical, other

태깅 결과로 같은 카테고리 내에서만 임베딩 클러스터링을 수행한다.
→ "음악 볼륨"과 "자막 크기"가 의미적으로 비슷해도 다른 클러스터에 배치됨
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..classifier.base import TOPIC_CATEGORIES, topic_to_scope


@dataclass
class TaggedGroup:
    """단일 주제 카테고리에 속하는 피드백 묶음."""

    category: str                         # TOPIC_CATEGORIES 키
    category_label: str                   # 한국어 표시명
    scope: str                            # content | technical
    feedback_indices: list[int] = field(default_factory=list)  # 원본 배열 내 인덱스
    feedback_texts: list[str] = field(default_factory=list)


def group_by_topic(
    texts: list[str],
    topic_categories: list[str],
) -> list[TaggedGroup]:
    """
    [팀 직접 구현]
    texts[i]의 topic_categories[i]를 기반으로 같은 카테고리끼리 묶는다.

    Args:
        texts: 피드백 텍스트 목록
        topic_categories: 각 피드백의 주제 카테고리 (classifier 결과)

    Returns:
        TaggedGroup 목록 (비어있는 카테고리는 제외)
    """
    groups: dict[str, TaggedGroup] = {}

    for idx, (text, category) in enumerate(zip(texts, topic_categories)):
        if category not in groups:
            groups[category] = TaggedGroup(
                category=category,
                category_label=TOPIC_CATEGORIES.get(category, category),
                scope=topic_to_scope(category),
            )
        groups[category].feedback_indices.append(idx)
        groups[category].feedback_texts.append(text)

    # 언급 수 내림차순 정렬 (빈도 높은 카테고리부터 처리)
    return sorted(groups.values(), key=lambda g: len(g.feedback_indices), reverse=True)


def infer_scope_from_category(category: str) -> str:
    """카테고리 코드에서 scope(content/technical)를 반환."""
    return topic_to_scope(category)
