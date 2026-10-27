"""
[팀 직접 설계·구현] 2단계 피드백 클러스터링 파이프라인 (v2).

흐름:
  1. tagger.py: topic_category 기준으로 피드백을 그룹화
  2. embedder.py: 다국어 문장 임베딩 생성 (로컬 오픈소스 모델)
  3. hierarchical.py: 각 그룹 내 클러스터링 (HDBSCAN 우선 / AgglomerativeClustering 폴백)
     - 카테고리별 적응형 임계값
     - 품질 가중 대표 텍스트 선택
     - 응집도 점수 및 클러스터 요약 생성
     - 대형 클러스터 서브클러스터링 (>15개)
  4. ranker.py: 5축 우선순위 점수 산정 (빈도·품질·심각도·긴급도·타당성)
"""

from __future__ import annotations

import logging

import numpy as np

from .embedder import get_embeddings
from .hierarchical import RawCluster, cluster_group
from .ranker import assign_priorities
from .tagger import group_by_topic

logger = logging.getLogger(__name__)


def run_clustering(
    texts: list[str],
    topic_categories: list[str],
    validity_scores: list[float],
    original_indices: list[int],
    embedding_model: str,
    embedding_fallback: str,
    distance_threshold: float = 0.40,
    priority_alpha: float = 0.35,
    priority_beta: float = 0.25,
    priority_gamma: float = 0.20,
    priority_delta: float = 0.12,
    priority_epsilon: float = 0.08,
    severity_scores: list[float] | None = None,
    urgency_scores: list[float] | None = None,
    quality_scores: list[float] | None = None,
) -> list[RawCluster]:
    """
    [팀 직접 구현] 전체 클러스터링 파이프라인을 실행한다.

    Args:
        texts: 건설적 피드백 텍스트 목록
        topic_categories: 각 피드백의 주제 카테고리 (classifier 출력)
        validity_scores: 각 피드백의 타당성 점수
        original_indices: texts[i]가 DB Feedback 배열에서 몇 번째인지
        embedding_model: 임베딩 모델 이름
        embedding_fallback: 폴백 임베딩 모델 이름
        distance_threshold: 계층적 군집화 기본 임계값 (카테고리별로 오버라이드됨)
        priority_alpha~epsilon: 5축 우선순위 가중치
        severity_scores: 각 피드백의 심각도 점수 (v2 신규)
        urgency_scores: 각 피드백의 긴급도 점수 (v2 신규)
        quality_scores: 각 피드백의 품질 점수 (v2 신규)

    Returns:
        우선순위 내림차순 정렬된 RawCluster 목록 (DB 저장 전)
    """
    if not texts:
        return []

    # Step 1: 임베딩 생성 [오픈소스 모델 활용]
    logger.info("임베딩 생성 중 (%d개 텍스트, 모델: %s)", len(texts), embedding_model)
    embeddings = get_embeddings(texts, embedding_model, embedding_fallback)

    # Step 2: 카테고리별 그룹화 [팀 직접 구현]
    groups = group_by_topic(texts, topic_categories)
    logger.info(
        "카테고리 그룹: %s",
        [f"{g.category}({len(g.feedback_indices)})" for g in groups],
    )

    # 원본 인덱스 → 점수 딕셔너리
    quality_by_idx: dict[int, float] = (
        {orig: q for orig, q in zip(original_indices, quality_scores)}
        if quality_scores else {}
    )
    severity_by_idx: dict[int, float] = (
        {orig: s for orig, s in zip(original_indices, severity_scores)}
        if severity_scores else {}
    )
    urgency_by_idx: dict[int, float] = (
        {orig: u for orig, u in zip(original_indices, urgency_scores)}
        if urgency_scores else {}
    )

    # Step 3: 각 그룹 내 클러스터링 [팀 직접 구현 + 오픈소스 알고리즘]
    all_clusters: list[RawCluster] = []
    for group in groups:
        group_clusters = cluster_group(
            group,
            embeddings,
            distance_threshold=distance_threshold,
            quality_by_idx=quality_by_idx or None,
            severity_by_idx=severity_by_idx or None,
            urgency_by_idx=urgency_by_idx or None,
        )
        all_clusters.extend(group_clusters)
        logger.info("  %s → %d 클러스터", group.category, len(group_clusters))

    if not all_clusters:
        return []

    # Step 4: 5축 우선순위 점수 산정 [팀 직접 구현]
    validity_by_orig_idx = {
        orig_idx: val
        for orig_idx, val in zip(original_indices, validity_scores)
    }
    assign_priorities(
        all_clusters,
        validity_scores=validity_by_orig_idx,
        total_constructive=len(texts),
        alpha=priority_alpha,
        beta=priority_beta,
        gamma=priority_gamma,
        delta=priority_delta,
        epsilon=priority_epsilon,
    )

    logger.info("클러스터링 완료: 총 %d개 클러스터", len(all_clusters))
    return all_clusters
