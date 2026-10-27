"""
[팀 직접 설계·구현] 클러스터 우선순위 점수 산정 (v2).

우선순위 공식 (팀 설계 v2):
  P = α·freq_score + β·avg_quality + γ·avg_severity + δ·avg_urgency + ε·avg_validity

  freq_score    = 클러스터 내 피드백 수 / 전체 건설적 피드백 수
  avg_quality   = 피드백 품질 평균 (구체성·타당성·심각도·길이 종합)
  avg_severity  = 심각도 평균 (문제가 얼마나 치명적인가)
  avg_urgency   = 긴급도 평균 (즉각 처리가 필요한가)
  avg_validity  = 타당성 평균 (합리적 근거 유무)

기본 가중치: α=0.35, β=0.25, γ=0.20, δ=0.12, ε=0.08

v1 대비 변경:
  - 빈도 가중치 감소 (0.60 → 0.35): 빈도만으론 우선순위 결정 불충분
  - 품질 축 신설 (0.25): 피드백 자체의 구체성·타당성 종합
  - 심각도 축 신설 (0.20): 치명적 문제 상향 조정
  - 긴급도 축 신설 (0.12): 긴급 처리 요청 반영
  - 타당성 소폭 반영 (0.08): 근거 있는 피드백 미세 보정

상반된 클러스터(contradicted_by_key ≠ None)는 우선순위를 40% 감산.
"""

from __future__ import annotations

import numpy as np

from .hierarchical import RawCluster


def assign_priorities(
    clusters: list[RawCluster],
    validity_scores: dict[int, float],
    total_constructive: int,
    alpha: float = 0.35,
    beta: float = 0.25,
    gamma: float = 0.20,
    delta: float = 0.12,
    epsilon: float = 0.08,
) -> list[RawCluster]:
    """
    [팀 직접 구현]
    각 클러스터에 우선순위 점수를 부여하고 내림차순 정렬한 목록을 반환.

    Args:
        clusters: RawCluster 목록 (avg_severity/urgency/quality 필드 이미 설정됨)
        validity_scores: {피드백 원본 인덱스: validity_score}
        total_constructive: 전체 건설적 피드백 수 (정규화 기준)
        alpha: 빈도 가중치
        beta:  품질 가중치
        gamma: 심각도 가중치
        delta: 긴급도 가중치
        epsilon: 타당성 가중치
    """
    if not clusters or total_constructive == 0:
        return clusters

    contradiction_keys = {c.contradicted_by_key for c in clusters if c.contradicted_by_key}

    for c in clusters:
        freq_score = c.count / total_constructive

        # 타당성: validity_scores 딕셔너리에서 클러스터 멤버 평균
        v_list = [validity_scores.get(i, 0.5) for i in c.feedback_indices]
        avg_validity = float(np.mean(v_list)) if v_list else 0.5

        # 품질·심각도·긴급도: RawCluster에 이미 계산된 필드 활용
        avg_quality = c.avg_quality_score
        avg_severity = c.avg_severity_score
        avg_urgency = c.avg_urgency_score

        raw_priority = (
            alpha * freq_score
            + beta * avg_quality
            + gamma * avg_severity
            + delta * avg_urgency
            + epsilon * avg_validity
        )

        # 상반된 피드백 페널티 (-40%)
        if c.cluster_key in contradiction_keys:
            raw_priority *= 0.60

        # 우선순위 점수 및 보조 지표 저장
        c.__dict__["priority_score"] = round(float(raw_priority), 4)
        c.__dict__["freq_score"] = round(float(freq_score), 4)
        c.__dict__["avg_validity_score"] = round(float(avg_validity), 4)

    clusters.sort(key=lambda c: c.__dict__.get("priority_score", 0), reverse=True)
    return clusters
