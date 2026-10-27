"""
[팀 직접 설계·구현] 2단계 계층적 클러스터링 파이프라인 (v2).

단계:
  1. 카테고리별 그룹화 (tagger.py에서 수행)
  2. 각 그룹 내 클러스터링 (HDBSCAN 우선, 폴백 AgglomerativeClustering)
  3. 대형 클러스터 서브클러스터링 (>15개)
  4. 품질 가중 대표 텍스트 선택
  5. 응집도(cohesion) 점수 산정
  6. 클러스터 요약 생성 (단어 빈도 기반 추출)

[오픈소스 활용]
  sklearn.cluster.AgglomerativeClustering — 메인 클러스터링 알고리즘
  hdbscan.HDBSCAN — 밀도 기반 클러스터링 (설치된 경우 우선 사용)

[팀 직접 설계]
  - 카테고리별 적응형 distance_threshold
  - 품질·심각도 가중 대표 텍스트 선택
  - 응집도 점수 (평균 쌍별 코사인 유사도)
  - 단어 빈도 기반 클러스터 요약
  - 상반된 피드백 감지 (키워드 쌍 매칭)
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

import numpy as np

from .tagger import TaggedGroup

logger = logging.getLogger(__name__)

# ── [팀 직접 설계] 카테고리별 적응형 거리 임계값 ──────────────────────────────
# 각 카테고리는 표현 다양성이 다르므로 최적 임계값도 다름.

_CATEGORY_THRESHOLDS: dict[str, float] = {
    "audio":     0.38,   # 오디오 피드백은 표현 다양 → 조금 넓게
    "subtitle":  0.30,   # 자막은 구체적 표현 → 좁게
    "pacing":    0.42,   # 속도/편집은 표현 다양
    "visual":    0.36,   # 색감/조명 표현은 중간
    "story":     0.48,   # 스토리는 매우 다양한 표현
    "content":   0.44,   # 내용 구성은 다양
    "technical": 0.26,   # 기술 이슈는 매우 구체적 → 가장 좁게
    "other":     0.40,   # 기타
}

# 서브클러스터링 임계값 = 기본 임계값 * 이 배율
_SUBCLUSTER_THRESHOLD_RATIO = 0.58

# 서브클러스터링 트리거 크기
_MAX_CLUSTER_SIZE = 15

# ── [팀 직접 설계] 상반된 피드백 키워드 쌍 ──────────────────────────────────
_CONTRADICTION_PAIRS: list[tuple[list[str], list[str]]] = [
    (
        ["빠르게", "빨리", "짧게", "faster", "shorter", "speed up"],
        ["느리게", "천천히", "길게", "slower", "longer", "slow down"],
    ),
    (
        ["밝게", "환하게", "brighter", "lighter"],
        ["어둡게", "어두운", "darker"],
    ),
    (
        ["크게", "높여", "louder", "increase volume", "turn up"],
        ["작게", "줄여", "조용히", "quieter", "lower volume", "turn down"],
    ),
    (
        ["웃기게", "재미있게", "유머", "funny", "humor", "lighthearted"],
        ["진지하게", "심각하게", "serious", "formal"],
    ),
    (
        ["더 많이", "늘려", "add more", "more content"],
        ["줄여", "빼", "remove", "less", "fewer", "shorter"],
    ),
    (
        ["화려하게", "colorful", "vibrant"],
        ["단순하게", "심플", "minimalistic", "simple"],
    ),
]

# 스톱워드 (클러스터 요약 생성 시 제외)
_STOPWORDS: frozenset[str] = frozenset({
    "있다", "없다", "것", "하는", "입니다", "합니다", "있어요", "해요", "있는",
    "이", "그", "저", "제", "는", "을", "를", "가", "으로", "와", "과",
    "그리고", "하지만", "그런데", "이렇게", "저는", "제가", "그게", "이게",
    "the", "is", "are", "and", "or", "to", "a", "an", "in", "of", "it",
    "this", "that", "for", "but", "with", "on", "at", "by", "from",
})


@dataclass
class RawCluster:
    """클러스터링 중간 결과 (DB 저장 전 순수 Python 객체)."""

    cluster_key: str
    category: str
    scope: str
    label: str
    representative_text: str
    feedback_indices: list[int]
    feedback_texts: list[str]
    count: int
    avg_similarity: float
    contradicted_by_key: str | None = None

    # v2 신규 필드
    cohesion_score: float = 0.0           # 클러스터 내부 응집도 (0~1)
    cluster_summary: str = ""             # 추출적 요약 문장
    avg_severity_score: float = 0.3       # 클러스터 내 평균 심각도
    avg_urgency_score: float = 0.2        # 클러스터 내 평균 긴급도
    avg_quality_score: float = 0.5        # 클러스터 내 평균 피드백 품질


def cluster_group(
    group: TaggedGroup,
    embeddings: np.ndarray,
    distance_threshold: float = 0.40,
    quality_by_idx: dict[int, float] | None = None,
    severity_by_idx: dict[int, float] | None = None,
    urgency_by_idx: dict[int, float] | None = None,
) -> list[RawCluster]:
    """
    [팀 직접 구현]
    단일 TaggedGroup(같은 카테고리) 내에서 계층적 클러스터링 수행.

    Args:
        group: 같은 카테고리로 묶인 피드백 그룹
        embeddings: 전체 피드백 임베딩 배열
        distance_threshold: 기본 거리 임계값 (카테고리별로 오버라이드됨)
        quality_by_idx: {원본_인덱스: quality_score}
        severity_by_idx: {원본_인덱스: severity_score}
        urgency_by_idx: {원본_인덱스: urgency_score}

    Returns:
        RawCluster 목록
    """
    n = len(group.feedback_indices)
    if n == 0:
        return []

    group_embs = embeddings[group.feedback_indices]  # (n, dim)

    if n == 1:
        orig_idx = group.feedback_indices[0]
        text = group.feedback_texts[0]
        return [_build_single_cluster(
            group, group_embs, [0], [text], [orig_idx],
            quality_by_idx, severity_by_idx, urgency_by_idx,
        )]

    # 카테고리별 적응형 임계값
    threshold = _CATEGORY_THRESHOLDS.get(group.category, distance_threshold)

    # 클러스터 레이블 생성
    labels = _cluster_embeddings(group_embs, group.category, threshold)

    # 레이블별로 인덱스 묶기
    label_groups: dict[int, list[int]] = {}
    for local_i, lbl in enumerate(labels):
        label_groups.setdefault(int(lbl), []).append(local_i)

    clusters: list[RawCluster] = []

    for lbl, local_indices in label_groups.items():
        cluster_embs = group_embs[local_indices]
        texts = [group.feedback_texts[i] for i in local_indices]
        orig_indices = [group.feedback_indices[i] for i in local_indices]

        # 대형 클러스터 서브클러스터링
        if len(local_indices) > _MAX_CLUSTER_SIZE:
            sub_threshold = threshold * _SUBCLUSTER_THRESHOLD_RATIO
            sub_labels = _agglomerative(cluster_embs, sub_threshold)
            unique_subs = np.unique(sub_labels)

            # 2개 이상 서브클러스터이고 모두 2개 이상 멤버여야 분할
            valid_split = len(unique_subs) >= 2 and all(
                (sub_labels == sl).sum() >= 2 for sl in unique_subs
            )

            if valid_split:
                for sub_lbl in unique_subs:
                    sub_mask = sub_labels == sub_lbl
                    sub_local = [local_indices[i] for i in range(len(local_indices)) if sub_mask[i]]
                    sub_texts = [group.feedback_texts[i] for i in sub_local]
                    sub_orig = [group.feedback_indices[i] for i in sub_local]
                    clusters.append(_build_single_cluster(
                        group, group_embs, sub_local, sub_texts, sub_orig,
                        quality_by_idx, severity_by_idx, urgency_by_idx,
                    ))
                continue  # 서브클러스터로 대체 — 원본 건너뜀

        clusters.append(_build_single_cluster(
            group, group_embs, local_indices, texts, orig_indices,
            quality_by_idx, severity_by_idx, urgency_by_idx,
        ))

    # 상반된 피드백 감지
    _mark_contradictions(clusters)
    return clusters


def _build_single_cluster(
    group: TaggedGroup,
    group_embs: np.ndarray,
    local_indices: list[int],
    texts: list[str],
    orig_indices: list[int],
    quality_by_idx: dict[int, float] | None,
    severity_by_idx: dict[int, float] | None,
    urgency_by_idx: dict[int, float] | None,
) -> RawCluster:
    """단일 클러스터 RawCluster 빌드 헬퍼."""
    cluster_embs = group_embs[local_indices]

    # 클러스터 내 평균 코사인 유사도
    centroid = cluster_embs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = cluster_embs @ centroid_norm
    avg_sim = float(np.mean(sims))

    # 품질·심각도·긴급도 점수 수집
    q_scores = [quality_by_idx.get(i, 0.5) for i in orig_indices] if quality_by_idx else [0.5] * len(orig_indices)
    s_scores = [severity_by_idx.get(i, 0.3) for i in orig_indices] if severity_by_idx else [0.3] * len(orig_indices)
    u_scores = [urgency_by_idx.get(i, 0.2) for i in orig_indices] if urgency_by_idx else [0.2] * len(orig_indices)

    # 품질 가중 대표 텍스트 선택
    best_idx = _select_representative(cluster_embs, texts, q_scores, sims)

    # 응집도 점수
    cohesion = _compute_cohesion(cluster_embs)

    # 클러스터 요약
    summary = _generate_cluster_summary(texts)

    return RawCluster(
        cluster_key=str(uuid.uuid4()),
        category=group.category,
        scope=group.scope,
        label=_auto_label(group.category, texts),
        representative_text=texts[best_idx],
        feedback_indices=orig_indices,
        feedback_texts=texts,
        count=len(texts),
        avg_similarity=round(avg_sim, 4),
        cohesion_score=cohesion,
        cluster_summary=summary,
        avg_severity_score=round(float(np.mean(s_scores)), 4),
        avg_urgency_score=round(float(np.mean(u_scores)), 4),
        avg_quality_score=round(float(np.mean(q_scores)), 4),
    )


# ── [팀 직접 설계] 클러스터링 알고리즘 ────────────────────────────────────────

def _cluster_embeddings(
    embs: np.ndarray,
    category: str,
    threshold: float,
) -> np.ndarray:
    """
    [팀 직접 구현] HDBSCAN (설치된 경우) 또는 AgglomerativeClustering.

    HDBSCAN: 밀도 기반, 노이즈 자동 감지, 임계값 불필요 — 더 자연스러운 군집
    AgglomerativeClustering: 계층적, 임계값 필요 — 설치 없이 동작하는 안정적 폴백
    """
    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)

    if np.allclose(embs, embs[0]):
        return np.zeros(n, dtype=int)

    # HDBSCAN 시도 (설치된 경우)
    try:
        import hdbscan as _hdbscan

        min_cs = max(2, min(3, n // 4))
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=1,
            metric="euclidean",   # L2-정규화 벡터에서 유클리드 ≈ 코사인
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.20,
        )
        labels = clusterer.fit_predict(embs)

        # 노이즈(-1)를 가장 가까운 클러스터로 재배정
        noise_mask = labels == -1
        if noise_mask.any() and (~noise_mask).any():
            valid_lbls = np.unique(labels[~noise_mask])
            for i in np.where(noise_mask)[0]:
                best_lbl = valid_lbls[0]
                best_dist = float("inf")
                for lbl in valid_lbls:
                    centroid = embs[labels == lbl].mean(axis=0)
                    dist = float(np.linalg.norm(embs[i] - centroid))
                    if dist < best_dist:
                        best_dist = dist
                        best_lbl = lbl
                labels[i] = best_lbl

        if (labels < 0).all():
            labels = np.zeros(n, dtype=int)

        logger.debug("HDBSCAN 클러스터링: %s → %d 클러스터", category, len(np.unique(labels)))
        return labels.astype(int)

    except ImportError:
        pass
    except Exception as e:
        logger.debug("HDBSCAN 실패 (%s) → AgglomerativeClustering 폴백", e)

    # AgglomerativeClustering 폴백 [오픈소스]
    return _agglomerative(embs, threshold)


def _agglomerative(embs: np.ndarray, distance_threshold: float) -> np.ndarray:
    """[오픈소스 활용] sklearn AgglomerativeClustering."""
    from sklearn.cluster import AgglomerativeClustering  # [오픈소스]

    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)

    if np.allclose(embs, embs[0]):
        return np.zeros(n, dtype=int)

    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    return model.fit_predict(embs)


# ── [팀 직접 설계] 대표 텍스트 선택 ─────────────────────────────────────────

def _select_representative(
    cluster_embs: np.ndarray,
    texts: list[str],
    quality_scores: list[float],
    centrality_sims: np.ndarray,
) -> int:
    """
    [팀 직접 구현] 품질·중심 근접도·길이를 결합한 대표 텍스트 선택.

    단순 centroid 최근접 방식에서 개선:
      - 품질 점수 30% 반영 → 더 구체적·타당한 텍스트가 대표로 선택됨
      - 길이 점수 20% 반영 → 너무 짧은 텍스트 배제
      - 중심 근접도 50% 반영 → 클러스터를 잘 대표하는 텍스트 우선
    """
    n = len(texts)
    if n == 1:
        return 0

    q_arr = np.array(quality_scores, dtype=float)
    # 품질 점수 정규화 (클러스터 내 상대 비교)
    if q_arr.max() > q_arr.min():
        q_norm = (q_arr - q_arr.min()) / (q_arr.max() - q_arr.min())
    else:
        q_norm = np.ones(n)

    # 길이 점수 (20~200자 범위 최적, 너무 짧거나 너무 긴 것 페널티)
    length_scores = np.array([
        min(len(t) / 60, 1.0) * max(0.3, 1.0 - max(len(t) - 150, 0) / 300)
        for t in texts
    ])

    # 중심 근접도를 [0, 1] 정규화
    sim_norm = (centrality_sims - centrality_sims.min()) / (
        centrality_sims.max() - centrality_sims.min() + 1e-9
    )

    final_scores = 0.50 * sim_norm + 0.30 * q_norm + 0.20 * length_scores
    return int(np.argmax(final_scores))


# ── [팀 직접 설계] 응집도 점수 ────────────────────────────────────────────────

def _compute_cohesion(cluster_embs: np.ndarray) -> float:
    """
    [팀 직접 구현] 클러스터 응집도 (0~1, 높을수록 좋음).

    평균 쌍별 코사인 유사도 기반.
    낮은 응집도 = 노이즈가 많거나 분할이 필요한 상태임을 의미.
    """
    n = len(cluster_embs)
    if n < 2:
        return 1.0

    # L2-정규화된 벡터에서 dot product = cosine similarity
    sim_matrix = cluster_embs @ cluster_embs.T
    mask = ~np.eye(n, dtype=bool)
    avg_sim = float(sim_matrix[mask].mean())

    # [-1, 1] → [0, 1] 변환
    cohesion = (avg_sim + 1) / 2
    return round(float(np.clip(cohesion, 0.0, 1.0)), 4)


# ── [팀 직접 설계] 클러스터 요약 생성 ───────────────────────────────────────

def _generate_cluster_summary(texts: list[str], max_len: int = 120) -> str:
    """
    [팀 직접 구현] 단어 빈도 기반 추출 요약 (외부 의존성 없음).

    전체 텍스트에서 공통적으로 등장하는 핵심 단어를 많이 포함한
    문장을 대표 요약으로 선택.
    """
    if len(texts) == 1:
        return texts[0][:max_len]

    # 단어 빈도 계산
    word_freq: dict[str, int] = {}
    for text in texts:
        words = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", text.lower())
        for word in words:
            if word not in _STOPWORDS:
                word_freq[word] = word_freq.get(word, 0) + 1

    if not word_freq:
        return texts[0][:max_len]

    # 문장별 점수 계산
    all_sentences: list[tuple[str, float]] = []
    for text in texts:
        for sent in re.split(r"[.!?\n。]+", text):
            sent = sent.strip()
            if len(sent) < 8:
                continue
            words = re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", sent.lower())
            if not words:
                continue
            relevant = [w for w in words if w not in _STOPWORDS]
            if not relevant:
                continue
            score = sum(word_freq.get(w, 0) for w in relevant) / len(relevant)
            all_sentences.append((sent, score))

    if not all_sentences:
        return texts[0][:max_len]

    best = max(all_sentences, key=lambda x: x[1])
    return best[0][:max_len]


# ── [팀 직접 설계] 자동 레이블 생성 ─────────────────────────────────────────

_CATEGORY_EMOJI = {
    "story":     "📖",
    "visual":    "🎨",
    "audio":     "🎵",
    "subtitle":  "📝",
    "pacing":    "⏩",
    "content":   "📋",
    "technical": "🔧",
    "other":     "💬",
}

_KW_LABELS: list[tuple[list[str], str]] = [
    (["볼륨", "음량", "volume", "loud"],                     "볼륨 조절"),
    (["음악", "배경음", "bgm", "music"],                    "배경음악"),
    (["효과음", "sound effect"],                             "효과음"),
    (["목소리", "나레이션", "voice", "narration"],           "목소리/나레이션"),
    (["자막", "subtitle", "caption"],                        "자막"),
    (["글씨", "폰트", "font", "text size"],                 "글씨 크기/폰트"),
    (["가독성", "readability", "읽기"],                      "가독성"),
    (["속도", "빠르", "느리", "speed", "fast", "slow"],     "편집 속도"),
    (["템포", "리듬", "tempo", "rhythm"],                   "리듬감"),
    (["인트로", "intro", "시작"],                            "인트로"),
    (["색", "color", "colour"],                              "색감"),
    (["조명", "lighting", "밝기"],                           "조명/밝기"),
    (["화질", "해상도", "quality", "resolution"],            "화질/해상도"),
    (["스토리", "이야기", "story", "plot"],                  "스토리"),
    (["캐릭터", "인물", "character"],                        "캐릭터"),
    (["내용", "정보", "content", "information"],             "내용 구성"),
    (["길이", "duration", "짧", "길"],                      "영상 길이"),
    (["전환", "transition", "컷", "cut"],                   "장면 전환"),
    (["구도", "구성", "composition", "framing"],             "구도/프레이밍"),
]


def _auto_label(category: str, texts: list[str]) -> str:
    combined = " ".join(texts).lower()
    emoji = _CATEGORY_EMOJI.get(category, "💬")
    for keywords, label in _KW_LABELS:
        if any(kw in combined for kw in keywords):
            return f"{emoji} {label}"
    from ..classifier.base import TOPIC_CATEGORIES
    return f"{emoji} {TOPIC_CATEGORIES.get(category, category)}"


# ── [팀 직접 설계] 상반된 피드백 감지 ────────────────────────────────────────

def _mark_contradictions(clusters: list[RawCluster]) -> None:
    """동일 카테고리 내 상반된 클러스터 쌍을 감지해 작은 쪽에 표시."""
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            if i >= j:
                continue
            if _are_contradicted(ca.feedback_texts, cb.feedback_texts):
                if ca.count <= cb.count:
                    ca.contradicted_by_key = cb.cluster_key
                else:
                    cb.contradicted_by_key = ca.cluster_key


def _are_contradicted(texts_a: list[str], texts_b: list[str]) -> bool:
    combined_a = " ".join(texts_a).lower()
    combined_b = " ".join(texts_b).lower()
    for pos_kws, neg_kws in _CONTRADICTION_PAIRS:
        a_pos = any(k in combined_a for k in pos_kws)
        b_neg = any(k in combined_b for k in neg_kws)
        a_neg = any(k in combined_a for k in neg_kws)
        b_pos = any(k in combined_b for k in pos_kws)
        if (a_pos and b_neg) or (a_neg and b_pos):
            return True
    return False
