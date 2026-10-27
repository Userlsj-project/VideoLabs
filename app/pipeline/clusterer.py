"""
Step 7: 피드백 클러스터링
건설적 피드백을 임베딩 기반으로 군집화하고 최다 언급 방향을 우선순위화한다.
상반된 피드백(예: "더 웃기게" vs "더 진지하게")이 있으면 최다 언급 방향을 선택한다.
"""

import json
import numpy as np
from dataclasses import dataclass, field


@dataclass
class FeedbackCluster:
    cluster_id: int
    label: str
    representative_text: str
    feedback_ids: list[int]
    feedback_texts: list[str]
    count: int
    priority_score: float  # 1.0이 가장 높은 우선순위


def cluster_feedbacks(
    feedback_texts: list[str],
    feedback_ids: list[int],
    n_clusters: int | None = None,
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> list[FeedbackCluster]:
    """
    건설적 피드백을 군집화한다.
    sentence-transformers가 없으면 TF-IDF 기반 폴백 사용.
    """
    if len(feedback_texts) < 2:
        if feedback_texts:
            return [
                FeedbackCluster(
                    cluster_id=0,
                    label="단독 피드백",
                    representative_text=feedback_texts[0],
                    feedback_ids=feedback_ids,
                    feedback_texts=feedback_texts,
                    count=1,
                    priority_score=1.0,
                )
            ]
        return []

    try:
        return _cluster_with_sentence_transformers(
            feedback_texts, feedback_ids, n_clusters, embedding_model
        )
    except ImportError:
        print("[Clusterer] sentence-transformers 없음, TF-IDF 폴백 사용")
        return _cluster_with_tfidf(feedback_texts, feedback_ids, n_clusters)


def _cluster_with_sentence_transformers(
    texts: list[str],
    ids: list[int],
    n_clusters: int | None,
    model_name: str,
) -> list[FeedbackCluster]:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)

    k = n_clusters or max(2, min(len(texts) // 2, 8))
    k = min(k, len(texts))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return _build_clusters(texts, ids, embeddings, labels, kmeans.cluster_centers_)


def _cluster_with_tfidf(
    texts: list[str],
    ids: list[int],
    n_clusters: int | None,
) -> list[FeedbackCluster]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts).toarray()

    k = n_clusters or max(2, min(len(texts) // 2, 8))
    k = min(k, len(texts))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return _build_clusters(texts, ids, np.array(X), labels, kmeans.cluster_centers_)


def _build_clusters(
    texts: list[str],
    ids: list[int],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> list[FeedbackCluster]:
    from sklearn.metrics.pairwise import cosine_similarity

    cluster_map: dict[int, list[tuple[int, str, np.ndarray]]] = {}
    for i, label in enumerate(labels):
        cluster_map.setdefault(label, []).append((ids[i], texts[i], embeddings[i]))

    clusters: list[FeedbackCluster] = []
    for cid, members in cluster_map.items():
        member_embs = np.array([m[2] for m in members])
        center = centers[cid].reshape(1, -1)
        sims = cosine_similarity(member_embs, center).flatten()
        best_idx = int(np.argmax(sims))

        member_texts = [m[1] for m in members]
        member_ids = [m[0] for m in members]
        label_text = _summarize_cluster_label(member_texts[:3])

        clusters.append(
            FeedbackCluster(
                cluster_id=cid,
                label=label_text,
                representative_text=members[best_idx][1],
                feedback_ids=member_ids,
                feedback_texts=member_texts,
                count=len(members),
                priority_score=0.0,
            )
        )

    # 최다 언급 순으로 우선순위 부여
    clusters.sort(key=lambda c: c.count, reverse=True)
    total = sum(c.count for c in clusters)
    for rank, c in enumerate(clusters):
        c.priority_score = round(c.count / total, 4)

    return clusters


def _summarize_cluster_label(sample_texts: list[str]) -> str:
    """짧은 대표 레이블 생성 (단순 키워드 추출)."""
    keywords = {
        "음악": "🎵 배경음악",
        "볼륨": "🔊 음량 조절",
        "자막": "📝 자막",
        "화질": "📹 화질",
        "편집": "✂️ 편집 속도",
        "속도": "⏩ 영상 속도",
        "조명": "💡 조명",
        "배경": "🖼️ 배경",
        "글씨": "🔤 폰트/글씨",
        "길이": "⏱️ 영상 길이",
        "인트로": "🎬 인트로",
        "목소리": "🎤 목소리/내레이션",
        "색": "🎨 색감",
        "밝": "☀️ 밝기",
        "내용": "📋 콘텐츠 내용",
    }
    combined = " ".join(sample_texts)
    for keyword, label in keywords.items():
        if keyword in combined:
            return label
    return "💬 기타 개선 의견"


def get_embeddings_for_storage(
    texts: list[str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> list[list[float]] | None:
    """DB 저장용 임베딩 벡터 생성."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    except ImportError:
        return None
