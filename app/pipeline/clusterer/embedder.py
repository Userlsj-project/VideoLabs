"""
[오픈소스 모델 활용]
다국어 문장 임베딩 생성 모듈.

사용 모델 (우선순위 순):
  1. paraphrase-multilingual-mpnet-base-v2  (Apache 2.0, ~278M 파라미터, 50+ 언어)
     https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  2. paraphrase-multilingual-MiniLM-L12-v2  (Apache 2.0, ~118M 파라미터, 폴백)

[팀 직접 설계]
- 모델 이름을 설정값으로 분리하여 교체 가능
- 모듈 레벨 캐싱으로 반복 로드 방지
- 폴백 체인: 설정 모델 → 경량 모델 → TF-IDF 벡터화
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# [팀 직접 구현] 모델 캐시 (프로세스당 한 번만 로드)
_model_cache: dict[str, object] = {}


def get_embeddings(
    texts: list[str],
    model_name: str,
    fallback_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """
    텍스트 목록을 임베딩 벡터로 변환한다.

    반환: shape (n, dim), L2-정규화된 단위 벡터
    폴백: sentence-transformers 실패 → TF-IDF
    """
    # ── 1차: 설정 모델 ───────────────────────────────────────────────────────
    try:
        model = _load_sentence_transformer(model_name)
        # [오픈소스 API 호출] SentenceTransformer.encode()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        logger.warning("임베딩 모델 '%s' 로드 실패: %s", model_name, e)

    # ── 2차: 경량 폴백 모델 ──────────────────────────────────────────────────
    if fallback_model and fallback_model != model_name:
        try:
            model = _load_sentence_transformer(fallback_model)
            logger.info("폴백 임베딩 모델 사용: %s", fallback_model)
            return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as e:
            logger.warning("폴백 임베딩 모델 실패: %s", e)

    # ── 3차: TF-IDF 벡터화 (다운로드 없이 동작) ──────────────────────────────
    logger.warning("TF-IDF 폴백 임베딩 사용 (정확도 낮음)")
    return _tfidf_embeddings(texts)


def _load_sentence_transformer(model_name: str):
    """[팀 직접 구현] 캐시된 SentenceTransformer 반환."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # [오픈소스]
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def clear_model_cache() -> None:
    """클러스터링 완료 후 ML 모델 캐시를 해제해 메모리를 반환한다.
    영상 재생성처럼 ML 모델이 불필요한 단계 직전에 호출한다."""
    import gc
    _model_cache.clear()
    gc.collect()
    logger.info("[Embedder] 모델 캐시 해제 완료")


def _tfidf_embeddings(texts: list[str]) -> np.ndarray:
    """[팀 직접 구현] TF-IDF 기반 폴백 벡터화."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # [오픈소스]
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
