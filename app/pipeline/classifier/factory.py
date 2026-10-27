"""
[팀 직접 설계]
분류기 팩토리: 설정값에 따라 적절한 분류기 인스턴스를 반환.

우선순위:
  1. EmbeddingClassifier  — sentence-transformers 기반, 인증 불필요 (기본)
  2. ZeroShotClassifier   — NLI 기반 (모델 접근 가능할 때)
  3. RuleBasedClassifier  — 키워드 규칙 기반 (완전 오프라인 폴백)

싱글턴 패턴:
  한 프로세스에서 항상 동일한 인스턴스를 반환한다.
  update_prototype()으로 업데이트된 프로토타입이 요청 간에 유지된다.
"""

from __future__ import annotations

import logging

from .base import BaseClassifier

logger = logging.getLogger(__name__)

# ── 싱글턴 ──────────────────────────────────────────────────────────────────
# 한 프로세스 내에서 동일 인스턴스를 재사용 → update_prototype() 학습 결과 유지

_classifier_instance: BaseClassifier | None = None


def get_classifier() -> BaseClassifier | None:
    """현재 활성 분류기 인스턴스를 반환한다 (없으면 None)."""
    return _classifier_instance


def release_classifier() -> None:
    """분류기(+내부 SentenceTransformer ~471MB)를 해제해 GC가 수거할 수 있게 한다.
    영상 재생성처럼 분류기가 불필요한 단계 직전에 호출한다."""
    global _classifier_instance
    import gc
    _classifier_instance = None
    gc.collect()
    logger.info("[Classifier] 분류기 인스턴스 해제 완료")


def create_classifier(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    specificity_threshold: float = 0.45,
    validity_threshold: float = 0.40,
    prototypes_path: str | None = None,
) -> BaseClassifier:
    """
    [팀 직접 설계] 분류기 생성. 계정 불필요한 임베딩 분류기를 우선 사용.

    두 번째 호출부터는 싱글턴을 반환한다.
    prototypes_path: 학습된 프로토타입 저장 경로 (첫 생성 시에만 적용)

    Args:
        model_name: 임베딩 또는 NLI 모델 ID (설정값으로 주입)
        specificity_threshold: 구체성 점수 최소 임계값
        validity_threshold: 타당성 점수 최소 임계값
        prototypes_path: 프로토타입 .npz 저장 경로 (확장자 제외 가능)
    """
    global _classifier_instance

    if _classifier_instance is not None:
        return _classifier_instance

    # ── 1순위: EmbeddingClassifier (sentence-transformers, 인증 불필요) ──────
    try:
        from sentence_transformers import SentenceTransformer  # 설치 여부 확인
        from .embedding_classifier import EmbeddingClassifier

        # 임베딩 모델명 결정: NLI 모델명이 들어오면 임베딩 모델로 대체
        embedding_model = _resolve_embedding_model(model_name)

        logger.info("임베딩 분류기 사용 (인증 불필요): %s", embedding_model)
        # EmbeddingClassifier는 규칙 기반 점수를 쓰므로 NLI보다 임계값을 낮게 설정
        _classifier_instance = EmbeddingClassifier(
            embedding_model=embedding_model,
            specificity_threshold=max(specificity_threshold * 0.75, 0.25),
            validity_threshold=max(validity_threshold * 0.70, 0.20),
            prototypes_path=prototypes_path,
        )
        return _classifier_instance
    except ImportError:
        logger.warning("sentence-transformers 미설치 → 다음 분류기 시도")
    except Exception as e:
        logger.warning("EmbeddingClassifier 초기화 실패: %s → 다음 분류기 시도", e)

    # ── 2순위: ZeroShotClassifier (NLI, 모델 접근 가능할 때만) ──────────────
    try:
        from transformers import pipeline as _  # 설치 여부 확인
        from .zeroshot import ZeroShotClassifier

        logger.info("제로샷 분류기 사용: %s", model_name)
        _classifier_instance = ZeroShotClassifier(
            model_name=model_name,
            specificity_threshold=specificity_threshold,
            validity_threshold=validity_threshold,
        )
        return _classifier_instance
    except ImportError:
        logger.warning("transformers 미설치 → 규칙 기반 분류기로 폴백")
    except Exception as e:
        logger.warning("ZeroShotClassifier 초기화 실패: %s → 규칙 기반 폴백", e)

    # ── 3순위: RuleBasedClassifier (완전 오프라인) ───────────────────────────
    logger.warning("규칙 기반 분류기 사용 (폴백)")
    from .rule_based import RuleBasedClassifier
    _classifier_instance = RuleBasedClassifier()
    return _classifier_instance


def _resolve_embedding_model(model_name: str) -> str:
    """
    [팀 직접 구현] 설정값에서 임베딩 모델명을 결정.

    NLI 모델명(예: mDeBERTa)이 들어오면 호환 임베딩 모델로 자동 대체.
    임베딩 모델명이 들어오면 그대로 사용.
    """
    _EMBEDDING_MODELS = {
        "paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    }

    if any(em in model_name for em in _EMBEDDING_MODELS):
        return model_name

    # NLI/기타 모델명 → 기본 임베딩 모델로 대체
    logger.info(
        "모델명 '%s'은 임베딩 모델이 아닙니다. "
        "paraphrase-multilingual-MiniLM-L12-v2로 대체합니다.",
        model_name,
    )
    return "paraphrase-multilingual-MiniLM-L12-v2"
