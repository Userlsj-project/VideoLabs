"""
어댑터 팩토리: 설정값에 따라 적절한 어댑터를 생성한다.

유료 어댑터(veo/runway) 호출 전 일일 생성 한도를 확인한다.
한도 초과 시 자동으로 mock 폴백 + 경고 로그를 남긴다.
"""

from __future__ import annotations

import logging

from .base import BaseVideoGeneratorAdapter
from .fal_adapter import FalAdapter
from .kling_adapter import KlingAdapter
from .mock_adapter import MockVideoGeneratorAdapter
from .runway_adapter import RunwayAdapter
from .veo_adapter import VeoAdapter

logger = logging.getLogger(__name__)

# 어댑터별 초당 예상 단가 (USD/sec)
_COST_PER_SECOND: dict[str, float] = {
    "veo":    0.75,   # Veo 3.1 실효 단가 (~$0.75/sec, preview 과금 기준)
    "runway": 0.05,   # Runway Gen-4 Turbo 기준 ($0.05/sec)
    "fal":    0.03,   # fal.ai Seedance 1 Lite 기준 ($0.03/sec 추정)
    "kling":  0.084,  # Kling V3 Standard, 오디오 비활성화 기준 ($0.084/sec)
    "mock":   0.00,
}


def create_adapter(
    adapter_name: str,
    upload_folder: str = "uploads",
    max_daily: int = 20,
    **kwargs,
) -> BaseVideoGeneratorAdapter:
    """
    어댑터 이름과 설정으로 어댑터 인스턴스를 반환.

    Args:
        adapter_name: "veo" | "runway" | "fal" | "kling" | "mock"
        upload_folder: 생성된 영상 저장 폴더
        max_daily: 유료 어댑터 일일 최대 호출 수 (초과 시 mock 폴백)
        **kwargs: 어댑터별 설정 (veo_api_key, runway_api_key, fal_api_key 등)
    """
    name = adapter_name.lower()

    # 유료 어댑터는 일일 한도 확인 후 생성
    if name in ("veo", "runway", "fal", "kling"):
        today_count = _get_today_paid_count()
        if today_count >= max_daily:
            logger.warning(
                "[VideoFactory] 일일 유료 생성 한도 초과 (%d/%d) — mock 폴백",
                today_count, max_daily,
            )
            return MockVideoGeneratorAdapter(upload_folder=upload_folder)

    if name == "veo":
        api_key = kwargs.get("veo_api_key", "")
        quality_tier = kwargs.get("veo_quality_tier", "fast")
        model = kwargs.get("veo_model", "veo-3.1-generate-preview")
        extend_target = kwargs.get("veo_extend_target_seconds", 8)
        adapter = VeoAdapter(
            api_key=api_key,
            quality_tier=quality_tier,
            upload_folder=upload_folder,
            model=model,
            extend_target_seconds=extend_target,
        )
        if adapter.is_available():
            return adapter
        logger.warning("[VideoFactory] VEO_API_KEY 없음 — mock 폴백")

    elif name == "runway":
        api_key = kwargs.get("runway_api_key", "")
        adapter = RunwayAdapter(api_key=api_key, upload_folder=upload_folder)
        if adapter.is_available():
            return adapter
        logger.warning("[VideoFactory] RUNWAY_API_KEY 없음 — mock 폴백")

    elif name == "fal":
        api_key = kwargs.get("fal_api_key", "")
        fal_model = kwargs.get("fal_model", "fal-ai/bytedance/seedance-1-lite")
        adapter = FalAdapter(api_key=api_key, upload_folder=upload_folder, model=fal_model)
        if adapter.is_available():
            return adapter
        logger.warning("[VideoFactory] FAL_API_KEY 없음 — mock 폴백")

    elif name == "kling":
        api_key = kwargs.get("fal_api_key", "")  # Kling도 fal.ai 플랫폼 → 동일 키 사용
        extend_target = kwargs.get("kling_extend_target_seconds", 8)
        model_id = kwargs.get("kling_model", "fal-ai/kling-video/v3/standard/text-to-video")
        adapter = KlingAdapter(
            api_key=api_key,
            model_id=model_id,
            upload_folder=upload_folder,
            extend_target_seconds=extend_target,
        )
        if adapter.is_available():
            return adapter
        logger.warning("[VideoFactory] FAL_API_KEY 없음 (Kling) — mock 폴백")

    return MockVideoGeneratorAdapter(upload_folder=upload_folder)


def get_estimated_cost(adapter_name: str, duration_seconds: int = 8) -> float:
    """어댑터 이름과 영상 길이(초)로 예상 비용(USD)을 반환한다."""
    per_sec = _COST_PER_SECOND.get(adapter_name.lower(), 0.0)
    return round(per_sec * max(duration_seconds, 1), 4)


def _get_today_paid_count() -> int:
    """오늘 실제 유료 어댑터(veo/runway)로 생성된 VideoRound 수를 반환."""
    try:
        from datetime import date
        from sqlalchemy import func, cast, Date as SADate
        from ...models import VideoRound

        today = date.today()
        count = (
            VideoRound.query
            .filter(
                VideoRound.generator_adapter.in_(["veo", "runway", "fal", "kling"]),
                func.date(VideoRound.created_at) == today.isoformat(),
            )
            .count()
        )
        return count or 0
    except Exception as e:
        logger.debug("[VideoFactory] 일일 카운트 조회 실패 (무시): %s", e)
        return 0
