"""
[팀 직접 구현] 음악 생성 어댑터 팩토리.

폴백 체인:
  elevenlabs_music (vocals+BGM, ELEVENLABS_API_KEY + 유료 플랜)
  elevenlabs_sound (BGM only, ELEVENLABS_API_KEY, 무료 티어)
  mock (항상 동작, 기본값)

일일 생성 한도 초과 시 자동으로 mock 폴백.
"""

import logging
from datetime import date

from .base import BaseMusicAdapter
from .mock_adapter import MockMusicAdapter

logger = logging.getLogger(__name__)

# 어댑터별 초당 비용 (USD)
_COST_PER_SECOND: dict[str, float] = {
    "elevenlabs_music": 0.005,
    "elevenlabs_sound": 0.0,
    "mock_music": 0.0,
}

# 일일 생성 횟수 (in-process 카운터, 프로세스 재시작 시 리셋)
_daily_counts: dict[str, int] = {}
_last_date: str = ""


def _get_today_count(adapter_name: str) -> int:
    global _daily_counts, _last_date
    today = date.today().isoformat()
    if today != _last_date:
        _daily_counts = {}
        _last_date = today
    return _daily_counts.get(adapter_name, 0)


def _increment_count(adapter_name: str) -> None:
    global _daily_counts
    _daily_counts[adapter_name] = _daily_counts.get(adapter_name, 0) + 1


def create_music_adapter(
    adapter_name: str | None = None,
    api_key: str = "",
    upload_folder: str = "uploads/music",
    max_daily: int = 10,
) -> BaseMusicAdapter:
    """
    어댑터 생성. API 키 없거나 일일 한도 초과 시 mock으로 자동 폴백.

    Args:
        adapter_name: "elevenlabs_music" | "elevenlabs_sound" | "mock"
        api_key: ELEVENLABS_API_KEY
        upload_folder: 음악 파일 저장 경로
        max_daily: 일일 유료 호출 최대 횟수
    """
    name = (adapter_name or "mock").lower()

    if name in ("elevenlabs_music", "elevenlabs_sound") and api_key:
        if _get_today_count(name) >= max_daily:
            logger.warning(
                "[MusicFactory] %s 일일 한도(%d) 초과, mock으로 폴백",
                name, max_daily,
            )
            return MockMusicAdapter()

        from .elevenlabs_adapter import ElevenLabsMusicAdapter, ElevenLabsSoundAdapter
        if name == "elevenlabs_music":
            return ElevenLabsMusicAdapter(api_key, upload_folder)
        return ElevenLabsSoundAdapter(api_key, upload_folder)

    if name != "mock":
        logger.warning(
            "[MusicFactory] '%s' 생성 불가 (API 키 없음 또는 미지원), mock으로 폴백",
            name,
        )
    return MockMusicAdapter()


def record_generation(adapter_name: str) -> None:
    """어댑터 호출 성공 후 일일 카운트 증가."""
    if adapter_name not in ("mock_music",):
        _increment_count(adapter_name)


def get_estimated_cost(adapter_name: str, duration_seconds: float) -> float:
    return round(_COST_PER_SECOND.get(adapter_name, 0.0) * duration_seconds, 4)
