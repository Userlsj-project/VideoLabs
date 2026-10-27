"""
[팀 직접 구현] 립싱크 어댑터 팩토리.

자동 폴백 체인:
  sadtalker (설치됐으면) → wav2lip (설치됐으면) → mock (항상)

LIPSYNC_ADAPTER 환경변수로 어댑터를 지정하더라도,
해당 어댑터가 is_available() = False이면 자동으로 다음 어댑터로 폴백한다.
데모가 어떤 환경에서도 항상 돌아가도록 보장.
"""

import logging

from .base import BaseLipSyncAdapter
from .mock_adapter import MockLipSyncAdapter

logger = logging.getLogger(__name__)


def create_lipsync_adapter(adapter_name: str | None = None) -> BaseLipSyncAdapter:
    """
    어댑터를 생성한다. 미설치 시 자동 폴백.

    Args:
        adapter_name: "sadtalker" | "wav2lip" | "mock"

    Returns:
        사용 가능한 어댑터 (최악의 경우 MockLipSyncAdapter)
    """
    name = (adapter_name or "mock").lower()

    if name == "sadtalker":
        from .sadtalker_adapter import SadTalkerAdapter
        adapter = SadTalkerAdapter()
        if adapter.is_available():
            logger.info("[LipSyncFactory] SadTalker 사용")
            return adapter
        logger.warning("[LipSyncFactory] SadTalker 미설치, Wav2Lip 시도")
        # sadtalker 지정이지만 없으면 wav2lip 시도
        name = "wav2lip"

    if name == "wav2lip":
        from .wav2lip_adapter import Wav2LipAdapter
        adapter = Wav2LipAdapter()
        if adapter.is_available():
            logger.info("[LipSyncFactory] Wav2Lip 사용")
            return adapter
        logger.warning("[LipSyncFactory] Wav2Lip도 미설치, mock으로 폴백")

    if name not in ("mock", "wav2lip", "sadtalker"):
        logger.warning("[LipSyncFactory] 알 수 없는 어댑터 '%s', mock 사용", name)

    return MockLipSyncAdapter()
