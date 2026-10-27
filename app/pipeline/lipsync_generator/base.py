"""
[팀 직접 구현] 립싱크 어댑터 베이스 클래스.

폴백 체인: sadtalker → wav2lip → mock
  - sadtalker: 표정+눈+입 자연스러움, 단일 이미지 입력, 비상업 연구용 → 졸작 OK
  - wav2lip: 입만 교체, MIT 라이선스, sadtalker 설치 실패 시 폴백
  - mock: 원본 영상 경로 그대로 반환 (파일 생성 없음), 항상 동작
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LipSyncResult:
    success: bool
    file_path: str | None = None    # 생성된 립싱크 영상 경로, mock은 원본 경로 그대로
    duration_seconds: float = 0.0
    error_message: str = ""
    adapter: str = ""


class BaseLipSyncAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        face_image_path: str,   # 캐릭터 이미지 (JPG/PNG)
        audio_path: str,        # ElevenLabs 생성 오디오 경로
        output_path: str,       # 결과 영상 저장 경로
    ) -> LipSyncResult:
        ...

    @abstractmethod
    def adapter_name(self) -> str:
        ...

    def is_available(self) -> bool:
        return True
