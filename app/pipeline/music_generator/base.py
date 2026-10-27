"""
[팀 직접 구현] 음악 생성 어댑터 베이스 클래스.

폴백 체인: elevenlabs_music → elevenlabs_sound → mock
  - elevenlabs_music: vocals+BGM 동시 생성 (ElevenLabs Music API, 유료 Starter $5/월)
  - elevenlabs_sound: BGM only (ElevenLabs Sound Generation, 무료 티어)
  - mock: 파일 없이 성공 반환 (개발·데모용, 기본값)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MusicGenerationResult:
    success: bool
    file_path: str | None = None        # 로컬 저장 경로 (.mp3/.wav), mock은 None
    duration_seconds: float = 0.0
    style_tags: list = field(default_factory=list)
    lyrics: str = ""                    # 생성된 가사 (있을 경우)
    error_message: str = ""
    estimated_cost_usd: float = 0.0
    adapter: str = ""


class BaseMusicAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float = 15.0,
        genre: str = "",
    ) -> MusicGenerationResult:
        ...

    @abstractmethod
    def adapter_name(self) -> str:
        ...

    def is_available(self) -> bool:
        return True
