"""
[팀 직접 구현] ElevenLabs 음악·사운드 어댑터.

두 가지 모드:
  ElevenLabsMusicAdapter  — Music API (vocals+BGM 동시 생성, 유료 Starter $5/월 필요)
  ElevenLabsSoundAdapter  — Sound Generation API (BGM only, 무료 티어 10,000자/월)

싱잉홍보영상: MusicAdapter 사용 (가사+BGM 자동 sync)
일반 홍보영상 BGM: SoundAdapter 사용 (무료)

보안: API 키는 로그에 절대 노출하지 않는다 (masking.py 적용).
"""

import os
import logging

from .base import BaseMusicAdapter, MusicGenerationResult
from ..utils.masking import sanitize_exception_msg

logger = logging.getLogger(__name__)


class ElevenLabsMusicAdapter(BaseMusicAdapter):
    """
    ElevenLabs Music API — vocals + BGM을 한 번에 생성.
    자동 sync이므로 별도 믹싱 불필요.
    ⚠️ 유료 플랜(Starter $5/월 이상) 필요.
    SDK: client.music.compose() → POST /v1/music (music_length_ms 단위: ms)
    """

    def __init__(self, api_key: str, upload_folder: str = "uploads/music"):
        self._api_key = api_key
        self._upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "elevenlabs_music"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(
        self,
        prompt: str,
        duration: float = 15.0,
        genre: str = "",
    ) -> MusicGenerationResult:
        try:
            from elevenlabs import ElevenLabs  # noqa: PLC0415
            client = ElevenLabs(api_key=self._api_key)
            audio_iter = client.music.compose(
                prompt=prompt,
                music_length_ms=int(duration * 1000),
            )
            file_name = f"music_{abs(hash(prompt))}.mp3"
            file_path = os.path.join(self._upload_folder, file_name)
            with open(file_path, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)
            logger.info("[ElevenLabsMusic] 음악 생성 완료: %s", file_path)
            return MusicGenerationResult(
                success=True,
                file_path=file_path,
                duration_seconds=duration,
                adapter=self.adapter_name(),
                estimated_cost_usd=round(duration * 0.005, 4),
            )
        except ImportError:
            msg = "elevenlabs SDK 미설치 — pip install elevenlabs"
            logger.error("[ElevenLabsMusic] %s", msg)
            return MusicGenerationResult(success=False, error_message=msg, adapter=self.adapter_name())
        except Exception as e:
            msg = sanitize_exception_msg(str(e))
            logger.error("[ElevenLabsMusic] 오류: %s", msg)
            return MusicGenerationResult(
                success=False,
                error_message=msg,
                adapter=self.adapter_name(),
            )


class ElevenLabsSoundAdapter(BaseMusicAdapter):
    """
    ElevenLabs Sound Generation API — 보컬 없는 BGM/효과음 생성.
    무료 티어(10,000 chars/월) 사용 가능.
    싱잉홍보영상이 아닌 일반 홍보영상 배경음악용.
    SDK: client.text_to_sound_effects.convert() → POST /v1/sound-generation
    """

    def __init__(self, api_key: str, upload_folder: str = "uploads/music"):
        self._api_key = api_key
        self._upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "elevenlabs_sound"

    def is_available(self) -> bool:
        return bool(self._api_key)

    # ElevenLabs text-to-sound-effects API 글자수 상한
    _MAX_PROMPT_CHARS = 450

    def generate(
        self,
        prompt: str,
        duration: float = 15.0,
        genre: str = "",
    ) -> MusicGenerationResult:
        # 글자수 초과 시 잘라냄 (API 오류 방지)
        if len(prompt) > self._MAX_PROMPT_CHARS:
            prompt = prompt[: self._MAX_PROMPT_CHARS]
            logger.warning("[ElevenLabsSound] 프롬프트가 %d자 초과 → %d자로 잘라냄", self._MAX_PROMPT_CHARS, self._MAX_PROMPT_CHARS)

        try:
            from elevenlabs import ElevenLabs  # noqa: PLC0415
            client = ElevenLabs(api_key=self._api_key)
            # SDK: POST /v1/sound-generation, duration_seconds 단위: 초
            audio_iter = client.text_to_sound_effects.convert(
                text=prompt,
                duration_seconds=float(duration),
            )
            file_name = f"sound_{abs(hash(prompt))}.mp3"
            file_path = os.path.join(self._upload_folder, file_name)
            with open(file_path, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)
            logger.info("[ElevenLabsSound] BGM 생성 완료: %s", file_path)
            return MusicGenerationResult(
                success=True,
                file_path=file_path,
                duration_seconds=duration,
                adapter=self.adapter_name(),
                estimated_cost_usd=0.0,
            )
        except ImportError:
            msg = "elevenlabs SDK 미설치 — pip install elevenlabs"
            logger.error("[ElevenLabsSound] %s", msg)
            return MusicGenerationResult(success=False, error_message=msg, adapter=self.adapter_name())
        except Exception as e:
            raw = str(e)
            msg = sanitize_exception_msg(raw)
            logger.error("[ElevenLabsSound] 오류: %s", msg)
            print(f"[ElevenLabsSound] BGM 생성 실패: {msg}", flush=True)
            try:
                import pathlib, datetime as _dt  # noqa: E401
                log_path = pathlib.Path("narration_debug.log")
                log_path.open("a", encoding="utf-8").write(
                    f"[{_dt.datetime.now().strftime('%H:%M:%S')}] [ElevenLabsSound 오류] {msg}\n"
                )
            except Exception:
                pass
            return MusicGenerationResult(
                success=False,
                error_message=msg,
                adapter=self.adapter_name(),
            )
