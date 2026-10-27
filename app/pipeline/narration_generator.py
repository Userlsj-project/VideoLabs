"""
[팀 직접 구현] 나레이션 자동 생성 모듈.

프롬프트에서 타임스탬프 큐를 파싱하고 ElevenLabs TTS로 오디오 파일을 생성한다.
ElevenLabs 무료 플랜으로 사용 가능 (10,000 크레딧/월, 나레이션 ~42자 = 무시할 수준).

파싱 대상 패턴:
  At 1s: "인공지능이 물리 세계와 만나는 순간."
  At 9s (Korean): "자율주행, 로보틱스, 사물인터넷."
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ElevenLabs TTS 기본 설정
_DEFAULT_VOICE_ID = "WnFTM7Gw75NWybKycaRt"  # Victor - Engaged, Authoritative Narrator
_TTS_MODEL = "eleven_v4"

# 나레이션 큐 파싱 패턴: At Xs, At Xs (Korean), At Xs (label): "text"
# ^ + MULTILINE 으로 줄 시작에서만 매칭 → "builds to climax at 16s" 같은 중간 문구 오매칭 방지
# chr(0x22)=ASCII"  chr(0x201C)=left"  chr(0x201D)=right" — 세 종류 따옴표 모두 인식
_Q = chr(0x22) + chr(0x201C) + chr(0x201D)
_CUE_PATTERN = re.compile(
    "^At\\s+(\\d+)s[^:\\n]*:\\s*[" + _Q + "]([^" + _Q + "]+)[" + _Q + "]",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class NarrationCue:
    time_seconds: int
    text: str
    audio_path: str | None = None


@dataclass
class NarrationResult:
    success: bool
    cues: list[NarrationCue] = field(default_factory=list)
    error_message: str | None = None


def parse_narration_cues(prompt: str) -> list[NarrationCue]:
    """프롬프트에서 나레이션 타임스탬프 큐를 파싱한다."""
    matches = _CUE_PATTERN.findall(prompt)
    cues = [NarrationCue(time_seconds=int(t), text=text.strip()) for t, text in matches]
    cues.sort(key=lambda c: c.time_seconds)
    logger.info("[Narration] 큐 파싱 완료: %d개 — %s", len(cues), [(c.time_seconds, c.text[:20]) for c in cues])
    return cues


def generate_all_narrations(
    cues: list[NarrationCue],
    output_dir: str,
    api_key: str = "",
    voice_id: str = _DEFAULT_VOICE_ID,
    voice_settings_override: dict | None = None,
    emotion_profile: dict | None = None,
) -> NarrationResult:
    """모든 나레이션 큐에 대해 ElevenLabs TTS로 오디오 파일을 생성한다."""
    if not cues:
        return NarrationResult(success=False, error_message="나레이션 큐가 없습니다")

    if not api_key:
        return NarrationResult(success=False, error_message="ELEVENLABS_API_KEY가 설정되지 않았습니다")

    os.makedirs(output_dir, exist_ok=True)
    total = len(cues)

    for i, cue in enumerate(cues):
        out_path = os.path.join(output_dir, f"narr_{cue.time_seconds}s_{i}.mp3")
        emotion_tag, speed = _get_cue_tag(i, total, emotion_profile) if emotion_profile else (None, 1.0)
        ok, err = _elevenlabs_tts(
            cue.text, api_key, voice_id, out_path,
            voice_settings_override, emotion_tag, speed,
        )
        if ok:
            cue.audio_path = out_path
            logger.info(
                "[Narration] TTS 완료: %ds tag=%s speed=%.2f — %s",
                cue.time_seconds, emotion_tag, speed, cue.text[:30],
            )
        else:
            return NarrationResult(
                success=False,
                cues=cues,
                error_message=f"큐 {i + 1} 오디오 생성 실패: {err}",
            )

    return NarrationResult(success=True, cues=cues)


def _get_cue_tag(index: int, total: int, profile: dict) -> tuple[str | None, float]:
    """큐 위치(첫/중간/마지막)에 따른 감정 태그와 속도를 반환."""
    if total == 1 or index == 0:
        tag = profile.get("opening")
    elif index == total - 1:
        tag = profile.get("closing")
    else:
        tag = profile.get("middle")
    return tag, float(profile.get("speed", 1.0))


def _preprocess_tts_text(text: str, emotion_tag: str | None = None) -> str:
    """쉼표 뒤 한글 앞에 SSML 일시정지 삽입 — 아나운서 호흡 표현."""
    processed = re.sub(r",\s*(?=[가-힣])", ', <break time="0.8s"/> ', text)
    if emotion_tag:
        return f"[{emotion_tag}] {processed}"
    return processed


def _elevenlabs_tts(
    text: str,
    api_key: str,
    voice_id: str,
    output_path: str,
    voice_settings_override: dict | None = None,
    emotion_tag: str | None = None,
    speed: float = 1.0,
) -> tuple[bool, str]:
    try:
        from elevenlabs import ElevenLabs
        from elevenlabs.types import VoiceSettings

        vs = voice_settings_override or {}
        client = ElevenLabs(api_key=api_key)
        tts_text = _preprocess_tts_text(text, emotion_tag)
        convert_kwargs: dict = dict(
            voice_id=voice_id,
            text=tts_text,
            model_id=_TTS_MODEL,
            voice_settings=VoiceSettings(
                stability=vs.get("stability", 0.72),
                similarity_boost=vs.get("similarity_boost", 0.82),
                style=vs.get("style", 0.08),
                use_speaker_boost=vs.get("use_speaker_boost", True),
            ),
        )
        audio = client.text_to_speech.convert(**convert_kwargs)
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return True, ""
    except ImportError:
        return False, "elevenlabs SDK 미설치 — pip install elevenlabs"
    except Exception as e:
        return False, str(e)


def cleanup_narration_files(cues: list[NarrationCue]) -> None:
    """믹싱 완료 후 임시 나레이션 오디오 파일을 삭제한다."""
    for cue in cues:
        if cue.audio_path and os.path.exists(cue.audio_path):
            try:
                os.remove(cue.audio_path)
            except OSError:
                pass
