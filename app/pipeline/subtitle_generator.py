"""
[팀 직접 구현] openai-whisper 기반 자막 자동 생성기.

MIT 라이선스 Whisper를 사용해 영상 음성을 인식하고 SRT 자막을 생성한다.
ffmpeg으로 오디오를 추출 → Whisper 음성 인식 → SRT 변환 → 선택적 burn-in.
whisper/ffmpeg 미설치 시 None 반환 (graceful fallback).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_SIZE = "small"  # tiny(74MB) · small(461MB) · medium(1.5GB)


def is_whisper_available() -> bool:
    """faster-whisper 또는 openai-whisper 설치 여부 확인."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_whisper_backend() -> str:
    """사용 가능한 Whisper 백엔드 반환: 'faster' | 'openai' | None."""
    try:
        import faster_whisper  # noqa: F401
        return "faster"
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return "openai"
    except ImportError:
        return None


def extract_subtitles(
    video_path: str,
    output_dir: str,
    language: str = "ko",
    burn_in: bool = False,
    model_size: str = _DEFAULT_MODEL_SIZE,
) -> str | None:
    """
    영상에서 음성을 추출해 SRT 자막을 생성한다.

    Args:
        video_path: 입력 영상 경로
        output_dir: SRT 파일 저장 디렉토리
        language: 언어 코드 (ko, en, ja, zh, ...)
        burn_in: True이면 ffmpeg으로 자막을 영상에 하드코딩한 새 파일 추가 생성
        model_size: Whisper 모델 크기 (tiny/small/medium/large)

    Returns:
        SRT 파일 경로. 실패 시 None.
    """
    backend = _get_whisper_backend()
    if not backend:
        logger.warning("[Subtitle] faster-whisper/openai-whisper 미설치 — 자막 생성 건너뜀")
        return None

    if not shutil.which("ffmpeg"):
        logger.warning("[Subtitle] ffmpeg 미설치 — 오디오 추출 불가")
        return None

    if not os.path.exists(video_path):
        logger.error("[Subtitle] 영상 파일 없음: %s", video_path)
        return None

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    srt_path = os.path.join(output_dir, f"{base}.srt")

    audio_path = None
    try:
        # 1. ffmpeg으로 16 kHz 모노 WAV 추출 (Whisper 입력 형식)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if r.returncode != 0:
            logger.error("[Subtitle] 오디오 추출 실패 (ffmpeg returncode %d)", r.returncode)
            return None

        # 2. Whisper 음성 인식 (faster-whisper 우선, openai-whisper 폴백)
        import gc
        if backend == "faster":
            from faster_whisper import WhisperModel  # [오픈소스] 4배 빠름
            wmodel = WhisperModel(model_size, device="cpu", compute_type="int8")
            lang_arg = language if language != "auto" else None
            segments_iter, _ = wmodel.transcribe(audio_path, language=lang_arg)
            segments = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments_iter
            ]
        else:
            import whisper  # [오픈소스] openai-whisper 폴백
            wmodel = whisper.load_model(model_size)
            result = wmodel.transcribe(audio_path, language=language, verbose=False)
            segments = result.get("segments", [])

        del wmodel
        gc.collect()

        # 3. SRT 포맷으로 저장
        srt_content = _segments_to_srt(segments)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info("[Subtitle] SRT 생성 완료 (%d 세그먼트, backend=%s): %s", len(segments), backend, srt_path)

        # 4. 선택적 burn-in (자막 하드코딩)
        if burn_in:
            burned_path = os.path.join(output_dir, f"{base}_subtitled.mp4")
            _burn_subtitles(video_path, srt_path, burned_path)

        return srt_path

    except Exception as e:
        logger.error("[Subtitle] 자막 생성 실패: %s", e)
        return None
    finally:
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def generate_srt_from_cues(cues: list, output_path: str, video_duration: float | None = None) -> str | None:
    """
    나레이션 큐 데이터로 SRT 자막을 생성한다.

    Whisper 재인식 없이 대본 텍스트를 그대로 사용하므로 오인식이 없다.

    Args:
        cues:           NarrationCue 리스트 (time_seconds, text 필드 필요)
        output_path:    저장할 SRT 파일 경로
        video_duration: 영상 전체 길이(초). 마지막 큐 끝 시간 계산에 사용.

    Returns:
        SRT 파일 경로. 실패 시 None.
    """
    if not cues:
        logger.warning("[Subtitle] 큐가 없어 SRT 생성 불가")
        return None

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = []
        for i, cue in enumerate(cues):
            start = float(cue.time_seconds)
            # 끝 시간: 다음 큐 시작 0.3초 전 / 텍스트 길이 기반 추정 중 짧은 쪽
            text_dur = max(2.0, len(cue.text) / 4.5)  # 한국어 평균 TTS ~4.5자/초
            if i + 1 < len(cues):
                next_start = float(cues[i + 1].time_seconds)
                end = min(next_start - 0.3, start + text_dur)
            else:
                end = start + text_dur
                if video_duration:
                    end = min(end, video_duration - 0.1)

            lines.append(f"{i + 1}\n{_fmt_time(start)} --> {_fmt_time(end)}\n{cue.text.strip()}\n")

        srt_content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        logger.info("[Subtitle] 큐 기반 SRT 생성 완료 (%d 세그먼트): %s", len(cues), output_path)
        return output_path
    except Exception as e:
        logger.error("[Subtitle] 큐 기반 SRT 생성 실패: %s", e)
        return None


def get_srt_path(video_path: str, output_dir: str) -> str:
    """영상 경로에서 예상 SRT 파일 경로를 반환한다 (존재 여부 무관)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{base}.srt")


def _segments_to_srt(segments: list) -> str:
    """Whisper segments 리스트를 SRT 포맷 문자열로 변환한다."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_time(seg.get("start", 0.0))
        end = _fmt_time(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """초 단위 시간을 SRT 타임코드(HH:MM:SS,mmm)로 변환."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _burn_subtitles(video_path: str, srt_path: str, output_path: str) -> str | None:
    """ffmpeg subtitles 필터로 자막을 영상에 하드코딩한다."""
    # Windows 경로: 역슬래시→슬래시, 드라이브 콜론 이스케이프
    srt_escaped = srt_path.replace("\\", "/")
    if len(srt_escaped) > 1 and srt_escaped[1] == ":":
        srt_escaped = srt_escaped[0] + "\\:" + srt_escaped[2:]

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_escaped}'",
        "-c:a", "copy",
        output_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        if r.returncode == 0:
            logger.info("[Subtitle] burn-in 완료: %s", output_path)
            return output_path
        logger.error("[Subtitle] burn-in 실패 (returncode %d)", r.returncode)
    except Exception as e:
        logger.error("[Subtitle] burn-in 예외: %s", e)
    return None
