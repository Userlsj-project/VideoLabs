"""
[팀 직접 구현] ffmpeg 기반 영상-오디오 믹서.

Veo 3.1 native audio는 분위기 맞춤 수준이라 전문가 품질에 미치지 못함.
이 모듈은 ElevenLabs Sound Generation으로 장르별 BGM을 생성하고
ffmpeg subprocess로 영상의 오디오 트랙을 교체해 완성도를 높인다.

적용 대상: 싱잉홍보영상을 제외한 모든 장르 (싱잉홍보영상은 별도 Step 3-B 처리).
ffmpeg 미설치 시 조용히 건너뜀 (graceful fallback — 원본 Veo 오디오 유지).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def _run_cmd(cmd: list, timeout: int) -> tuple[int, str]:
    """Windows 파이프 데드락 방지 + stdout 버퍼링 방지 ffmpeg 헬퍼."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        try:
            _, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                _, stderr_bytes = proc.communicate(timeout=10)
            except Exception:
                stderr_bytes = b""
            return -1, "timeout"
        return proc.returncode, (stderr_bytes or b"").decode("utf-8", errors="replace")
    except Exception as exc:
        return -1, str(exc)

# 장르별 ElevenLabs Sound Generation 프롬프트
_GENRE_BGM_PROMPTS: dict[str, str] = {
    "홍보영상": (
        "Upbeat, professional, energetic corporate promotional background music. "
        "Bright piano melody, light percussion, modern and polished. "
        "Suitable for a commercial advertisement video."
    ),
    "드라마": (
        "Emotional and cinematic orchestral background music with strings and piano. "
        "Subtle tension, rich atmosphere, suitable for a dramatic narrative scene."
    ),
    "교육": (
        "Light, friendly, and engaging educational background music. "
        "Acoustic instruments, positive and clear tone, approachable atmosphere."
    ),
    "뮤직비디오": (
        "Dynamic, rhythmic, modern music video background track. "
        "Electronic beats, strong bass, stylish and energetic."
    ),
    "다큐멘터리": (
        "Calm, ambient documentary background music. "
        "Minimal instrumentation, neutral and informative tone."
    ),
    "예능/버라이어티": (
        "Fun, playful, upbeat variety show background music. "
        "Light-hearted, comedic energy, bright and cheerful."
    ),
}
_DEFAULT_BGM_PROMPT = (
    "Professional and versatile background music for a video production. "
    "Pleasant atmosphere, neutral and polished."
)


def is_ffmpeg_available() -> bool:
    """ffmpeg이 시스템 PATH에 설치되어 있는지 확인한다."""
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: str) -> float | None:
    """ffprobe로 영상 길이(초)를 반환한다. ffprobe 미설치 시 None."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.Popen(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        stdout_bytes, _ = proc.communicate(timeout=30)
        if proc.returncode == 0:
            return float(stdout_bytes.decode("utf-8", errors="replace").strip())
    except Exception:
        pass
    return None


def mix_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str | None = None,
    timeout: int = 180,
) -> str | None:
    """
    ffmpeg으로 영상의 오디오 트랙을 audio_path로 교체한다.

    - 영상 스트림: -c:v copy 무손실 복사 (재인코딩 없음)
    - 음량 정규화: loudnorm I=-14 LUFS (YouTube 표준)
    - 페이드아웃: 영상 끝 1.5초 자연스럽게 감쇠

    Returns:
        성공 시 결과 파일 경로. 실패(ffmpeg 미설치 포함) 시 None.
    """
    if not is_ffmpeg_available():
        logger.warning("[AudioMixer] ffmpeg 미설치 — 오디오 믹싱 건너뜀")
        return None

    if not os.path.exists(video_path):
        logger.error("[AudioMixer] 영상 파일 없음: %s", video_path)
        return None

    if not os.path.exists(audio_path):
        logger.error("[AudioMixer] 오디오 파일 없음: %s", audio_path)
        return None

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_bgm{ext}"

    # 음량 정규화 + 페이드아웃 필터 체인 구성
    _FADE_DURATION = 1.5  # 페이드아웃 길이(초)
    duration = get_video_duration(video_path)
    if duration and duration > _FADE_DURATION * 2:
        fade_start = duration - _FADE_DURATION
        audio_filter = (
            f"loudnorm=I=-14:TP=-1.5:LRA=11,"
            f"afade=t=out:st={fade_start:.3f}:d={_FADE_DURATION}"
        )
    else:
        # 영상이 너무 짧으면 페이드 없이 음량 정규화만
        audio_filter = "loudnorm=I=-14:TP=-1.5:LRA=11"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",          # 영상 스트림 무손실 복사
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", audio_filter,     # 음량 정규화 + 페이드아웃
        "-map", "0:v:0",         # 첫 번째 입력의 비디오
        "-map", "1:a:0",         # 두 번째 입력의 오디오
        "-shortest",             # 영상/오디오 중 짧은 쪽에 맞춤
        output_path,
    ]

    code, stderr = _run_cmd(cmd, timeout=timeout)
    if code != 0:
        logger.error("[AudioMixer] ffmpeg 오류(code=%s):\n%s", code, stderr[-400:])
        return None
    logger.info(
        "[AudioMixer] 오디오 믹싱 완료 (%.1fs, loudnorm -14 LUFS, fade %.1fs): %s",
        duration or 0, _FADE_DURATION, output_path,
    )
    return output_path


def get_bgm_prompt(genre: str) -> str:
    """장르에 맞는 BGM 생성 프롬프트를 반환한다."""
    for key, prompt in _GENRE_BGM_PROMPTS.items():
        if key in genre:
            return prompt
    return _DEFAULT_BGM_PROMPT


def duck_bgm_into_video(
    video_path: str,
    bgm_path: str,
    output_path: str | None = None,
    bgm_db: float = -18.0,
    timeout: int = 180,
) -> str | None:
    """
    원본 오디오를 유지하면서 BGM을 낮은 볼륨으로 덧붙인다.

    - 원본 오디오: 그대로 유지
    - BGM: bgm_db 음량으로 감쇠 후 amix (기본 -18dB)
    - 결과: 두 트랙이 섞인 영상

    Returns:
        성공 시 결과 파일 경로. 실패 시 None.
    """
    if not is_ffmpeg_available():
        logger.warning("[AudioMixer] ffmpeg 없음 — duck 믹싱 건너뜀")
        return None

    if not os.path.exists(video_path):
        logger.error("[AudioMixer] duck: 영상 없음 %s", video_path)
        return None
    if not os.path.exists(bgm_path):
        logger.error("[AudioMixer] duck: BGM 없음 %s", bgm_path)
        return None

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_duck{ext}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", bgm_path,
        "-c:v", "copy",
        "-filter_complex",
        (
            f"[1:a]volume={bgm_db}dB,apad[bgm];"
            "[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[aout]"
        ),
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    code, stderr = _run_cmd(cmd, timeout=timeout)
    if code != 0:
        logger.error("[AudioMixer] duck 오류(code=%s):\n%s", code, stderr[-400:])
        return None
    logger.info("[AudioMixer] duck BGM 믹싱 완료 (BGM %+.0fdB): %s", bgm_db, output_path)
    return output_path


def _video_has_audio(video_path: str) -> bool:
    """ffprobe로 영상에 오디오 스트림이 있는지 확인한다."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    try:
        proc = subprocess.Popen(
            [
                ffprobe, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        stdout_bytes, _ = proc.communicate(timeout=10)
        return "audio" in stdout_bytes.decode("utf-8", errors="replace")
    except Exception:
        return False


def mix_narration_into_video(
    video_path: str,
    narration_cues: list,
    output_path: str | None = None,
    bgm_path: str | None = None,
    bgm_volume_db: float = -12.0,
    narration_db: float = 0.0,
    timeout: int = 180,
) -> str | None:
    """
    [팀 직접 구현] 나레이션 큐를 타임스탬프에 맞춰 영상에 삽입한다.

    - 나레이션 각 큐: adelay 필터로 정확한 타임스탬프에 배치
    - 영상에 기존 오디오(BGM)가 있으면 -6dB로 낮춰 함께 믹싱 (BGM 소실 방지)
    - BGM(선택 인자): bgm_volume_db로 낮춰 배경에 추가
    - 영상 오디오 없는 경우(Kling generate_audio=False) 정상 동작

    Args:
        narration_cues: NarrationCue 리스트 (audio_path, time_seconds 필드 필요)
        bgm_path: 선택적 외부 BGM 파일 경로
        bgm_volume_db: 외부 BGM 볼륨 (기본 -12dB)

    Returns:
        성공 시 출력 파일 경로, 실패 시 None
    """
    if not is_ffmpeg_available():
        logger.warning("[AudioMixer] ffmpeg 없음 — 나레이션 믹싱 건너뜀")
        return None

    valid_cues = [c for c in narration_cues if c.audio_path and os.path.exists(c.audio_path)]
    if not valid_cues:
        logger.error("[AudioMixer] 유효한 나레이션 오디오 파일 없음")
        return None

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_narrated{ext}"

    # 영상에 기존 오디오(BGM 등)가 있으면 함께 믹싱
    video_has_bgm = _video_has_audio(video_path)

    # ffmpeg 입력 구성: [영상, 나레이션1, 나레이션2, ..., 외부BGM(선택)]
    cmd = ["ffmpeg", "-y", "-i", video_path]
    for cue in valid_cues:
        cmd += ["-i", cue.audio_path]
    if bgm_path and os.path.exists(bgm_path):
        cmd += ["-i", bgm_path]

    # filter_complex 구성
    narr_count = len(valid_cues)
    filters = []

    # 나레이션 각 큐를 지정 타임스탬프에 배치 (볼륨 조정 포함)
    for i, cue in enumerate(valid_cues):
        delay_ms = cue.time_seconds * 1000
        if narration_db != 0.0:
            filters.append(
                f"[{i + 1}:a]volume={narration_db}dB,adelay={delay_ms}|{delay_ms}[a{i}]"
            )
        else:
            filters.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(narr_count))
    mix_count = narr_count

    # 외부 BGM이 없을 때만 원본 영상 오디오를 믹싱에 포함
    # 외부 BGM이 있으면 원본 영상 오디오(Kling 자체 생성 음성 포함 가능)를 제외해 에코 방지
    if video_has_bgm and not (bgm_path and os.path.exists(bgm_path)):
        filters.append("[0:a]volume=-6dB[orig_bgm]")
        mix_inputs = "[orig_bgm]" + mix_inputs
        mix_count += 1

    # 외부 BGM 파일이 별도로 전달된 경우
    if bgm_path and os.path.exists(bgm_path):
        bgm_idx = narr_count + 1
        filters.append(f"[{bgm_idx}:a]volume={bgm_volume_db}dB[ext_bgm]")
        mix_inputs += "[ext_bgm]"
        mix_count += 1

    # duration=longest + apad 제거 + -t로 영상 길이 제한
    # apad를 쓰면 BGM 스트림이 무한 패딩되어 3분 타임아웃 발생 → apad 없이 -t로 컷
    filters.append(
        f"{mix_inputs}amix=inputs={mix_count}:duration=longest:normalize=0[out]"
    )

    video_dur = get_video_duration(video_path)
    extra_t = ["-t", str(video_dur)] if video_dur else []

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "0:v:0",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-max_muxing_queue_size", "4096",
        *extra_t,
        output_path,
    ]

    code, stderr = _run_cmd(cmd, timeout=timeout)
    if code != 0:
        err_msg = stderr[-800:]
        logger.error("[AudioMixer] 나레이션 믹싱 오류(code=%s):\n%s", code, err_msg)
        print(f"[AudioMixer] ffmpeg 오류:\n{err_msg}", flush=True)
        try:
            import pathlib
            pathlib.Path("narration_debug.log").open("a", encoding="utf-8").write(
                f"[ffmpeg 오류] {err_msg}\n[cmd] {' '.join(cmd)}\n"
            )
        except Exception:
            pass
        return None
    logger.info(
        "[AudioMixer] 나레이션 믹싱 완료 (%d큐, 기존BGM포함=%s, narr%+.0fdB): %s",
        narr_count, video_has_bgm, narration_db, output_path,
    )
    return output_path


def generate_and_mix_bgm(
    video_path: str,
    genre: str,
    elevenlabs_key: str,
    upload_folder: str = "uploads/music",
    target_duration: int = 8,
    audio_mode: str = "replace",
    bgm_prompt_override: str | None = None,
    bgm_cache_path: str | None = None,
    bgm_db: float = -18.0,
) -> str | None:
    """
    ElevenLabs Sound로 장르별 BGM을 생성하고 영상에 믹싱한다.

    Args:
        audio_mode:     'replace' — 원본 오디오 교체
                        'duck'    — 원본 유지 + BGM -18dB 믹스
                        'skip'    — BGM 생성 건너뜀
        bgm_cache_path: 지정 시 생성된 BGM 파일을 해당 경로에 복사 보관 (나레이션 재합성용)

    Returns:
        믹싱된 영상 경로. 실패 시 None.
    """
    if audio_mode == "skip":
        return None

    if not is_ffmpeg_available():
        logger.warning("[AudioMixer] ffmpeg 없음 — BGM 믹싱 건너뜀")
        return None

    from .music_generator.elevenlabs_adapter import ElevenLabsSoundAdapter

    adapter = ElevenLabsSoundAdapter(
        api_key=elevenlabs_key,
        upload_folder=upload_folder,
    )

    bgm_prompt = bgm_prompt_override if bgm_prompt_override else get_bgm_prompt(genre)
    bgm_duration = float(max(target_duration + 5, 15))

    logger.info(
        "[AudioMixer] BGM 생성 시작: 장르=%s, %d초, 모드=%s, 프롬프트 %d자",
        genre, int(bgm_duration), audio_mode, len(bgm_prompt),
    )
    print(f"[AudioMixer] BGM 생성 시작: {int(bgm_duration)}초, 프롬프트 {len(bgm_prompt)}자", flush=True)
    music_result = adapter.generate(
        prompt=bgm_prompt,
        duration=bgm_duration,
        genre=genre,
    )

    if not music_result.success or not music_result.file_path:
        logger.warning("[AudioMixer] BGM 생성 실패: %s", music_result.error_message)
        print(f"[AudioMixer] BGM 생성 실패: {music_result.error_message}", flush=True)
        return None

    # 캐시 저장 — 나레이션 재생성 시 BGM 자동 재합성에 사용
    if bgm_cache_path:
        try:
            os.makedirs(os.path.dirname(bgm_cache_path), exist_ok=True)
            import shutil as _shutil
            _shutil.copy2(music_result.file_path, bgm_cache_path)
            logger.info("[AudioMixer] BGM 캐시 저장: %s", bgm_cache_path)
        except Exception as e:
            logger.warning("[AudioMixer] BGM 캐시 저장 실패 (믹싱은 계속): %s", e)

    if audio_mode == "duck":
        mixed = duck_bgm_into_video(video_path, music_result.file_path, bgm_db=bgm_db)
    else:
        mixed = mix_audio_into_video(video_path, music_result.file_path)

    try:
        if os.path.exists(music_result.file_path):
            os.remove(music_result.file_path)
    except OSError:
        pass

    return mixed
