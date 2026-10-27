"""
[팀 직접 구현] 영상 후처리 모듈.

기능:
  1. 로고 배경 제거 (rembg)
  2. 로고 카드 이미지 생성 (Pillow) — 네이비 배경 + 투명 로고 + 학과명
  3. 영상 끝에 페이드아웃 + 로고 카드 이어붙이기 (ffmpeg)
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys


def _run_cmd(cmd: list, timeout: int) -> tuple[int, str]:
    """Windows 파이프 데드락 방지용 ffmpeg 실행 헬퍼.
    capture_output=True 대신 stderr만 PIPE로 받아 TimeoutExpired 후 강제 kill + drain.
    """
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

logger = logging.getLogger(__name__)

_LOGO_SRC_NAME = "군산대학교 로고.jpg"
_LOGO_TRANSPARENT_NAME = "군산대학교 로고_transparent.png"
_BG_COLOR = (13, 27, 62)          # 네이비 (#0D1B3E)
_UNIVERSITY = "군산대학교"
_DEPARTMENT = "임베디드소프트웨어학과"
_FADE_DURATION = 1.5               # 메인 영상 페이드아웃 길이(초)
_LOGO_CARD_DURATION = 6.0          # 로고 카드 표시 시간(초)

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",   # 맑은 고딕 Bold
    "C:/Windows/Fonts/malgun.ttf",     # 맑은 고딕
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def ensure_transparent_logo(project_root: str) -> str | None:
    """코너 flood fill로 바깥 흰 배경만 제거. 원 안쪽 흰색은 유지."""
    src = os.path.join(project_root, _LOGO_SRC_NAME)
    dst = os.path.join(project_root, _LOGO_TRANSPARENT_NAME)

    if os.path.exists(dst):
        logger.info("[PostProcess] 투명 로고 재사용: %s", dst)
        return dst

    if not os.path.exists(src):
        logger.warning("[PostProcess] 로고 원본 없음: %s", src)
        return None

    try:
        from PIL import Image, ImageDraw
        logo = Image.open(src).convert("RGBA")
        w, h = logo.size

        # 4 코너에서 flood fill → 바깥 흰 배경만 투명화
        # 원 안쪽 흰색은 파란 테두리로 막혀 있어 침범 안 됨
        for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            ImageDraw.floodfill(logo, corner, (0, 0, 0, 0), thresh=30)

        logo.save(dst)
        logger.info("[PostProcess] flood fill 배경 제거 완료: %s", dst)
        return dst
    except Exception as e:
        logger.error("[PostProcess] 투명화 실패: %s", e)
        return src


def _get_video_info(video_path: str) -> tuple[int, int, float]:
    """ffprobe로 영상 너비, 높이, 길이(초) 반환. 실패 시 기본값."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 1920, 1080, 22.0
    try:
        proc1 = subprocess.Popen(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        out1, _ = proc1.communicate(timeout=15)
        w, h = (int(x) for x in out1.decode("utf-8", errors="replace").strip().split(","))

        proc2 = subprocess.Popen(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        out2, _ = proc2.communicate(timeout=15)
        duration = float(out2.decode("utf-8", errors="replace").strip())
        return w, h, duration
    except Exception as e:
        logger.warning("[PostProcess] ffprobe 오류, 기본값 사용: %s", e)
        return 1920, 1080, 22.0


def _create_logo_card_image(
    logo_path: str,
    width: int,
    height: int,
    tmp_dir: str,
) -> str | None:
    """Pillow로 로고 카드 이미지 생성."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.error("[PostProcess] Pillow 미설치 — pip install Pillow")
        return None

    img = Image.new("RGB", (width, height), _BG_COLOR)

    try:
        logo = Image.open(logo_path).convert("RGBA")
    except Exception as e:
        logger.error("[PostProcess] 로고 열기 실패: %s", e)
        return None

    # 로고 크기: 높이의 38%, 가로 비율 유지
    logo_h = int(height * 0.38)
    logo_w = int(logo.width * logo_h / logo.height)
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # 로고: 화면 위쪽 중앙 배치
    logo_x = (width - logo_w) // 2
    logo_y = int(height * 0.10)
    img.paste(logo, (logo_x, logo_y), logo)

    # 폰트 로드
    draw = ImageDraw.Draw(img)
    uni_font = dept_font = None
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                uni_font = ImageFont.truetype(fp, int(height * 0.042))
                dept_font = ImageFont.truetype(fp, int(height * 0.058))
                break
            except Exception:
                continue
    if not uni_font:
        uni_font = dept_font = ImageFont.load_default()
        logger.warning("[PostProcess] 한글 폰트 없음 — 기본 폰트 사용")

    text_y = logo_y + logo_h + int(height * 0.04)

    # "군산대학교" (작게, 회색빛 흰색)
    bb1 = draw.textbbox((0, 0), _UNIVERSITY, font=uni_font)
    draw.text(((width - bb1[2] + bb1[0]) // 2, text_y), _UNIVERSITY,
              font=uni_font, fill=(200, 210, 230))

    # "임베디드소프트웨어학과" (크게, 흰색)
    y2 = text_y + (bb1[3] - bb1[1]) + int(height * 0.025)
    bb2 = draw.textbbox((0, 0), _DEPARTMENT, font=dept_font)
    draw.text(((width - bb2[2] + bb2[0]) // 2, y2), _DEPARTMENT,
              font=dept_font, fill=(255, 255, 255))

    out_path = os.path.join(tmp_dir, "logo_card_frame.png")
    img.save(out_path)
    logger.info("[PostProcess] 로고 카드 이미지 생성: %dx%d", width, height)
    return out_path


def _create_logo_card_video(
    card_image_path: str,
    width: int,
    height: int,
    duration: float,
    output_path: str,
) -> bool:
    """로고 카드 이미지를 영상으로 변환. 페이드인 + 무음 오디오 포함."""
    fade_in = min(0.5, duration * 0.2)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", card_image_path,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=disable,"
            f"fps=24,fade=t=in:st=0:d={fade_in}"
        ),
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-shortest",
        output_path,
    ]
    code, stderr = _run_cmd(cmd, timeout=60)
    if code != 0:
        logger.error("[PostProcess] 로고 카드 영상 실패(code=%s):\n%s", code, stderr[-400:])
        return False
    return True


def _fadeout_and_concat(
    main_video: str,
    card_video: str,
    fade_start: float,
    fade_duration: float,
    output_path: str,
) -> bool:
    """페이드아웃 + 로고 카드 이어붙이기를 단일 ffmpeg 호출로 처리.
    기존 _apply_fadeout → _concat_videos 2회 인코딩을 1회로 줄여 CPU/메모리 절약."""
    v1_has_audio = _video_has_audio(main_video)
    if v1_has_audio:
        filter_complex = (
            f"[0:v]fade=t=out:st={fade_start:.3f}:d={fade_duration}[v0];"
            f"[0:a]afade=t=out:st={fade_start:.3f}:d={fade_duration}[a0];"
            "[v0][a0][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", main_video,
            "-i", card_video,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-max_muxing_queue_size", "4096",
            output_path,
        ]
    else:
        # anullsrc를 filter_complex 안에 넣으면 무한 오디오 스트림이 생겨
        # concat 필터가 segment 1 종료를 기다리며 메모리를 무한 소비한다.
        # 해결: 메인 영상 길이에 맞춘 침묵 스트림을 별도 입력으로 전달.
        _, _, main_dur = _get_video_info(main_video)
        silence_dur = main_dur + 0.5  # 약간 여유를 주어 concat이 확실히 처리하도록
        filter_complex = (
            f"[0:v]fade=t=out:st={fade_start:.3f}:d={fade_duration}[v0];"
            "[v0][2:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", main_video,                               # [0]
            "-i", card_video,                               # [1]
            "-f", "lavfi",                                  # [2] 길이가 제한된 침묵
            "-t", f"{silence_dur:.3f}",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-max_muxing_queue_size", "4096",
            output_path,
        ]
    code, stderr = _run_cmd(cmd, timeout=180)
    if code != 0:
        logger.error("[PostProcess] 페이드+이어붙이기 실패(code=%s):\n%s", code, stderr[-400:])
        return False
    return True


def _video_has_audio(video_path: str) -> bool:
    """ffprobe로 영상에 오디오 스트림이 있는지 확인한다."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return False
    try:
        proc = subprocess.Popen(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        )
        out, _ = proc.communicate(timeout=10)
        return "audio" in out.decode("utf-8", errors="replace")
    except Exception:
        return False




def apply_outro(
    video_path: str,
    output_path: str,
    project_root: str,
    fade_duration: float = _FADE_DURATION,
    logo_card_duration: float = _LOGO_CARD_DURATION,
    skip_main_fade: bool = False,
) -> str | None:
    """
    [팀 직접 구현] 영상 끝에 페이드아웃 + 학교 로고 카드를 추가한다.

    처리 순서:
      1. rembg로 투명 로고 확보
      2. Pillow로 네이비 배경 로고 카드 이미지 생성
      3. ffmpeg로 로고 카드 영상 생성 (ultrafast, 저메모리)
      4. ffmpeg 단일 호출로 메인 영상 페이드아웃 + 로고 카드 이어붙이기 (인코딩 1회)

    Returns:
        성공 시 output_path, 실패 시 None
    """
    if not shutil.which("ffmpeg"):
        logger.warning("[PostProcess] ffmpeg 없음 — outro 건너뜀")
        return None

    w, h, duration = _get_video_info(video_path)
    logger.info("[PostProcess] 영상 정보: %dx%d, %.1fs", w, h, duration)

    tmp_dir = os.path.join(os.path.dirname(output_path), "_outro_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        logo_path = ensure_transparent_logo(project_root)
        if not logo_path:
            logger.warning("[PostProcess] 로고 없음 — outro 건너뜀")
            return None
        card_img = _create_logo_card_image(logo_path, w, h, tmp_dir)
        if not card_img:
            return None

        card_video = os.path.join(tmp_dir, "logo_card.mp4")
        if not _create_logo_card_video(card_img, w, h, logo_card_duration, card_video):
            return None

        gc.collect()  # 로고 카드 ffmpeg 프로세스 메모리 OS 반환 유도

        fade_start = max(0.0, duration - fade_duration) if not skip_main_fade else duration
        if not _fadeout_and_concat(video_path, card_video, fade_start, fade_duration, output_path):
            return None

        logger.info(
            "[PostProcess] outro 완료 (%.1fs → %.1fs): %s",
            duration, duration + logo_card_duration, output_path,
        )
        return output_path

    finally:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
