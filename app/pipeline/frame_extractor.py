"""
[팀 직접 구현] 영상 프레임 추출 유틸리티 (ffmpeg)

- 마지막 프레임: 프레임 체이닝용 (다음 클립 image_url에 전달)
- 첫 번째 프레임: Before/After 비교 또는 썸네일 대용
- concat: 멀티샷 클립 합치기 (재인코딩 없음)

thumbnail_generator.py와 동일한 subprocess 패턴 사용.
"""
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def is_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_last_frame(video_path: str, output_dir: str) -> str | None:
    """
    영상의 마지막 프레임을 JPG로 추출.
    반환: 추출된 이미지 경로, 또는 None(ffmpeg 없거나 실패).

    -sseof -1 : 파일 끝에서 1초 전부터 역방향 탐색
    -update 1 : 새 프레임마다 같은 파일 덮어쓰기 → 최종 프레임만 남음
    VFR(가변 프레임률) 영상에도 안전하게 동작.
    """
    if not is_ffmpeg_available():
        logger.warning("[FrameExtractor] ffmpeg 없음 — 프레임 체이닝 비활성화")
        return None
    if not os.path.exists(video_path):
        logger.warning("[FrameExtractor] 영상 파일 없음: %s", video_path)
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "last_frame.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-1",
        "-i", video_path,
        "-update", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, timeout=60)
        if ret.returncode == 0 and os.path.exists(output_path):
            logger.info("[FrameExtractor] 마지막 프레임 추출 완료: %s", output_path)
            return output_path
        logger.warning(
            "[FrameExtractor] 마지막 프레임 추출 실패 (returncode=%d): %s",
            ret.returncode,
            ret.stderr.decode(errors="ignore")[:200],
        )
    except subprocess.TimeoutExpired:
        logger.warning("[FrameExtractor] ffmpeg 타임아웃 (60s)")
    except Exception as e:
        logger.warning("[FrameExtractor] 예외: %s", e)
    return None


def extract_first_frame(video_path: str, output_dir: str) -> str | None:
    """첫 번째 프레임 추출 (Before/After 비교 또는 썸네일 대용)."""
    if not is_ffmpeg_available():
        return None
    if not os.path.exists(video_path):
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "first_frame.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path,
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, timeout=60)
        if ret.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception as e:
        logger.warning("[FrameExtractor] first_frame 추출 예외: %s", e)
    return None


def concat_video_files(video_paths: list, output_path: str) -> tuple[bool, str]:
    """
    ffmpeg concat demuxer로 클립 목록을 하나로 합치기.
    동일 코덱·해상도 전제 → -c copy (재인코딩 없음, 빠름).
    반환: (성공 여부, 결과 파일 경로)

    실패 시 video_paths[-1]을 반환해 graceful degradation.
    Windows 경로 역슬래시를 슬래시로 변환해 ffmpeg concat list 오류 방지.
    """
    if not is_ffmpeg_available():
        return False, video_paths[-1] if video_paths else ""
    if not video_paths:
        return False, ""
    if len(video_paths) == 1:
        return True, video_paths[0]

    # concat list 임시 파일 작성
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for p in video_paths:
                # 상대 경로 → 절대 경로 변환 (ffmpeg은 concat txt 위치 기준으로 상대 경로 해석)
                safe_path = os.path.abspath(p).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
            list_file = f.name
    except Exception as e:
        logger.error("[FrameExtractor] concat list 파일 생성 실패: %s", e)
        return False, video_paths[-1]

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path,
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, timeout=300)
        os.unlink(list_file)
        if ret.returncode == 0:
            logger.info("[FrameExtractor] concat 완료: %s (%d개 클립)",
                        output_path, len(video_paths))
            return True, output_path
        logger.error(
            "[FrameExtractor] concat 실패 (returncode=%d): %s",
            ret.returncode,
            ret.stderr.decode(errors="ignore")[:200],
        )
    except subprocess.TimeoutExpired:
        logger.warning("[FrameExtractor] concat 타임아웃 (300s)")
    except Exception as e:
        logger.error("[FrameExtractor] concat 예외: %s", e)
        try:
            os.unlink(list_file)
        except Exception:
            pass

    return False, video_paths[-1]
