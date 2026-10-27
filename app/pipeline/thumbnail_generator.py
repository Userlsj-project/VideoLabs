"""
[팀 직접 구현] ffmpeg 기반 영상 썸네일 자동 추출기.

생성된 Veo 영상에서 대표 프레임을 자동 추출해 YouTube 업로드용 썸네일을 생성한다.
ffmpeg 미설치 시 빈 리스트 반환 (graceful fallback).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def extract_thumbnails(
    video_path: str,
    output_dir: str,
    count: int = 3,
    width: int = 1280,
) -> list[str]:
    """
    영상에서 균등 간격으로 썸네일 이미지를 추출한다.

    Args:
        video_path: 입력 영상 경로
        output_dir: 썸네일 저장 디렉토리
        count: 추출할 썸네일 수 (기본 3)
        width: 썸네일 너비 (높이는 비율 유지, 기본 1280px)

    Returns:
        추출된 이미지 파일 경로 목록
    """
    if not shutil.which("ffmpeg"):
        logger.warning("[Thumbnail] ffmpeg 미설치")
        return []

    if not os.path.exists(video_path):
        logger.error("[Thumbnail] 영상 파일 없음: %s", video_path)
        return []

    duration = _get_duration(video_path)
    if not duration or duration < 1.0:
        logger.warning("[Thumbnail] 영상 너무 짧거나 길이 조회 실패")
        return []

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    results = []

    # 양쪽 10% 제외하고 균등 간격 타임스탬프 계산 (도입·마무리 프레임 회피)
    margin = duration * 0.10
    span = duration - 2 * margin
    step = span / max(count - 1, 1)
    timestamps = [margin + i * step for i in range(count)]

    for idx, ts in enumerate(timestamps):
        ts = min(max(ts, 0.0), duration - 0.1)
        out_path = os.path.join(output_dir, f"{base}_thumb_{idx + 1}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "2",       # JPEG 품질 (1=최고, 2=고품질)
            out_path,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.exists(out_path):
                results.append(out_path)
        except Exception as exc:
            logger.error("[Thumbnail] 추출 실패 ts=%.1fs: %s", ts, exc)

    logger.info("[Thumbnail] %d개 추출 완료 → %s", len(results), output_dir)
    return results


def get_expected_thumbnail_paths(video_path: str, upload_folder: str, count: int = 3) -> list[str]:
    """영상 경로에서 예상 썸네일 파일 경로 목록을 반환한다 (파일 존재 여부 무관)."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    thumb_dir = os.path.join(upload_folder, "thumbnails")
    return [os.path.join(thumb_dir, f"{base}_thumb_{i}.jpg") for i in range(1, count + 1)]


def get_existing_thumbnails(video_path: str, upload_folder: str, count: int = 3) -> list[str]:
    """이미 추출된 썸네일 파일 경로 목록을 반환한다 (존재하는 파일만)."""
    return [p for p in get_expected_thumbnail_paths(video_path, upload_folder, count)
            if os.path.exists(p)]


def _get_duration(video_path: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return None
