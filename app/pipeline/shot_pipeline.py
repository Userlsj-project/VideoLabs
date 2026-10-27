"""
[팀 직접 구현] 멀티샷 시퀀스 생성 오케스트레이터 (프레임 체이닝)

흐름:
  샷N 생성 완료 → ffmpeg last_frame 추출
  → 샷N+1 image_url에 로컬 경로 전달
  → Kling image-to-video 엔드포인트로 시각적 연속성 유지
  → 전체 완료 후 ffmpeg concat
"""
import logging
import os
from dataclasses import dataclass, field

from .frame_extractor import concat_video_files, extract_last_frame

logger = logging.getLogger(__name__)


@dataclass
class Shot:
    """단일 샷 정의."""
    shot_num: int
    prompt: str
    camera: str = "medium shot"
    duration_note: str = ""
    # 생성 후 채워지는 필드
    video_path: str = ""
    last_frame_path: str = ""
    status: str = "pending"   # pending | generating | done | failed
    error: str = ""


@dataclass
class ShotSequenceResult:
    success: bool
    final_video_path: str = ""
    shots: list = field(default_factory=list)
    error_message: str = ""


def generate_shot_sequence(
    shots: list[Shot],
    adapter,
    output_dir: str,
    session_id: str,
    round_id: int,
) -> ShotSequenceResult:
    """
    샷 목록을 프레임 체이닝으로 순차 생성 후 concat.

    - 각 샷 생성 완료 시 ffmpeg로 마지막 프레임 추출
    - 다음 샷 image_url에 해당 경로를 전달 (Veo base64 인라인)
    - 실패한 샷에서 중단 → 지금까지 성공한 클립으로 concat (graceful degradation)
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_paths: list[str] = []
    prev_last_frame_path: str | None = None

    for shot in shots:
        shot.status = "generating"
        logger.info("[ShotPipeline] 샷 %d/%d 생성 시작: %s",
                    shot.shot_num, len(shots), shot.prompt[:60])

        result = adapter.generate(
            prompt=shot.prompt,
            image_url=prev_last_frame_path,   # None이면 text-to-video, 있으면 image-to-video 자동 전환
        )

        if not result.success:
            shot.status = "failed"
            shot.error = result.error_message or "알 수 없는 오류"
            logger.error("[ShotPipeline] 샷 %d 실패: %s", shot.shot_num, shot.error)
            break

        shot.video_path = result.file_path
        shot.status = "done"
        generated_paths.append(result.file_path)

        # 다음 샷을 위해 마지막 프레임 추출
        frame_dir = os.path.join(output_dir, f"frames_shot{shot.shot_num}")
        last_frame = extract_last_frame(result.file_path, frame_dir)
        if last_frame:
            shot.last_frame_path = last_frame
            prev_last_frame_path = last_frame
        else:
            prev_last_frame_path = None   # 추출 실패 시 다음 샷은 체이닝 없이 생성

    if not generated_paths:
        return ShotSequenceResult(
            success=False,
            shots=shots,
            error_message="생성된 클립 없음",
        )

    if len(generated_paths) == 1:
        return ShotSequenceResult(
            success=True,
            final_video_path=generated_paths[0],
            shots=shots,
        )

    output_path = os.path.join(output_dir, f"sequence_s{session_id}_r{round_id}.mp4")
    ok, final_path = concat_video_files(generated_paths, output_path)
    return ShotSequenceResult(
        success=ok,
        final_video_path=final_path,
        shots=shots,
        error_message="" if ok else "concat 실패",
    )
