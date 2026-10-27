"""
[팀 직접 구현] SadTalker 립싱크 어댑터.

단일 이미지 + 오디오 → 표정·눈 깜빡임·입 움직임이 자연스러운 립싱크 영상.
subprocess로 SadTalker/inference.py를 호출한다.

라이선스: 비상업 연구용 → 졸업작품 OK.
설치: git clone https://github.com/OpenTalker/SadTalker && pip install -r requirements.txt
      + python scripts/download_models.py (체크포인트 ~2GB)

CPU 기준 처리 시간: 30초 영상 ≈ 20~40분 (데모용 사전 생성 권장).
"""

import logging
import os
import subprocess

from .base import BaseLipSyncAdapter, LipSyncResult

logger = logging.getLogger(__name__)

_SADTALKER_DIR = os.environ.get("SADTALKER_DIR", "SadTalker")
_SADTALKER_CHECKPOINT = os.environ.get(
    "SADTALKER_CHECKPOINT",
    os.path.join(_SADTALKER_DIR, "checkpoints"),
)


class SadTalkerAdapter(BaseLipSyncAdapter):
    def adapter_name(self) -> str:
        return "sadtalker"

    def is_available(self) -> bool:
        script = os.path.join(_SADTALKER_DIR, "inference.py")
        checkpoint = _SADTALKER_CHECKPOINT
        return os.path.exists(script) and os.path.isdir(checkpoint)

    def generate(
        self,
        face_image_path: str,
        audio_path: str,
        output_path: str,
        original_video_path: str = "",
    ) -> LipSyncResult:
        script = os.path.join(_SADTALKER_DIR, "inference.py")
        if not os.path.exists(script):
            return LipSyncResult(
                success=False,
                error_message=f"SadTalker 미설치: {script} 없음. git clone 필요.",
                adapter=self.adapter_name(),
            )

        out_dir = os.path.dirname(output_path)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", script,
            "--driven_audio", audio_path,
            "--source_image", face_image_path,
            "--result_dir", out_dir,
            "--still",           # 몸 움직임 최소화 (홍보 영상에 적합)
            "--preprocess", "full",
            "--checkpoint_dir", _SADTALKER_CHECKPOINT,
        ]

        logger.info("[SadTalker] 립싱크 시작: %s + %s", face_image_path, audio_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,   # 30분 제한 (CPU 기준 충분)
            )
            if result.returncode != 0:
                logger.error("[SadTalker] 오류:\n%s", result.stderr[-500:])
                return LipSyncResult(
                    success=False,
                    error_message=result.stderr[-300:],
                    adapter=self.adapter_name(),
                )

            # SadTalker는 result_dir/{source_image_basename}.mp4 로 저장
            base = os.path.splitext(os.path.basename(face_image_path))[0]
            generated = os.path.join(out_dir, f"{base}.mp4")
            if os.path.exists(generated) and generated != output_path:
                os.rename(generated, output_path)

            if not os.path.exists(output_path):
                return LipSyncResult(
                    success=False,
                    error_message="SadTalker 완료됐으나 출력 파일 없음",
                    adapter=self.adapter_name(),
                )

            logger.info("[SadTalker] 완료: %s", output_path)
            return LipSyncResult(
                success=True,
                file_path=output_path,
                adapter=self.adapter_name(),
            )
        except subprocess.TimeoutExpired:
            return LipSyncResult(
                success=False,
                error_message="SadTalker 타임아웃 (30분 초과)",
                adapter=self.adapter_name(),
            )
        except Exception as e:
            return LipSyncResult(
                success=False,
                error_message=str(e),
                adapter=self.adapter_name(),
            )
