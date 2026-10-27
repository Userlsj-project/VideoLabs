"""
[팀 직접 구현] Wav2Lip 립싱크 어댑터 (SadTalker 폴백).

이미지 또는 영상 + 오디오 → 입 부분만 교체하는 립싱크 영상.
SadTalker 설치 실패 시 자동으로 이 어댑터를 사용한다.

라이선스: MIT → 상업 사용 포함 제한 없음.
설치: git clone https://github.com/Rudrabha/Wav2Lip
      + checkpoints/wav2lip_gan.pth 수동 다운로드 (~400MB)

CPU 처리 시간: 30초 영상 ≈ 10~20분.
"""

import logging
import os
import subprocess

from .base import BaseLipSyncAdapter, LipSyncResult

logger = logging.getLogger(__name__)

_WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "Wav2Lip")
_WAV2LIP_CHECKPOINT = os.environ.get(
    "WAV2LIP_CHECKPOINT",
    os.path.join(_WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"),
)


class Wav2LipAdapter(BaseLipSyncAdapter):
    def adapter_name(self) -> str:
        return "wav2lip"

    def is_available(self) -> bool:
        script = os.path.join(_WAV2LIP_DIR, "inference.py")
        return os.path.exists(script) and os.path.exists(_WAV2LIP_CHECKPOINT)

    def generate(
        self,
        face_image_path: str,
        audio_path: str,
        output_path: str,
        original_video_path: str = "",
    ) -> LipSyncResult:
        script = os.path.join(_WAV2LIP_DIR, "inference.py")
        if not os.path.exists(script):
            return LipSyncResult(
                success=False,
                error_message=f"Wav2Lip 미설치: {script} 없음.",
                adapter=self.adapter_name(),
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "python", script,
            "--checkpoint_path", _WAV2LIP_CHECKPOINT,
            "--face", face_image_path,
            "--audio", audio_path,
            "--outfile", output_path,
        ]

        logger.info("[Wav2Lip] 립싱크 시작: %s + %s", face_image_path, audio_path)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,
            )
            if result.returncode != 0:
                logger.error("[Wav2Lip] 오류:\n%s", result.stderr[-500:])
                return LipSyncResult(
                    success=False,
                    error_message=result.stderr[-300:],
                    adapter=self.adapter_name(),
                )
            logger.info("[Wav2Lip] 완료: %s", output_path)
            return LipSyncResult(
                success=True,
                file_path=output_path,
                adapter=self.adapter_name(),
            )
        except subprocess.TimeoutExpired:
            return LipSyncResult(
                success=False,
                error_message="Wav2Lip 타임아웃 (20분 초과)",
                adapter=self.adapter_name(),
            )
        except Exception as e:
            return LipSyncResult(
                success=False,
                error_message=str(e),
                adapter=self.adapter_name(),
            )
