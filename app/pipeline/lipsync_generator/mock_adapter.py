"""
[팀 직접 구현] 립싱크 Mock 어댑터.

SadTalker/Wav2Lip 미설치 환경에서 파이프라인이 중단되지 않도록
항상 success=True를 반환한다.

데모 동작:
  - 오디오가 있으면: output_path에 원본 영상을 복사해서 반환
    (실제 립싱크 없지만 파일은 존재)
  - 오디오가 없으면: original_video_path를 그대로 반환
"""

import os
import shutil

from .base import BaseLipSyncAdapter, LipSyncResult


class MockLipSyncAdapter(BaseLipSyncAdapter):
    def adapter_name(self) -> str:
        return "mock_lipsync"

    def generate(
        self,
        face_image_path: str,
        audio_path: str,
        output_path: str,
        original_video_path: str = "",  # mock 전용: 반환할 원본 영상
    ) -> LipSyncResult:
        # 원본 영상이 있으면 output_path로 복사 (파일이 존재하도록)
        src = original_video_path or face_image_path
        if src and os.path.exists(src) and src != output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(src, output_path)
                return LipSyncResult(
                    success=True,
                    file_path=output_path,
                    adapter=self.adapter_name(),
                )
            except Exception:
                pass

        return LipSyncResult(
            success=True,
            file_path=original_video_path or None,
            adapter=self.adapter_name(),
        )
