"""
Mock 어댑터: 실제 영상 생성 없이 스토리보드와 플레이스홀더를 반환.
API 키 없이 파이프라인 전체를 데모할 수 있게 해준다.
"""

import os
from datetime import datetime

from .base import BaseVideoGeneratorAdapter, VideoGenerationResult


class MockVideoGeneratorAdapter(BaseVideoGeneratorAdapter):
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        genre: str = "",
        image_url: str | None = None,
    ) -> VideoGenerationResult:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        storyboard = self._generate_storyboard(prompt, genre, image_url)

        filename = f"storyboard_{timestamp}.txt"
        file_path = os.path.join(self.upload_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(storyboard)

        return VideoGenerationResult(
            success=True,
            file_path=file_path,
            storyboard=storyboard,
            metadata={
                "adapter": "mock",
                "generated_at": timestamp,
                "genre": genre,
                "prompt_length": len(prompt),
                "image_url": image_url,
            },
        )

    def _generate_storyboard(
        self, prompt: str, genre: str, image_url: str | None
    ) -> str:
        mode = "Image-to-Video" if image_url else "Text-to-Video"
        image_line = f"시작 이미지: {image_url}" if image_url else "시작 이미지: 없음 (텍스트 기반 생성)"

        lines = [
            "=== AI 영상 스토리보드 ===",
            f"장르: {genre or '미지정'}",
            f"생성 모드: {mode}",
            f"생성 시각: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            image_line,
            "",
            "--- 영상 생성 프롬프트 ---",
            prompt,
            "",
            "--- 스토리보드 ---",
            "[ 장면 1 - 오프닝 ]",
            "  시각: 0:00 - 0:03",
            "  내용: 인트로 / 타이틀 카드",
            "  시각적 요소: 부드러운 페이드인, 배경음악 시작",
            "",
            "[ 장면 2 - 본문 ]",
            "  시각: 0:03 - 0:45",
            "  내용: 주요 내용 전달",
            "  시각적 요소: 프롬프트 기반 시각 자료",
            "",
            "[ 장면 3 - 클로징 ]",
            "  시각: 0:45 - 1:00",
            "  내용: 마무리 및 CTA",
            "  시각적 요소: 페이드아웃, 엔딩 카드",
            "",
            "--- 제작 노트 ---",
            "이 스토리보드를 바탕으로 Veo, RunwayML 등의",
            "AI 영상 생성 도구를 사용하여 실제 영상을 제작하세요.",
            "제작 후 '영상 업로드' 단계에서 파일을 업로드하거나 YouTube URL을 입력하세요.",
        ]
        return "\n".join(lines)
