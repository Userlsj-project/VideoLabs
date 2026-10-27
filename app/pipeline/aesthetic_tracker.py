"""
[팀 직접 구현] F-9: 채널 미적 일관성 지문 (Aesthetic Fingerprint).

영상에서 색상 지문을 추출하고, 채널 기준선과 비교해 브랜드 일관성을 측정한다.
Runway, CapCut, Adobe 어디에도 없는 크로스-영상 스타일 추적 기능.

opencv-python-headless 필요: pip install opencv-python-headless
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def extract_color_fingerprint(video_path: str, sample_frames: int = 10) -> dict | None:
    """
    영상에서 색상 지문을 추출한다.

    Returns:
        {avg_hue, avg_saturation, avg_brightness, color_temperature, dominant_palette}
        실패 시 None 반환.
    """
    if not is_opencv_available():
        logger.warning("[AestheticTracker] opencv 미설치 — 색상 지문 추출 불가")
        return None

    if not os.path.exists(video_path):
        return None

    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        step = max(1, total // sample_frames)
        all_pixels = []

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            # BGR → HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 다운샘플링 (64x64)으로 속도 향상
            small = cv2.resize(hsv, (64, 64))
            all_pixels.append(small.reshape(-1, 3))

        cap.release()

        if not all_pixels:
            return None

        pixels = np.vstack(all_pixels).astype(float)
        avg_hue = float(np.mean(pixels[:, 0]))
        avg_saturation = float(np.mean(pixels[:, 1]))
        avg_brightness = float(np.mean(pixels[:, 2]))

        # 색상 온도 분류 (Hue 기반)
        if avg_hue < 30 or avg_hue > 150:
            color_temp = "warm"    # 빨강~주황~노랑
        elif 90 <= avg_hue <= 150:
            color_temp = "cool"    # 청록~파랑
        else:
            color_temp = "neutral"

        # 밝기 분류
        brightness_norm = avg_brightness / 255.0
        if brightness_norm > 0.65:
            brightness_label = "bright"
        elif brightness_norm < 0.35:
            brightness_label = "dark"
        else:
            brightness_label = "balanced"

        return {
            "avg_hue": round(avg_hue, 1),
            "avg_saturation": round(avg_saturation / 255.0, 3),
            "avg_brightness": round(brightness_norm, 3),
            "color_temperature": color_temp,
            "brightness_label": brightness_label,
            "frame_count_analyzed": len(all_pixels),
        }

    except Exception as e:
        logger.error("[AestheticTracker] 색상 지문 추출 실패: %s", e)
        return None


def compare_fingerprints(baseline: dict | None, current: dict | None) -> dict:
    """
    두 색상 지문의 차이를 계산한다.

    Returns:
        {deviation_score, consistent, suggestions, details}
    """
    if not baseline or not current:
        return {"deviation_score": None, "consistent": None, "suggestions": []}

    hue_diff = abs(current.get("avg_hue", 0) - baseline.get("avg_hue", 0))
    # Hue는 원형(0~180), 180 넘으면 반대 방향이 더 짧음
    hue_diff = min(hue_diff, 180 - hue_diff)
    brightness_diff = abs(current.get("avg_brightness", 0) - baseline.get("avg_brightness", 0))
    saturation_diff = abs(current.get("avg_saturation", 0) - baseline.get("avg_saturation", 0))

    # 0~100 편차 점수 (가중 합산)
    deviation = min(100, hue_diff * 0.4 + brightness_diff * 100 * 0.4 + saturation_diff * 100 * 0.2)

    suggestions = []
    if current.get("color_temperature") != baseline.get("color_temperature"):
        suggestions.append(
            f"색상 온도가 '{baseline['color_temperature']}'에서 '{current['color_temperature']}'으로 변경됨. "
            f"프롬프트에 '{baseline['color_temperature']} color tone' 추가를 고려하세요."
        )
    if brightness_diff > 0.15:
        direction = "밝아짐" if current["avg_brightness"] > baseline["avg_brightness"] else "어두워짐"
        suggestions.append(
            f"밝기가 {direction} (차이: {round(brightness_diff * 100)}%). "
            f"프롬프트에 '{baseline.get('brightness_label', 'balanced')} lighting' 명시를 고려하세요."
        )

    return {
        "deviation_score": round(deviation, 1),
        "consistent": deviation < 20,
        "suggestions": suggestions,
        "details": {
            "hue_diff": round(hue_diff, 1),
            "brightness_diff": round(brightness_diff * 100, 1),
            "saturation_diff": round(saturation_diff * 100, 1),
        },
    }


def compute_session_baseline(session_rounds: list, upload_folder: str) -> dict | None:
    """
    세션의 첫 N개 라운드 영상에서 기준선 색상 지문을 계산한다.
    """
    fingerprints = []
    for rnd in session_rounds[:3]:  # 최대 3개 라운드로 기준선 설정
        if not rnd.video_file_path:
            continue
        vpath = rnd.video_file_path
        if not os.path.isabs(vpath):
            vpath = os.path.normpath(os.path.join(upload_folder, vpath))
        fp = extract_color_fingerprint(vpath)
        if fp:
            fingerprints.append(fp)

    if not fingerprints:
        return None

    # 평균 지문 계산
    import numpy as np
    avg = {
        "avg_hue": float(np.mean([f["avg_hue"] for f in fingerprints])),
        "avg_saturation": float(np.mean([f["avg_saturation"] for f in fingerprints])),
        "avg_brightness": float(np.mean([f["avg_brightness"] for f in fingerprints])),
    }

    # 지배적 색상 온도
    temps = [f["color_temperature"] for f in fingerprints]
    avg["color_temperature"] = max(set(temps), key=temps.count)

    bright_labels = [f.get("brightness_label", "balanced") for f in fingerprints]
    avg["brightness_label"] = max(set(bright_labels), key=bright_labels.count)
    avg["based_on_rounds"] = len(fingerprints)

    return avg
