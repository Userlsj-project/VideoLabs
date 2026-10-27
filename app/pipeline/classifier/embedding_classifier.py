"""
[팀 직접 설계·구현] 임베딩 기반 다국어 피드백 분류기 (v2).

HuggingFace 계정 없이 동작하는 sentence-transformers 임베딩 모델만 사용.
(paraphrase-multilingual-MiniLM-L12-v2 — 인증 불필요)

분류 방식:
  1. [오픈소스] 다국어 문장 임베딩 생성
  2. [팀 직접 설계] 원형 문장(prototype) 코사인 유사도로 1차 레이블 판단
  3. [팀 직접 설계] 구체성·타당성 키워드 패턴으로 2축 점수 산정
  4. [팀 직접 설계] 심각도·긴급도·품질 추가 3축 점수 산정
  5. [팀 직접 설계] 복합 시그널로 최종 레이블 결정
  6. [팀 직접 설계] 멀티레이블 토픽 태깅 (하나의 댓글 → 여러 카테고리)
  7. [팀 직접 설계] 동적 프로토타입 업데이트: 승인된 피드백으로 점진적 학습
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from .base import (
    CONTENT_SCOPES,
    TECHNICAL_SCOPES,
    TOPIC_CATEGORIES,
    BaseClassifier,
    ClassificationResult,
    topic_to_scope,
)

logger = logging.getLogger(__name__)

# ── [팀 직접 설계] 원형 문장 정의 ────────────────────────────────────────────
# 각 레이블을 대표하는 문장들. 다국어 임베딩 공간에서 가장 가까운 원형으로 분류.

_PROTOTYPES: dict[str, list[str]] = {
    "constructive": [
        "Please improve this specific aspect of the video.",
        "The background music is too loud, please lower the volume.",
        "The subtitles are too small, please increase the font size.",
        "The pacing is too fast, please slow down the scene transitions.",
        "배경음악이 너무 커서 목소리가 안 들려요. 볼륨을 줄여주세요.",
        "자막 글씨가 너무 작아서 읽기 힘들어요. 크게 해주세요.",
        "영상 속도가 너무 빠릅니다. 장면 전환을 천천히 해주세요.",
        "스토리 전개가 갑작스러워서 이해하기 어려웠어요. 설명을 더 추가해주세요.",
        "조명이 너무 어두워서 얼굴이 잘 안 보여요. 밝게 해주세요.",
        "This is actionable feedback with a clear improvement suggestion.",
        "I suggest changing X to Y because of Z reason.",
        "Please add English subtitles for international viewers.",
        "Please include more details in the description.",
        "You should add transitions between scenes.",
        "It would be better if you added background music.",
        "영어 자막을 추가해주세요. 외국 시청자들을 위해서요.",
        "내용을 더 추가해주시면 좋겠어요.",
        "이 부분이 이해하기 어려워서 설명을 추가해 주셨으면 합니다.",
        "색감이 너무 칙칙해서 밝게 보정해 주세요.",
        "배우 목소리가 음악에 묻혀서 잘 안 들립니다. 음악 볼륨을 낮춰주세요.",
    ],
    "non_constructive": [
        "This video is okay.",
        "Nice video, I liked it.",
        "Not bad but could be better.",
        "This video is not good.",
        "재밌어요.",
        "좋아요.",
        "별로예요.",
        "ㅋㅋㅋ",
        "괜찮네요.",
        "I don't like this.",
        "This is just my opinion with no specific suggestion.",
        "기대 이하였어요.",
        "그냥 그래요.",
        "최고예요!",
    ],
    "spam": [
        "Subscribe to my channel for free giveaway.",
        "Click here for amazing deals.",
        "First comment!",
        "구독 좋아요 눌러주세요 제 채널도 와주세요.",
        "광고입니다. 홍보글입니다.",
        "This is completely irrelevant spam content.",
        "1111111111",
        "aaaaaaaaaa",
    ],
    "inappropriate": [
        "This content is offensive and should be removed.",
        "This video contains hateful language.",
        "욕설이 포함된 부적절한 내용입니다.",
        "혐오 발언, 폭력적 내용이 포함되어 있습니다.",
    ],
}

# ── [팀 직접 설계] 구체성 키워드 패턴 ───────────────────────────────────────

_SPECIFICITY_SIGNALS: list[tuple[str, float]] = [
    (r"\d+\s*(?:초|분|px|배|db|dB|번|회|%|점|단계|fps|hz|Hz)", 0.35),
    (r"(?:해주세요|해줘|줄여|늘려|바꿔|고쳐|수정해|개선해|추가해|삭제해|없애|바꿔주|높여|낮춰|키워|줄이)", 0.35),
    (r"\b(?:please|should|could|would|suggest|recommend|improve|fix|change|add|remove|increase|decrease|reduce|lower|raise)\b", 0.30),
    (r"\bplease\s+(?:add|include|make|use|change|fix|improve|reduce|increase|lower|raise|remove|put|show|provide|consider)\b", 0.20),
    (r"(?:너무|매우|굉장히|심하게|훨씬|조금|좀 더|too|very|much|way too|a bit|slightly)", 0.20),
    (r"(?:자막|배경음악|bgm|효과음|화질|조명|글씨|폰트|색감|구도|스토리|내레이션|인트로|아웃트로|편집|속도|템포|볼륨|음량|해상도|프레임)", 0.25),
    (r"\b(?:subtitle|caption|background music|bgm|sound effect|lighting|color|font|resolution|intro|outro|editing|volume|pacing|tempo|narration|frame rate|fps)\b", 0.25),
    (r"(?:필요합니다|필요해요|필요해|필요한|필요가|부족합니다|부족해|부족한|아쉬워요|아쉽습니다|아쉽네요)", 0.35),
    (r"(?:개선이|수정이|변경이|보완이|추가가|삭제가)\s*(?:필요|됐으면|되면)", 0.30),
    (r"(?:왜냐하면|때문에|그래서|그러므로|결과적으로|because|therefore|so that|as a result|since)", 0.15),
    # 비교 표현: 이전 영상과 비교
    (r"(?:이전|지난번|예전|전 영상|last video|previous|before)", 0.15),
    # 한국어 제안 표현 ("하시면 좋을것 같", "넣으면 좋을거", "어떨까요" 등)
    (r"(?:좋을것 같|좋을 것 같|더 좋을|더 나을|나을것 같|어떨까요|하면 어때|했으면 좋겠|하셨으면|넣으면 좋|넣어주|넣어 주)", 0.30),
    # 구체적 요소 지목 (목소리·로고·자막 등 특정 요소 언급 자체가 구체성)
    (r"(?:목소리|음성|나레이션|내레이션|보이스|성우|ai음성|ai 음성)", 0.20),
    (r"(?:로고|상표|워터마크|엠블럼|학과 로고|학교 로고|브랜드)", 0.20),
]

# ── [팀 직접 설계] 타당성 키워드 패턴 ───────────────────────────────────────

_VALIDITY_SIGNALS: list[tuple[str, float]] = [
    (r"(?:때문에|왜냐하면|그래서|이유는|원인은|결과적으로|해서|라서)", 0.30),
    (r"\b(?:because|since|due to|as a result|therefore|thus|hence|so|reason)\b", 0.30),
    (r"(?:안 들려|안 보여|이해하기 어려|불편|힘들|어려워|못|읽기 힘|집중이 안)", 0.20),
    (r"\b(?:can't|cannot|hard to|difficult to|unable to|distract|annoying|confusing|unclear)\b", 0.20),
    (r"(?:항상|매번|계속|반복|자주|sometimes|always|often|every time|repeatedly)", 0.15),
    (r"(?:시청자|관객|사용자|viewer|audience|user|we|us|people)\b", 0.10),
    (r"(?:경험상|개인적으로|직접|in my experience|personally|from my perspective)", 0.15),
    # 결과/영향 서술
    (r"(?:몰입이 깨|집중이 안|보기 힘|볼 수가 없|따라가기 어|이해가 안)", 0.25),
    (r"\b(?:immersion|distraction|focus|hard to follow|can't understand|lose track)\b", 0.20),
    # 관찰형 타당성 — 문제점을 직접 묘사 (이유가 내포됨)
    (r"(?:티가 나|티나네|티가나|어색하|부자연스|느끼하|인공적|로봇 같|ai같|ai 같|인공지능 같)", 0.25),
    (r"(?:같은게|같은 게|처럼 들려|처럼 보여|처럼 느껴)", 0.15),
    # 암시적 비교 ("녹음으로 하시면", "직접 찍으면" 등)
    (r"(?:녹음으로|직접 목소리|사람 목소리|실제 목소리)", 0.20),
]

# ── [팀 직접 설계] 심각도 패턴 (v2 신규) ─────────────────────────────────────
# 문제가 얼마나 치명적인가 (0=경미, 1=시청 불가)

_SEVERITY_SIGNALS: list[tuple[str, float]] = [
    # 치명적: 시청/이해 완전 불가
    (r"(?:아예|전혀|완전히|절대)\s*(?:안|못|불가)", 0.55),
    (r"(?:못 보겠|볼 수가 없|들을 수가 없|이해 불가|이해가 전혀 안)", 0.55),
    (r"\b(?:can't watch|impossible to watch|completely unwatchable|totally unable)\b", 0.50),
    (r"(?:눈이 아프|귀가 아프|머리가 아프|시청이 불가)", 0.45),
    # 심각: 매우 방해됨
    (r"(?:너무 심하|굉장히 거슬|매우 불편|심각하게|really annoying|very distracting|way too)\b", 0.30),
    (r"(?:계속 신경 쓰여|집중을 전혀|몰입이 깨져|keeps breaking)", 0.30),
    # 경미: 약간의 문제 (점수 낮춤)
    (r"(?:조금|살짝|약간|좀\s*더|a bit|slightly|a little|minor|just a tad)", -0.10),
]

# ── [팀 직접 설계] 긴급도 패턴 (v2 신규) ─────────────────────────────────────
# 즉각 처리가 필요한가

_URGENCY_SIGNALS: list[tuple[str, float]] = [
    (r"(?:꼭|반드시|제발|빨리|urgently)\b", 0.40),
    (r"\bplease please\b|제발\s*제발", 0.35),
    (r"(?:다음 영상부터|앞으로는|from now on|next video|every video)", 0.25),
    # 감탄부호 3개 이상 = 강한 강조
    (r"[!！]{3,}", 0.20),
    # 대문자 단어 4글자 이상 (영문 강조)
    (r"\b[A-Z]{4,}\b", 0.15),
    # 개인적 불편함 강조
    (r"(?:이거 때문에|이래서|때문에 영상을|this is why|that's why|stopped watching)", 0.20),
]

# ── [팀 직접 설계] 스팸·부적절 패턴 ─────────────────────────────────────────

_SPAM_PATTERNS: list[str] = [
    r"(?:구독|좋아요|알림|subscribe|like|follow|check out|click|giveaway|free|할인|이벤트)",
    r"(?:홍보|광고|ad\b|sponsor|affiliated)",
    r"^[가-힣a-z\s!?ㅋㅎ]{1,5}$",
    r"(.)\1{4,}",
    r"^(?:first|1st|ㅋ+|ㅎ+|ㅠ+|wow+|lol+|omg+)[\s!]*$",
]

_INAPPROPRIATE_PATTERNS: list[str] = [
    r"(?:시발|씨발|개새끼|병신|미친|fuck|shit|bitch|asshole|bastard|hate|kill|die)",
    r"(?:혐오|차별|인종|성희롱|폭력|욕설)",
]

# ── [팀 직접 설계] 주제 카테고리 키워드 ─────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "audio": [
        "음악", "배경음", "bgm", "효과음", "볼륨", "음량", "소리", "목소리", "노이즈",
        "music", "bgm", "sound", "volume", "audio", "noise", "voice", "loud", "quiet",
        "bass", "treble", "echo", "reverb", "믹싱", "mixing",
    ],
    "subtitle": [
        "자막", "글씨", "폰트", "가독성", "텍스트", "caption",
        "subtitle", "caption", "font", "text", "readability", "readable",
        "타이포그라피", "typography", "글자 크기", "자막 크기",
    ],
    "pacing": [
        "속도", "빠르", "느리", "템포", "리듬", "편집", "전환", "장면", "컷",
        "speed", "fast", "slow", "tempo", "pace", "pacing", "cut", "transition",
        "editing", "montage", "timing", "시간",
    ],
    "visual": [
        "화질", "조명", "색감", "구도", "밝기", "색상", "필터", "카메라", "해상도",
        "lighting", "color", "colour", "resolution", "bright", "dark", "visual",
        "camera", "angle", "composition", "grade", "grading", "4k", "hd",
    ],
    "story": [
        "스토리", "이야기", "캐릭터", "인물", "내러티브", "전개", "플롯", "결말", "복선",
        "story", "character", "narrative", "plot", "ending", "beginning", "arc",
        "drama", "scene", "dialogue", "대사",
    ],
    "content": [
        "내용", "정보", "구성", "주제", "설명", "소개", "메시지", "정확",
        "content", "information", "topic", "explanation", "structure", "message",
        "accuracy", "factual",
    ],
    "technical": [
        "기술", "오류", "버그", "끊김", "로딩", "화면", "렌더링",
        "technical", "bug", "error", "glitch", "lag", "crash", "screen",
        "render", "compression", "artifact", "블러", "blur",
    ],
}


class EmbeddingClassifier(BaseClassifier):
    """
    [팀 직접 구현] sentence-transformers 임베딩 + 원형 유사도 + 규칙 기반 다축 판단.

    v2 추가:
      - 심각도(severity) · 긴급도(urgency) · 품질(quality) 스코어 산정
      - 멀티레이블 토픽 태깅 (최대 3개 카테고리)
      - 동적 프로토타입 업데이트 (승인 피드백으로 온라인 학습)
      - 개선된 언어 감지 (일본어·중국어·혼합어 처리)
      - 점수 기반 개선 제안 추출
    """

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        specificity_threshold: float = 0.35,
        validity_threshold: float = 0.30,
        prototypes_path: str | None = None,
    ):
        self.embedding_model_name = embedding_model
        self.specificity_threshold = specificity_threshold
        self.validity_threshold = validity_threshold
        self.prototypes_path = prototypes_path  # 학습 결과 저장 경로
        self._model = None
        self._prototype_embeddings: dict[str, np.ndarray] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # [오픈소스]
            logger.info("임베딩 분류기 모델 로드: %s", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
            self._precompute_prototypes()
        return self._model

    def _precompute_prototypes(self):
        """[팀 직접 구현] 원형 문장 임베딩 사전 계산 (1회만 수행)."""
        model = self._model
        for label, sentences in _PROTOTYPES.items():
            embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            self._prototype_embeddings[label] = centroid
        logger.debug("원형 임베딩 사전 계산 완료: %s", list(self._prototype_embeddings.keys()))

        # 이전 세션에서 학습된 프로토타입이 있으면 덮어쓰기
        if self.prototypes_path:
            self.load_prototypes_if_exists(self.prototypes_path)

    def update_prototype(self, label: str, new_texts: list[str], weight: float = 0.1) -> None:
        """
        [팀 직접 구현] 승인된 피드백으로 프로토타입을 온라인 업데이트.

        weight: 새 데이터의 영향력 (0.1 = 기존 90% 유지 — 점진적 적응)
        라운드가 쌓일수록 이 프로젝트의 피드백 스타일에 자동 최적화됨.
        """
        if not new_texts or label not in self._prototype_embeddings:
            return

        model = self._get_model()
        new_embs = model.encode(new_texts, normalize_embeddings=True, show_progress_bar=False)
        new_centroid = np.mean(new_embs, axis=0)
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid /= norm

        # 지수 이동 평균으로 점진적 업데이트
        old = self._prototype_embeddings[label]
        updated = (1 - weight) * old + weight * new_centroid
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated /= norm

        self._prototype_embeddings[label] = updated
        logger.info("프로토타입 업데이트: label=%s, n_examples=%d", label, len(new_texts))

        # 업데이트 즉시 저장 (서버 재시작 후에도 학습 유지)
        self.save_prototypes()

    def save_prototypes(self, path: str | None = None) -> None:
        """[팀 직접 구현] 현재 프로토타입 임베딩을 .npz 파일로 저장."""
        save_path = path or self.prototypes_path
        if not save_path or not self._prototype_embeddings:
            return
        try:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            np.savez_compressed(save_path, **self._prototype_embeddings)
            logger.debug("프로토타입 저장: %s", save_path)
        except Exception as e:
            logger.warning("프로토타입 저장 실패: %s", e)

    def load_prototypes_if_exists(self, path: str) -> bool:
        """[팀 직접 구현] 저장된 프로토타입 파일이 있으면 불러온다."""
        import os
        npz_path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(npz_path):
            return False
        try:
            data = np.load(npz_path)
            loaded = 0
            for label in data.files:
                if label in self._prototype_embeddings:
                    self._prototype_embeddings[label] = data[label]
                    loaded += 1
            logger.info("학습된 프로토타입 로드: %d개 레이블 (%s)", loaded, npz_path)
            return loaded > 0
        except Exception as e:
            logger.warning("프로토타입 로드 실패: %s", e)
            return False

    def classify(self, texts: list[str]) -> list[ClassificationResult]:
        """[팀 직접 구현] 텍스트 목록을 분류한다."""
        if not texts:
            return []

        model = self._get_model()

        # [오픈소스] 다국어 문장 임베딩
        text_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        results = []
        for i, text in enumerate(texts):
            result = self._classify_one(text, text_embs[i])
            results.append(result)

        return results

    def _classify_one(self, text: str, emb: np.ndarray) -> ClassificationResult:
        """[팀 직접 구현] 단일 텍스트 분류 — 6축 평가."""
        lower = text.lower()

        # ── 1단계: 부적절·스팸 빠른 필터 ────────────────────────────────
        for pat in _INAPPROPRIATE_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                return ClassificationResult(
                    constructive_label="inappropriate",
                    topic_category="other",
                    scope="content",
                    specificity_score=0.0,
                    validity_score=0.0,
                    confidence=0.95,
                    severity_score=0.0,
                    urgency_score=0.0,
                    quality_score=0.0,
                    detected_language=_detect_lang(text),
                    topic_categories_multi=[("other", 0.0)],
                )

        spam_score = sum(1 for pat in _SPAM_PATTERNS if re.search(pat, lower, re.IGNORECASE))
        if spam_score >= 2 or (
            len(text.strip()) <= 5
            and not re.search(r"[a-zA-Z가-힣].*[a-zA-Z가-힣]", text)
        ):
            return ClassificationResult(
                constructive_label="spam",
                topic_category="other",
                scope="content",
                specificity_score=0.0,
                validity_score=0.0,
                confidence=0.85,
                severity_score=0.0,
                urgency_score=0.0,
                quality_score=0.0,
                detected_language=_detect_lang(text),
                topic_categories_multi=[("other", 0.0)],
            )

        # ── 2단계: 원형 코사인 유사도 → 1차 레이블 ────────────────────
        similarities: dict[str, float] = {
            label: float(np.dot(emb, proto_emb))
            for label, proto_emb in self._prototype_embeddings.items()
        }
        top_label = max(similarities, key=lambda k: similarities[k])

        # ── 3단계: 구체성·타당성 2축 규칙 점수 ──────────────────────────
        specificity = _score_signals(lower, _SPECIFICITY_SIGNALS)
        validity = _score_signals(lower, _VALIDITY_SIGNALS)

        # 길이 보정
        length_bonus = min(len(text) / 120, 0.20)
        specificity = min(specificity + length_bonus * 0.5, 1.0)
        validity = min(validity + length_bonus * 0.4, 1.0)

        # ── 4단계: 심각도·긴급도 2축 규칙 점수 (신규) ────────────────────
        severity = _score_signals(lower, _SEVERITY_SIGNALS)
        severity = max(severity, 0.0)   # 음수 방지 (경미 패턴 -값 적용 후)
        urgency = _score_signals(lower, _URGENCY_SIGNALS)

        # ── 5단계: 최종 constructive 여부 결정 ──────────────────────────
        proto_constructive = similarities.get("constructive", 0)
        proto_non = similarities.get("non_constructive", 0)

        constructive_signal = (
            0.50 * proto_constructive
            + 0.30 * specificity
            + 0.20 * validity
        )
        non_constructive_signal = 0.50 * proto_non + 0.50 * (1 - specificity)

        if top_label == "spam":
            final_label = "spam"
            confidence = similarities[top_label]
        elif top_label == "inappropriate":
            final_label = "inappropriate"
            confidence = similarities[top_label]
        elif constructive_signal > non_constructive_signal and (
            specificity >= self.specificity_threshold
            or validity >= self.validity_threshold
        ):
            final_label = "constructive"
            confidence = constructive_signal
        elif (
            specificity >= self.specificity_threshold
            and validity >= self.validity_threshold
        ):
            final_label = "constructive"
            confidence = (specificity + validity) / 2
        else:
            final_label = "non_constructive"
            confidence = non_constructive_signal

        # 신뢰도가 낮으면 수동 검토 플래그
        signal_gap = abs(constructive_signal - non_constructive_signal)
        needs_review = (
            signal_gap < 0.05
            and final_label in ("constructive", "non_constructive")
        )

        # ── 6단계: 품질 점수 종합 ─────────────────────────────────────
        quality = _compute_quality_score(text, specificity, validity, severity)

        # ── 7단계: 멀티레이블 토픽 태깅 ──────────────────────────────
        topics_multi = _detect_topics_multi(lower)
        primary_topic = topics_multi[0][0] if topics_multi else "other"
        scope = topic_to_scope(primary_topic)
        lang = _detect_lang(text)

        suggestion = _extract_suggestion(text) if final_label == "constructive" else None

        return ClassificationResult(
            constructive_label=final_label,
            topic_category=primary_topic,
            scope=scope,
            specificity_score=round(specificity, 4),
            validity_score=round(validity, 4),
            confidence=round(confidence, 4),
            detected_language=lang,
            improvement_suggestion=suggestion,
            severity_score=round(min(severity, 1.0), 4),
            urgency_score=round(min(urgency, 1.0), 4),
            quality_score=quality,
            topic_categories_multi=topics_multi,
            needs_review=needs_review,
        )


# ── 헬퍼 함수들 ─────────────────────────────────────────────────────────────

def _score_signals(text: str, signals: list[tuple[str, float]]) -> float:
    """[팀 직접 구현] 패턴 매칭으로 점수 산정."""
    score = 0.0
    for pattern, weight in signals:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
    # 음수 포함 가능 (경미 패턴), clamp 0~1
    return float(np.clip(score, 0.0, 1.0))


def _detect_topics_multi(text: str, top_k: int = 3, min_score: float = 0.03) -> list[tuple[str, float]]:
    """
    [팀 직접 구현] 멀티레이블 토픽 감지.

    Returns: [(topic, normalized_score), ...] 내림차순, 최대 top_k개.
    """
    scores: dict[str, float] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in text)
        if hit > 0:
            scores[topic] = hit / len(keywords)  # 키워드 커버리지 비율

    sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [(t, round(s, 4)) for t, s in sorted_topics[:top_k] if s >= min_score]

    return result if result else [("other", 0.0)]


def _compute_quality_score(
    text: str,
    specificity: float,
    validity: float,
    severity: float,
) -> float:
    """
    [팀 직접 구현] 피드백 품질 종합 점수 (0~1).

    구체성·타당성·심각도의 가중 합산 + 길이·수치·비교 표현 보너스.
    클러스터 대표 텍스트 선택 및 우선순위 산정에 활용.
    """
    length = len(text)
    if length < 10:
        length_score = 0.1
    elif length < 20:
        length_score = 0.3
    elif length <= 200:
        length_score = min(length / 80, 1.0)
    else:
        # 너무 긴 텍스트는 약간 페널티
        length_score = max(0.5, 1.0 - (length - 200) / 600)

    # 수치 포함 (구체적 피드백 시그널)
    number_bonus = 0.15 if re.search(r'\d+', text) else 0.0

    # 이전 대비 비교 표현
    comparison_bonus = 0.10 if re.search(
        r'(?:이전|예전|전에|지난번|전 영상|compared to|than before|used to|was better)',
        text, re.IGNORECASE
    ) else 0.0

    quality = (
        0.30 * specificity
        + 0.25 * validity
        + 0.15 * severity
        + 0.15 * length_score
        + number_bonus
        + comparison_bonus
    )
    return round(min(quality, 1.0), 4)


def _detect_lang(text: str) -> str:
    """
    [팀 직접 구현] 다국어 감지 (한국어·영어·일본어·중국어·혼합어).

    단순 글자 수 비교에서 스크립트 비율 기반 감지로 개선.
    """
    korean = len(re.findall(r"[가-힣]", text))
    english = len(re.findall(r"[a-zA-Z]", text))
    japanese = len(re.findall(r"[぀-ゟ゠-ヿ]", text))
    chinese = len(re.findall(r"[一-鿿]", text))

    total = korean + english + japanese + chinese
    if total == 0:
        return "unknown"

    ratios = {
        "ko": korean / total,
        "en": english / total,
        "ja": japanese / total,
        "zh": chinese / total,
    }

    dominant = max(ratios, key=ratios.get)
    # 지배적 언어가 70% 미만이면 혼합어
    if ratios[dominant] < 0.70:
        top2 = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:2]
        return f"mixed_{top2[0][0]}_{top2[1][0]}"

    return dominant


def _extract_suggestion(text: str) -> str:
    """
    [팀 직접 구현] 텍스트에서 핵심 개선 제안 문장 추출 (점수 기반).

    단순 첫 번째 매칭 → 모든 문장 점수화 후 최고 점수 반환으로 개선.
    """
    sentences = [s.strip() for s in re.split(r"[.!?。\n]+", text) if s.strip()]

    if not sentences:
        return text[:120].strip()

    if len(sentences) == 1:
        return sentences[0][:120]

    _SUGGESTION_PATTERNS = [
        (r"(?:해주세요|해줘|줄여|늘려|바꿔|수정해|개선해|추가해)", 3.0),
        (r"\b(?:please|should|could|suggest|recommend|improve|fix|change|add|remove)\b", 2.5),
        (r"(?:필요합니다|필요해요|아쉬워요|부족합니다)", 2.0),
        (r"(?:would be better|it would|I think|이렇게 하면)", 1.5),
        (r"\d+", 1.0),          # 수치 포함
        (r"(?:때문에|because|since)", 0.5),  # 근거 제시
    ]

    def score(sent: str) -> float:
        s = 0.0
        lower = sent.lower()
        for pat, weight in _SUGGESTION_PATTERNS:
            if re.search(pat, lower):
                s += weight
        s += min(len(sent) / 30, 1.0)   # 길이 보너스 (최대 1.0)
        return s

    best = max(sentences, key=score)
    return best[:120]
