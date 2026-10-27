# FrameLoop — AI 영상 피드백 자동 개선 시스템

> **군산대학교 임베디드소프트웨어학과 졸업작품**  
> Flask · SQLite · Kling V3 · ElevenLabs · Gemini

---

## 🎬 인터랙티브 파이프라인 블럭도

> 아래 링크를 클릭하면 **14단계 애니메이션 블럭도**가 새 탭에서 열립니다.

**[▶ FrameLoop 파이프라인 블럭도 보기](https://htmlpreview.github.io/?https://github.com/Userlsj-project/VideoLabs/blob/main/diagram.html)**

---

## 1. 프로젝트 개요

**FrameLoop**는 AI로 생성한 유튜브 영상의 시청자 댓글을 자동 분석하고, 그 피드백을 다음 라운드 영상에 자동 반영하는 **자동 피드백 루프 시스템**입니다.

```
시청자 댓글 → AI 분류 → 클러스터링 → 사용자 승인 → 프롬프트 자동 수정 → 새 영상 생성 → 반복
```

| 항목 | 내용 |
|------|------|
| **차별점** | 단순 영상 생성이 아닌 "피드백 DNA" — 라운드가 쌓일수록 영상이 시청자 취향에 최적화 |
| **서버** | Flask 단일 서버 (포트 5000), SQLite WAL 모드 |
| **영상 생성** | fal.ai Kling V3 Standard — 22초 기준 약 $1.85 |
| **오디오** | ElevenLabs TTS(나레이션) + Sound(BGM) — 무료 플랜 |
| **프롬프트 엔진** | Gemini `gemini-3.1-flash-lite` (결제 미연결 프로젝트) |

---

## 2. 전체 파이프라인

**① 영상 제작**

```mermaid
flowchart LR
    U([👤 사용자\n대본 입력]) --> F[Flask 서버]
    F --> LLM[🤖 Gemini LLM\n프롬프트 생성]
    LLM --> K[🎥 Kling V3\n22초]
    LLM --> TTS[🔊 ElevenLabs\n나레이션+BGM]
    K --> MX[⚙️ ffmpeg\n합성]
    TTS --> MX
    MX --> OUT[🎬 후처리\n로고 아웃트로]
    OUT --> YT[▶️ YouTube]

    style U fill:#7c6cf5,color:#fff
    style LLM fill:#a855f7,color:#fff
    style K fill:#a855f7,color:#fff
    style TTS fill:#f59e0b,color:#fff
    style MX fill:#14b8a6,color:#fff
    style OUT fill:#14b8a6,color:#fff
    style YT fill:#ef4444,color:#fff
    style F fill:#3b82f6,color:#fff
```

**② 피드백 분석 루프**

```mermaid
flowchart LR
    YT[▶️ YouTube] --> COL[💬 댓글 수집\nData API v3]
    COL --> FIL[🔍 피드백 필터\n건설성 분류]
    FIL --> CLS[📊 6축 분류기\n임베딩 모델]
    CLS --> CLU[🗂️ 클러스터링\nHDBSCAN]
    CLU --> APR[✅ 인간 검토\n승인 / 거절]
    APR -->|개선된 프롬프트| NEXT[🔄 다음 라운드]

    style YT fill:#ef4444,color:#fff
    style COL fill:#ef4444,color:#fff
    style FIL fill:#a855f7,color:#fff
    style CLS fill:#a855f7,color:#fff
    style CLU fill:#a855f7,color:#fff
    style APR fill:#3b82f6,color:#fff
    style NEXT fill:#a855f7,color:#fff
```

---

## 3. VideoRound 상태 머신

각 라운드는 아래 상태를 순서대로 거칩니다.  
**오디오 피드백만 있을 경우** 영상 재생성 없이 `video_ready`로 바로 진입해 비용을 절감합니다.

```mermaid
stateDiagram-v2
    [*] --> prompt_built : 세션 생성 / 라운드 신규

    prompt_built --> video_ready : Kling V3 영상 생성
    video_ready --> uploaded : YouTube 업로드
    uploaded --> feedback_collected : 댓글 수집
    feedback_collected --> feedback_classified : 6축 분류
    feedback_classified --> feedback_clustered : 클러스터링
    feedback_clustered --> pending_review : 검토 화면 진입
    pending_review --> prompt_updated : 클러스터 승인 완료

    prompt_updated --> prompt_built : 시각 변경 있음 → Kling 재생성
    prompt_updated --> video_ready  : 오디오/후처리만 → 영상 재사용 💰

    note right of pending_review
        사용자가 approved / rejected / deferred
        중 하나로 각 클러스터를 직접 판정
    end note
```

> **서버 재시작 안전장치**: `generating` 상태로 멈춘 좀비 라운드는 서버 시작 시 `prompt_built`로 자동 복구됩니다.

---

## 4. 댓글 클러스터링 파이프라인 ⭐ 핵심

### 4-1. 전체 흐름

```mermaid
flowchart TD
    A([📥 댓글 텍스트 입력]) --> B{스팸 / 부적절\n빠른 필터}

    B -->|패턴 2개 이상 매칭| SP[spam / inappropriate\n분류 종료]
    B -->|통과| C[sentence-transformers\n임베딩 생성\nparaphrase-multilingual-MiniLM-L12-v2]

    C --> D[원형 코사인 유사도\n4개 레이블 중 1차 판단]
    D --> E[구체성 specificity\n규칙 점수 산정]
    D --> F[타당성 validity\n규칙 점수 산정]
    E --> G[심각도 severity\n규칙 점수 산정]
    F --> G
    G --> H[긴급도 urgency\n규칙 점수 산정]

    H --> I{최종 레이블 결정\nconstructive 판별}
    I -->|specificity≥0.35\nOR validity≥0.30| CONS[✅ constructive\n건설적 피드백]
    I -->|미달| NONC[non_constructive]

    CONS --> J[품질 quality 종합 점수\n0.30·spec + 0.25·val + 0.15·sev + 0.15·len]
    CONS --> K[멀티레이블 토픽 태깅\naudio / visual / subtitle / pacing / story 등]
    J --> L[(SQLite DB\nFeedback 저장)]
    K --> L

    L --> M{클러스터링}
    M -->|HDBSCAN 성공| HDBSCAN[HDBSCAN\n밀도 기반 클러스터링]
    M -->|실패 / 데이터 부족| AGG[AgglomerativeClustering\n계층적 클러스터링 폴백]

    HDBSCAN --> TH[카테고리별 적응형 임계값]
    AGG --> TH

    TH --> PR[우선순위 P 계산]
    PR --> BO[재발 부스트 적용]
    BO --> CO[상반 클러스터 페널티]
    CO --> AP{인간 검토}

    AP -->|approved| UP[Gemini LLM\n프롬프트 자동 수정]
    AP -->|rejected| REJ[다음 라운드 미반영]
    AP -->|deferred| DEF[보류 — 나중에 재검토]

    UP --> OL[온라인 학습\nconstructive 프로토타입 업데이트\nweight=0.1]
    UP --> NR([🔄 새 라운드 시작])

    style CONS fill:#10b981,color:#fff
    style SP fill:#6b7280,color:#fff
    style NONC fill:#6b7280,color:#fff
    style UP fill:#a855f7,color:#fff
    style NR fill:#7c6cf5,color:#fff
    style AP fill:#3b82f6,color:#fff
```

---

### 4-2. 6축 피드백 분류 점수

각 댓글은 아래 6가지 축으로 **동시에** 점수화됩니다.

| 축 | 설명 | 판단 방식 | 활용 |
|----|------|-----------|------|
| **specificity** | 구체성 — 얼마나 명확한 요구인가 | 키워드 패턴 점수 (≥0.35 → constructive) | constructive 판별 기준 |
| **validity** | 타당성 — 근거가 있는 의견인가 | 키워드 패턴 점수 (≥0.30 → constructive) | constructive 판별 보조 |
| **severity** | 심각도 — 시청에 얼마나 방해되는가 | 패턴 점수 (음수 = 경미) | 우선순위 가중치 |
| **urgency** | 긴급도 — 즉각 처리가 필요한가 | 패턴 점수 | 우선순위 가중치 |
| **quality** | 품질 종합 | 5축 + 길이 + 수치 포함 여부 합산 | 대표 텍스트 선택 |
| **actionability** | 실행가능성 | 기본값 0.5 | 클러스터 우선순위 보조 |

> **언어 감지**: 한국어/영어/일본어/중국어/혼합어 자동 감지.  
> 한 언어가 70% 미만이면 `mixed_ko_en` 등으로 분류합니다.

---

### 4-3. 클러스터링 알고리즘

#### 카테고리별 적응형 거리 임계값

| 카테고리 | 임계값 | 이유 |
|----------|--------|------|
| `technical` | **0.26** | 기술 이슈는 표현이 매우 구체적 → 좁게 |
| `subtitle` | **0.30** | 자막 피드백은 구체적 표현 집중 |
| `visual` | **0.36** | 색감/조명은 표현 중간 |
| `audio` | **0.38** | 오디오 피드백은 표현 다양 |
| `other` | **0.40** | 기타 |
| `content` | **0.44** | 내용 구성은 표현 다양 |
| `pacing` | **0.42** | 속도/편집은 표현 다양 |
| `story` | **0.48** | 스토리는 표현이 매우 다양 → 넓게 |

- **대형 클러스터(> 15개)**: 임계값 × 0.58 로 서브클러스터링 자동 실행
- **응집도(cohesion)**: 클러스터 내 쌍별 코사인 유사도 평균으로 계산

---

### 4-4. 우선순위 공식

$$P = 0.35 \times \text{freq} + 0.25 \times \text{quality} + 0.20 \times \text{severity} + 0.12 \times \text{urgency} + 0.08 \times \text{validity}$$

```
P = 0.35 × 빈도점수
  + 0.25 × 평균 품질점수
  + 0.20 × 평균 심각도점수
  + 0.12 × 평균 긴급도점수
  + 0.08 × 평균 타당성점수
```

#### 보정 규칙

| 상황 | 보정 | 최대값 |
|------|------|--------|
| 재발 1회 (이전 라운드에도 등장) | P × **1.10** | — |
| 재발 2회 | P × **1.20** | — |
| 재발 3회 이상 | P × **1.30** | 1.0 상한 |
| 상반된 피드백 클러스터 존재 | P × **0.60** | — |

> 예: "BGM 줄여주세요"와 "BGM 더 키워주세요"가 동시에 등장하면, 두 클러스터 모두 40% 감산됩니다.

---

### 4-5. 클러스터 승인 워크플로

```mermaid
flowchart TD
    CLU[클러스터 생성\nstatus: pending] --> REVIEW[검토 화면\n/session/id/review]

    REVIEW --> APP[approved ✅\n프롬프트에 반영]
    REVIEW --> REJ[rejected ❌\n이번 라운드 제외]
    REVIEW --> DEF[deferred ⏸\n다음 라운드 재검토]

    APP --> AUDIO{카테고리?}
    AUDIO -->|audio bgm| BGM[bgm_db ±6dB\nbgm_prompt_override\nGlobalTechnicalSetting]
    AUDIO -->|audio voice| VOICE[voice_selector.py\n최적 보이스 자동 선택\nGlobalTechnicalSetting]
    AUDIO -->|visual 등| VISUAL[Gemini LLM\n프롬프트 텍스트 수정\nPromptVersion 저장]

    APP --> INJ[인젝션 검사\n25+ 패턴 정규식\nneedsReview=True]
    APP --> LEARN[온라인 학습\n지수이동평균 weight=0.1]

    style APP fill:#10b981,color:#fff
    style REJ fill:#ef4444,color:#fff
    style DEF fill:#f59e0b,color:#fff
```

---

## 5. 피드백 자동 반영 요약

> **핵심 원칙**: 피드백 적용 = "나레이션 버튼" 한 번으로 모든 설정이 자동 반영됩니다.

| 피드백 예시 | 자동 처리 | 저장 위치 | 적용 시점 |
|------------|-----------|-----------|---------|
| "BGM이 너무 커요" | bgm_db **-6dB** | `GlobalTechnicalSetting` audio.bgm_db | 나레이션 버튼 |
| "배경음악이 너무 작아요" | bgm_db **+6dB** | `GlobalTechnicalSetting` audio.bgm_db | 나레이션 버튼 |
| "나레이션이 잘 안 들려요" | narration_db **+3dB** | `GlobalTechnicalSetting` audio.narration_db | 나레이션 버튼 |
| "배경음악 더 웅장하게" | bgm_prompt_override 저장 | `GlobalTechnicalSetting` audio.bgm_prompt_override | BGM 버튼 |
| "목소리 바꿔주세요" | voice_selector → 5종 중 자동 선택 | `GlobalTechnicalSetting` voice.voice_id | 나레이션 버튼 |
| "끝부분을 자연스럽게" | fade_duration **2.5초**로 확장 | `GlobalTechnicalSetting` video_post.fade_duration | 나레이션 버튼 |
| "로고 넣어주세요" | logo_outro = **true** | `GlobalTechnicalSetting` video_post.logo_outro | 나레이션 버튼 |
| "화면이 더 밝았으면" | Gemini LLM이 프롬프트 style 섹션 수정 | `PromptVersion` (unified diff 저장) | Kling 버튼 |

---

## 6. 영상 생성 — Kling V3

```mermaid
flowchart TD
    P[프롬프트] --> STRIP[_strip_audio_for_kling\nAUDIO 섹션 제거]
    STRIP --> SUBMIT[fal_client.submit\nrequest_id 즉시 확보]
    SUBMIT --> POLL{폴링\n30초 간격\n최대 10분}
    POLL -->|진행 중| POLL
    POLL -->|완료| DL[영상 다운로드]
    POLL -->|타임아웃| FAIL[실패 처리\n중복 제출 없음]

    DL --> LEN{목표 길이?}
    LEN -->|≤15초| DONE[단일 클립 저장]
    LEN -->|>15초 기본 22초| SPLIT[15초 + 7초 자동 분할]
    SPLIT --> C1[클립1 생성\n15초]
    SPLIT --> C2[클립2 생성\n7초\n마지막 프레임 체이닝]
    C1 --> CONCAT[ffmpeg concat]
    C2 --> CONCAT
    CONCAT --> DONE

    style FAIL fill:#ef4444,color:#fff
    style DONE fill:#10b981,color:#fff
```

| 항목 | 값 |
|------|----|
| 단가 | $0.084 / 초 (오디오 OFF) |
| 22초 영상 | **$1.85** (~₩2,680) |
| 최대 클립 | 15초 / 요청 |
| 일일 안전 한도 | `MAX_PAID_GENERATIONS_PER_DAY=20` |
| 프롬프트 최대 | 2,500자 |

---

## 7. 나레이션 파이프라인

```mermaid
flowchart TD
    A[나레이션 버튼 클릭] --> B[GlobalTechnicalSetting 읽기\nvoice_id / bgm_db / narration_db\nfade_duration / logo_outro]
    B --> C[parse_narration_cues\nAt Xs: text 패턴 파싱]
    C --> D[ElevenLabs TTS\neleven_v4 모델\n감정 태그 + 속도 적용]
    D --> E{logo_outro = true?}
    E -->|Yes| F[apply_outro 먼저 실행\n페이드아웃 + 로고카드 6초\nffmpeg 1회 인코딩 ~50MB]
    E -->|No| G
    F --> G[mix_narration_into_video\nadelay 타임스탬프 배치\nBGM + 나레이션 + 원본 믹싱]
    G --> H[DB 업데이트\nvideo_file_path 갱신]
    H --> I{subtitle_auto = true?}
    I -->|Yes| J[Whisper 자막 burn-in\nSRT 자동 생성]
    I -->|No| K([완료])
    J --> K

    style A fill:#7c6cf5,color:#fff
    style K fill:#10b981,color:#fff
```

> **BGM 동점 분류 로직**: "음악을 줄이거나 나레이션소리를 키우면" 처럼 bgm/voice 점수가 동점인 경우, BGM 볼륨 액션 키워드(줄여/낮춰/줄이) 포함 여부로 최종 분류합니다.

---

## 8. 보안 설계

| 위협 | 방어 방식 |
|------|----------|
| **프롬프트 인젝션** | 25+ 패턴 정규식 감지 → `needs_review=True` 마킹, LLM 피드백 데이터를 `<feedback_data>` XML 태그로 격리 |
| **SSRF** | `_validate_image_url()` — 허용 도메인 7개 화이트리스트, 내부 IP 차단 |
| **경로 탐색** | `_safe_send_file()` — `os.path.realpath()` + upload_root 경계 검사 |
| **SQL 인젝션** | SQLAlchemy ORM 파라미터 바인딩 (raw SQL 미사용) |
| **CSRF** | Flask-WTF (1시간 토큰) |
| **Rate Limit** | Flask-Limiter (AI API 10/분, 변형 5/분) |
| **레이스 컨디션** | `_narration_running: dict[id, float]` 락 + 10분 자동 해제 |
| **파일 업로드** | 확장자 검증 + 500MB 제한 |
| **API 키 노출** | `utils/masking.py` — 로그 마스킹 |

---

## 9. 기술 스택

```mermaid
mindmap
  root((FrameLoop))
    백엔드
      Flask Python
      SQLite WAL 32MB캐시
      Flask-WTF CSRF
      Flask-Limiter
    AI 엔진
      Kling V3 fal.ai 영상생성
      Gemini 프롬프트엔진
      sentence-transformers 분류기
      HDBSCAN 클러스터링
      Whisper 자막
    오디오
      ElevenLabs TTS eleven_v4
      ElevenLabs Sound BGM
      ffmpeg 믹싱
      Pillow 로고카드
    외부 서비스
      YouTube Data API v3
      fal.ai API
      Google Gemini API
```

| 모듈 | 모델 / 버전 | 역할 |
|------|------------|------|
| 피드백 분류 | `paraphrase-multilingual-MiniLM-L12-v2` | 6축 분류 (코사인 유사도) |
| 클러스터 임베딩 | `paraphrase-multilingual-mpnet-base-v2` | 문장 임베딩 생성 |
| 프롬프트 엔진 | `gemini-3.1-flash-lite` | 프롬프트 생성·수정 |
| 영상 생성 | `fal-ai/kling-video/v3/standard/text-to-video` | 22초 영상 |
| TTS | ElevenLabs `eleven_v4` | 나레이션 5종 보이스 |
| 자막 | OpenAI Whisper (로컬 small) | SRT 자막 |

---

## 10. DB 모델 관계도

```mermaid
erDiagram
    PipelineSession ||--o{ VideoRound : "1:N rounds"
    PipelineSession ||--o{ PromptVersion : "버전 이력"
    PipelineSession ||--o{ CharacterProfile : "참조 이미지"
    VideoRound ||--o{ Feedback : "1:N feedbacks"
    VideoRound ||--o{ ClusterResult : "1:N clusters"
    VideoRound ||--o{ ShotRound : "샷 단위"
    VideoRound ||--|| PromptChange : "변경 레거시"
    Feedback }o--|| ClusterResult : "cluster_key"
    ShotRound ||--o{ Feedback : "shot_round_id"

    PipelineSession {
        int id PK
        string genre
        string title
        text user_script
        string owner_token
        text canvas_json
    }
    VideoRound {
        int id PK
        string status
        text video_gen_prompt
        string video_file_path
        string youtube_url
        string last_frame_path
        float positive_feedback_ratio
    }
    ClusterResult {
        string cluster_key PK
        string category
        float priority_score
        int reoccurrence_count
        string status
    }
    GlobalTechnicalSetting {
        string category
        string setting_key
        string setting_value
        bool is_active
    }
```

---

*최종 업데이트: 2026-10-27 | 16차 세션 완료*
