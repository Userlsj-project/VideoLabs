# AI 영상 피드백 자동 개선 시스템

> 시청자 댓글 → 건설적 피드백 추출 → 자동 군집화 → 프롬프트 반영  
> 외부 계정·GPU 없이 로컬 CPU에서 완전 동작하는 엔드-투-엔드 파이프라인

---

## 1. 시스템 개요

```mermaid
flowchart TD
    A[제작자가 영상 프롬프트 작성] --> B[AI 영상 생성\nmock / Runway Gen-3]
    B --> C[영상 업로드\nYouTube or 시뮬레이션]
    C --> D[시청자 댓글 수집\nYouTube API or 시뮬레이션]
    D --> E[피드백 분류\n6축 평가]
    E --> F[클러스터링\n유사 피드백 그룹화]
    F --> G[제작자 검토·승인\n웹 UI]
    G --> H[다음 영상 프롬프트 자동 업데이트]
    H --> B

    style E fill:#4A90D9,color:#fff
    style F fill:#27AE60,color:#fff
    style G fill:#E67E22,color:#fff
```

**핵심 가치**: 수백 개 댓글 중 실제로 반영할 만한 피드백만 추려 다음 영상에 자동 적용 → 제작자의 반복 수작업 제거

---

## 2. 파이프라인 상태 머신

```mermaid
stateDiagram-v2
    [*] --> prompt_built : 초기 프롬프트 작성
    prompt_built --> video_ready : AI 영상 생성 완료
    video_ready --> uploaded : YouTube 업로드
    uploaded --> feedback_collected : 댓글 수집
    feedback_collected --> feedback_classified : 피드백 분류 (6축)
    feedback_classified --> pending_review : 클러스터링 완료
    pending_review --> prompt_updated : 승인 적용
    prompt_updated --> [*] : 다음 라운드 생성
```

각 단계는 Flask 라우트 POST 요청으로 전환. DB에 상태 저장 → 새로고침해도 진행 상태 유지.

---

## 3. 피드백 분류 파이프라인 (6축 평가)

```mermaid
flowchart TD
    subgraph INPUT
        A[원문 댓글]
    end

    subgraph PREPROCESS
        B[언어 감지\nKO / EN / JA / ZH / mixed]
        C[스팸·부적절 필터링\n정규식 패턴 매칭]
    end

    subgraph EMBEDDING
        D[다국어 문장 임베딩\nparaphrase-multilingual-MiniLM-L12-v2\nApache 2.0 · 118MB · CPU]
        E[카테고리 원형 문장과\n코사인 유사도 비교]
    end

    subgraph SIX_AXIS["6축 점수 계산"]
        F1[구체성\nSpecificity\n수치·비교·요청 패턴]
        F2[타당성\nValidity\n인과 연결·논리 표현]
        F3[심각도\nSeverity\n부정 강도 신호]
        F4[긴급도\nUrgency\n감탄부호·대문자 빈도]
        F5[품질\nQuality\n0.30·spec + 0.25·val\n+ 0.15·sev + 0.15·len]
        F6[다중 레이블 주제\nTopic Multi-label\n최대 3개 카테고리]
    end

    subgraph OUTPUT
        G1[constructive / spam / inappropriate\n/ non_constructive]
        G2[needs_review 플래그\n신호 격차 < 0.05 시 활성화]
    end

    A --> B --> C --> D --> E
    E --> F1 & F2 & F3 & F4 & F5 & F6
    F1 & F2 & F3 & F4 & F5 & F6 --> G1 --> G2
```

### 폴백 체인

```mermaid
flowchart LR
    A[EmbeddingClassifier\n원형 코사인 유사도\n인증 불필요] -->|로드 실패| B[ZeroShotClassifier\nmDeBERTa NLI\nHuggingFace 인증 필요]
    B -->|로드 실패| C[RuleBasedClassifier\n키워드 규칙\n완전 오프라인]
```

---

## 4. 클러스터링 파이프라인

```mermaid
flowchart TD
    subgraph INPUT2[입력]
        I1[constructive 피드백 목록\n원문 comment_text 사용]
    end

    subgraph EMBED2[임베딩]
        E1[다국어 문장 임베딩\nparaphrase-multilingual-mpnet-base-v2\n278MB · 50+ 언어]
        E2[폴백: MiniLM-L12-v2\n또는 TF-IDF]
    end

    subgraph TOPIC[1단계: 주제 태깅]
        T1[topic_category 기준 그룹 분리\naudio / subtitle / pacing\nvisual / story / content\ntechnical / other]
    end

    subgraph CLUSTER[2단계: 그룹 내 군집화]
        C1{HDBSCAN\n사용 가능?}
        C2[HDBSCAN\n밀도 기반 군집화\n노이즈 -1 → 최근접 클러스터 재배정]
        C3[AgglomerativeClustering\n계층적 군집화\ncosine + average linkage]
        C4[카테고리별 적응형 거리 임계값\ntechnical=0.26 ~ story=0.48]
    end

    subgraph SUBCLUSTER[대형 클러스터 처리]
        S1{클러스터 크기\n> 15개?}
        S2[임계값 × 0.58로\n하위 군집화 시도]
        S3[하위 클러스터 각 ≥ 2개 이상 시\n분할 적용]
    end

    subgraph REPRESENT[대표 텍스트 선정]
        R1[중심성 점수\n코사인 유사도 기반 50%]
        R2[품질 점수 30%]
        R3[길이 점수 20%]
        R4[가중 합산으로 최적 텍스트 선택]
    end

    subgraph METRICS[클러스터 품질 지표]
        M1[응집도 cohesion_score\n평균 쌍별 코사인 유사도 → 0~1 정규화]
        M2[추출 요약 cluster_summary\n단어 빈도 기반 대표 문장 선택]
        M3[평균 심각도·긴급도·품질 점수\navg_severity / urgency / quality]
    end

    INPUT2 --> EMBED2 --> TOPIC --> C1
    C1 -->|Yes| C2
    C1 -->|No| C3
    C2 & C3 --> C4 --> S1
    S1 -->|Yes| S2 --> S3
    S1 -->|No| REPRESENT
    S3 --> REPRESENT
    R1 & R2 & R3 --> R4 --> METRICS
```

---

## 5. 우선순위 산정 공식 (5축)

```mermaid
flowchart LR
    subgraph FORMULA["P = α·freq + β·quality + γ·severity + δ·urgency + ε·validity"]
        W1["α = 0.35\n빈도 (클러스터 크기)"]
        W2["β = 0.25\n품질 점수"]
        W3["γ = 0.20\n심각도 평균"]
        W4["δ = 0.12\n긴급도 평균"]
        W5["ε = 0.08\n타당성 평균"]
    end

    subgraph ADJUST[조정 요소]
        A1[재발 부스트\n이전 라운드 동일 주제 재등장\n+10%/라운드, 최대 +30%]
        A2[상반된 피드백 페널티\n'빠르게' ↔ '느리게' 동시 존재\n×0.60]
    end

    FORMULA --> ADJUST --> OUT[최종 우선순위\n0.0 ~ 1.0]
```

---

## 6. 승인 워크플로 및 프롬프트 버전 관리

```mermaid
flowchart TD
    CR[ClusterResult\npending_review] --> UI[제작자 검토 웹 UI]
    UI --> AP[approved]
    UI --> RJ[rejected]
    UI --> DF[deferred]

    AP --> PU[apply_approved_clusters\n승인된 클러스터 반영]
    PU --> PV[PromptVersion 저장\nunified diff 포함]
    PV --> NR[다음 라운드 생성\nprompt_built]

    PV --> RB[롤백 기능\n특정 버전으로 되돌리기]

    style AP fill:#27AE60,color:#fff
    style RJ fill:#E74C3C,color:#fff
    style DF fill:#F39C12,color:#fff
```

---

## 7. 기술 스택 및 모델

| 구분 | 기술/모델 | 라이선스 | 크기 | 인증 |
|------|-----------|---------|------|------|
| 피드백 분류 | paraphrase-multilingual-MiniLM-L12-v2 | Apache 2.0 | 118MB | 불필요 |
| 문장 임베딩 | paraphrase-multilingual-mpnet-base-v2 | Apache 2.0 | 278MB | 불필요 |
| 군집화 (기본) | HDBSCAN 0.8.x | BSD | - | 불필요 |
| 군집화 (폴백) | sklearn AgglomerativeClustering | BSD | - | 불필요 |
| 백엔드 | Python 3.11, Flask 3.x, SQLAlchemy | - | - | - |
| 프론트엔드 | Bootstrap 5, Chart.js | MIT | - | - |
| AI 영상 생성 | Runway Gen-3 Alpha Turbo (또는 mock) | 상용 | - | API 키 |
| 프롬프트 생성 | Claude API (또는 템플릿 폴백) | 상용 | - | API 키 |

**모든 핵심 기능은 로컬 CPU에서 실행 — GPU 불필요, 외부 계정 불필요**

---

## 8. 직접 설계·구현한 핵심 모듈

```mermaid
mindmap
  root((팀 직접 구현))
    분류기
      prototype 코사인 유사도 분류기
      6축 평가 로직
      동적 원형 업데이트 EMA
      다중 레이블 주제 감지 최대 3개
    클러스터링
      2단계 파이프라인 주제 태깅 → 군집화
      카테고리별 적응형 거리 임계값
      품질 가중 대표 텍스트 선정
      응집도 점수 계산
      추출 요약 생성
      대형 클러스터 자동 분할
    우선순위
      5축 가중 공식
      재발 부스트 라운드별 이력 추적
      상반된 피드백 페널티
    워크플로
      승인 검토 UI
      프롬프트 버전 관리 및 롤백
      unified diff 저장
      전역 기술 설정 승격
```

---

## 9. 구현 완성도

| 기능 | 상태 |
|------|------|
| 피드백 수집 (시뮬레이션 / YouTube API) | ✅ 완료 |
| 6축 피드백 분류 | ✅ 완료 |
| HDBSCAN + 적응형 임계값 클러스터링 | ✅ 완료 |
| 5축 우선순위 산정 (재발 부스트·페널티 포함) | ✅ 완료 |
| 클러스터 응집도·요약 자동 생성 | ✅ 완료 |
| 제작자 검토 웹 UI | ✅ 완료 |
| 프롬프트 버전 관리 + 롤백 | ✅ 완료 |
| 분석 리포트 (Chart.js) | ✅ 완료 |
| Runway Gen-3 영상 생성 어댑터 | ✅ 완료 |
| Kling 영상 생성 어댑터 | 🔲 미구현 |
| 실제 YouTube 업로드 연동 | 🔲 미테스트 |
