from app import create_app
from app.models import db, Feedback, VideoRound, ClusterResult
from app.pipeline.classifier import create_classifier
from app.pipeline.clusterer import run_clustering
from app.pipeline.approval import save_clusters
from datetime import datetime

app = create_app()
SESSION_ID = 1

with app.app_context():
    rnd = VideoRound.query.filter_by(session_id=SESSION_ID, round_number=1).first()

    # ── 1. 피드백 재분류 (수정된 분류기 적용) ────────────────────────────
    all_fb = rnd.feedbacks
    clf = create_classifier(
        model_name=app.config.get("CLASSIFIER_MODEL"),
        specificity_threshold=app.config.get("SPECIFICITY_THRESHOLD", 0.45),
        validity_threshold=app.config.get("VALIDITY_THRESHOLD", 0.40),
    )
    results = clf.classify([f.comment_text for f in all_fb])
    now = datetime.utcnow()
    for fb, r in zip(all_fb, results):
        fb.category = r.constructive_label
        fb.improvement_suggestion = r.improvement_suggestion
        fb.confidence = r.confidence
        fb.specificity_score = r.specificity_score
        fb.validity_score = r.validity_score
        fb.topic_category = r.topic_category
        fb.scope = r.scope
        fb.detected_language = r.detected_language
        fb.classified_at = now
    rnd.status = "feedback_classified"
    db.session.commit()

    constructive = [f for f in all_fb if f.category == "constructive"]
    print(f"재분류 완료: 건설적 {len(constructive)}/{len(all_fb)}개")
    for fb in constructive:
        print(f"  [{fb.topic_category:8s}] {fb.comment_text[:55]}")

    # ── 2. 클러스터링 ─────────────────────────────────────────────────────
    print("\n클러스터링 실행 중...")
    texts         = [f.comment_text for f in constructive]
    topic_cats    = [f.topic_category or "other" for f in constructive]
    validity_sc   = [f.validity_score or 0.5 for f in constructive]
    severity_sc   = [f.severity_score or 0.3 for f in constructive]
    urgency_sc    = [f.urgency_score or 0.2 for f in constructive]
    quality_sc    = [f.quality_score or 0.5 for f in constructive]
    orig_indices  = list(range(len(constructive)))

    raw_clusters = run_clustering(
        texts=texts,
        topic_categories=topic_cats,
        validity_scores=validity_sc,
        severity_scores=severity_sc,
        urgency_scores=urgency_sc,
        quality_scores=quality_sc,
        original_indices=orig_indices,
        embedding_model=app.config.get("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2"),
        embedding_fallback=app.config.get("EMBEDDING_FALLBACK_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        distance_threshold=app.config.get("CLUSTERING_DISTANCE_THRESHOLD", 0.40),
        priority_alpha=app.config.get("PRIORITY_ALPHA", 0.35),
        priority_beta=app.config.get("PRIORITY_BETA", 0.25),
        priority_gamma=app.config.get("PRIORITY_GAMMA", 0.20),
        priority_delta=app.config.get("PRIORITY_DELTA", 0.12),
        priority_epsilon=app.config.get("PRIORITY_EPSILON", 0.08),
    )

    # 기존 pending 클러스터 제거 후 저장
    ClusterResult.query.filter_by(video_round_id=rnd.id, status="pending").delete()
    saved = save_clusters(raw_clusters, rnd, constructive)
    rnd.status = "pending_review"
    db.session.commit()

    # ── 3. 결과 출력 ─────────────────────────────────────────────────────
    print(f"\n=== 클러스터링 결과: {len(saved)}개 클러스터 ===")
    for cr in sorted(saved, key=lambda c: c.priority_score or 0, reverse=True):
        scope_tag = "[전역]" if cr.scope == "technical" else "[콘텐츠]"
        members = Feedback.query.filter_by(cluster_key=cr.cluster_key).all()
        print(f"\n{scope_tag} {cr.label}")
        print(f"  우선순위: {cr.priority_score:.3f}  |  피드백 {cr.count}개  |  유사도 {cr.avg_similarity:.2f}")
        print(f"  대표: {cr.representative_text[:60]}")
        for m in members:
            print(f"    - {m.comment_text[:55]}")

    print(f"\n웹 검토 화면: http://127.0.0.1:5000/session/{SESSION_ID}/review")
