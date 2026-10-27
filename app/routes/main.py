from flask import Blueprint, render_template
from ..models import PipelineSession, Metric

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    sessions = PipelineSession.query.order_by(PipelineSession.created_at.desc()).all()
    return render_template("index.html", sessions=sessions)
