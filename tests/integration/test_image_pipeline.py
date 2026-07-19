"""Integration test: full POST /analyze/xray pipeline against live services."""

import io
from pathlib import Path

import httpx
import pytest
from PIL import Image as PILImage

SAMPLE_XRAY = Path(__file__).parent / "fixtures" / "sample_xray.png"

ALL_CONDITIONS = {
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
}


def test_analyze_xray_end_to_end(client, doctor, db, cleanup_conversation):
    """A real chest X-ray upload produces a complete, internally consistent response and DB record."""
    assert SAMPLE_XRAY.exists(), (
        f"Missing test fixture at {SAMPLE_XRAY} — place a real chest X-ray PNG there first."
    )

    with open(SAMPLE_XRAY, "rb") as f:
        response = client.post(
            "/analyze/xray",
            files={"file": ("sample_xray.png", f, "image/png")},
            data={"conversation_id": ""},
            headers=doctor["headers"],
        )

    assert response.status_code == 200, response.text
    body = response.json()
    cleanup_conversation(body["conversation_id"])

    assert body["interaction_id"] == body["conversation_id"]
    assert set(body["all_scores"].keys()) == ALL_CONDITIONS
    assert isinstance(body["low_confidence_flag"], bool)
    assert body["latency_ms"] > 0

    if body["low_confidence_flag"]:
        assert body["above_threshold"] == []
        assert body["gradcam_findings"] == []
        assert body["conditions"] == []
    else:
        assert len(body["above_threshold"]) > 0
        assert len(body["gradcam_findings"]) == len(body["above_threshold"])
        assert len(body["conditions"]) == len(body["above_threshold"])
        for finding in body["gradcam_findings"]:
            assert finding["heatmap_url"].startswith("https://")
            assert finding["condition"] in body["above_threshold"]

    from scripts.db_models import Interaction, CNNResult, GradCAMFinding, RAGLog, LLMOutput

    interaction = db.query(Interaction).filter_by(id=body["interaction_id"]).first()
    assert interaction is not None
    assert interaction.interaction_type == "image"
    assert interaction.conversation_id == body["conversation_id"]
    assert interaction.xray_storage_url.startswith("https://")

    cnn_result = db.query(CNNResult).filter_by(interaction_id=interaction.id).first()
    assert cnn_result is not None
    assert cnn_result.above_threshold == body["above_threshold"]

    gradcam_rows = db.query(GradCAMFinding).filter_by(interaction_id=interaction.id).all()
    assert len(gradcam_rows) == len(body["gradcam_findings"])
    for row in gradcam_rows:
        assert row.heatmap_storage_url.startswith("https://")
        assert isinstance(row.dominant_zones, list)
        assert isinstance(row.aligned, bool)

    llm_output = db.query(LLMOutput).filter_by(interaction_id=interaction.id).first()
    assert llm_output is not None
    assert llm_output.call2_output["clinical_summary"] == body["clinical_summary"]
    assert llm_output.text_response is None  # image path never populates text_response

    if not body["low_confidence_flag"]:
        rag_logs = db.query(RAGLog).filter_by(interaction_id=interaction.id).all()
        assert len(rag_logs) == len(body["above_threshold"])
        for log in rag_logs:
            assert log.condition in body["above_threshold"]


def test_analyze_xray_rejects_non_xray_image(client, doctor):
    """A clearly non-chest-X-ray image (solid color square) is rejected by CLIP validation."""
    buf = io.BytesIO()
    PILImage.new("RGB", (224, 224), color=(255, 0, 0)).save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/analyze/xray",
        files={"file": ("not_xray.png", buf, "image/png")},
        data={"conversation_id": ""},
        headers=doctor["headers"],
    )
    assert response.status_code == 422


def test_analyze_xray_requires_auth(client):
    """Uploading without a valid bearer token is rejected."""
    buf = io.BytesIO()
    PILImage.new("RGB", (224, 224), color="white").save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/analyze/xray", files={"file": ("x.png", buf, "image/png")}, data={"conversation_id": ""},
    )
    assert response.status_code == 401


def test_analyze_xray_stores_true_png_regardless_of_upload_format(client, doctor, db, cleanup_conversation):
    """A JPEG upload of a real X-ray is normalized and stored as a genuine PNG (regression test).

    Uses the real sample X-ray re-encoded as JPEG, not a synthetic image — a synthetic
    solid-color image is correctly rejected by CLIP validation before the normalization
    code is ever reached, which would test nothing.
    """
    if not SAMPLE_XRAY.exists():
        pytest.skip(f"Missing test fixture at {SAMPLE_XRAY} — place a real chest X-ray PNG there first.")

    original = PILImage.open(SAMPLE_XRAY).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    response = client.post(
        "/analyze/xray",
        files={"file": ("upload.jpg", buf, "image/jpeg")},
        data={"conversation_id": ""},
        headers=doctor["headers"],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    cleanup_conversation(body["conversation_id"])

    from scripts.db_models import Interaction
    interaction = db.query(Interaction).filter_by(id=body["interaction_id"]).first()

    stored_bytes = httpx.get(interaction.xray_storage_url).content
    assert stored_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Stored file is not a genuine PNG despite the .png path"