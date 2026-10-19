"""LLM layer: the fallback must work with no API key, and parsing must be robust."""

from __future__ import annotations


import pytest

from src import recommendations as R
from src.llm_client import LLMClient


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Guarantee the no-key path for every test in this module."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.LLM_PROVIDER", "auto")


def test_provider_falls_back_without_keys():
    client = LLMClient()
    assert client.provider == "fallback"
    assert client.is_live is False
    assert "fallback" in client.label().lower()


def test_fallback_returns_full_schema():
    """With no key, every recommendation still has the complete contract."""
    ctx = {"tenant_id": "TN1", "tenant_name": "Test", "risk_score": 82, "late_count_6m": 3}
    rec = R.RecommendationEngine().for_tenant(ctx)
    assert rec.source == "fallback"
    for field in ("action", "reason", "urgency", "suggested_message"):
        assert hasattr(rec, field)
    assert rec.urgency in R.URGENCY_LEVELS
    assert rec.action  # non-empty


@pytest.mark.parametrize(
    "score,expected",
    [(10, "Low"), (40, "Medium"), (60, "High"), (90, "Critical")],
)
def test_fallback_urgency_tracks_risk(score, expected):
    rec = R.RecommendationEngine().for_tenant(
        {"tenant_id": f"T{score}", "tenant_name": "X", "risk_score": score}
    )
    assert rec.urgency == expected


def test_parse_accepts_clean_and_fenced_json():
    clean = '{"action":"Call","reason":"High risk","urgency":"Critical","suggested_message":"Hi"}'
    fenced = "```json\n" + clean + "\n```"
    assert R._parse(clean) is not None
    assert R._parse(fenced) is not None


def test_parse_snaps_invalid_urgency():
    bad = '{"action":"x","reason":"y","urgency":"NOPE","suggested_message":""}'
    assert R._parse(bad)["urgency"] == "Medium"


def test_parse_rejects_junk_and_incomplete():
    assert R._parse("I cannot help with that.") is None
    assert R._parse("") is None
    assert R._parse('{"action":"only one field"}') is None


def test_live_path_degrades_on_malformed_output():
    """A 'live' client returning junk must degrade to the rule-based schema."""
    class JunkClient(LLMClient):
        def __init__(self):
            self.provider = "openai"
            self.model = "fake"

        @property
        def is_live(self):
            return True

        def label(self):
            return "openai · fake"

        def generate(self, system, prompt):
            return "sorry, I can't do that"

    rec = R.RecommendationEngine(client=JunkClient()).for_tenant(
        {"tenant_id": "TN9", "tenant_name": "Y", "risk_score": 55}
    )
    assert rec.source == "fallback"
    assert rec.urgency in R.URGENCY_LEVELS


def test_recommendations_are_cached():
    engine = R.RecommendationEngine()
    ctx = {"tenant_id": "TNX", "tenant_name": "Z", "risk_score": 70}
    first = engine.for_tenant(ctx)
    second = engine.for_tenant(ctx)
    assert first is second  # same cached object
