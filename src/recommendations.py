"""Layer 3 - Smart recommendation engine (generative AI).

Layer 1 says *who* is risky and Layer 2 says *what* revenue is coming. Neither
tells a non-technical landlord *what to do*. This layer turns the ML outputs
into a plain-English action per tenant:

    {action, reason, urgency, suggested_message}

Prompt engineering (see `_SYSTEM`):
* **Role** - the model is cast as a property-management advisor for the Indian
  rental market, so its tone and suggestions fit the domain.
* **Context injection** - the tenant's risk score, payment behaviour and
  profile are injected as JSON, not prose, so the model reads structured facts.
* **Output contract** - the model must return a single JSON object with four
  named fields; we parse and validate it, and any deviation degrades to the
  deterministic generator rather than surfacing malformed output.

The deterministic fallback (`_rule_based`) produces the exact same schema from
simple thresholds, so the engine - and the demo - works with no API key.
Results are cached per tenant so re-renders don't re-bill the API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .llm_client import LLMClient

URGENCY_LEVELS = ("Low", "Medium", "High", "Critical")

_SYSTEM = (
    "You are a property-management advisor for landlords in the Indian rental "
    "market, built into the TenantEase platform. You receive one tenant's "
    "risk score and payment behaviour as JSON and recommend a single, concrete "
    "next action a non-technical landlord can take today.\n\n"
    "Respond with ONLY a JSON object, no prose or markdown, with exactly these "
    "keys:\n"
    '  "action"            - the recommended action, under 12 words\n'
    '  "reason"            - one sentence grounding it in the data\n'
    '  "urgency"           - one of: Low, Medium, High, Critical\n'
    '  "suggested_message" - a short, polite message (<=320 chars) the '
    "landlord could send the tenant; empty string if no message is warranted\n\n"
    "Be specific and reference the tenant's actual numbers. Use Indian Rupee "
    "(INR) for money. Do not invent facts beyond the JSON provided."
)


@dataclass
class Recommendation:
    """One tenant's recommendation, however it was produced."""

    tenant_id: str
    action: str
    reason: str
    urgency: str
    suggested_message: str
    source: str  # "anthropic" | "openai" | "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


def _build_prompt(ctx: dict) -> str:
    """Inject the tenant context as JSON for the model to read."""
    keep = {
        "tenant_name": ctx.get("tenant_name"),
        "unit": f"{ctx.get('building', '')} {ctx.get('unit_no', '')}".strip(),
        "property_type": ctx.get("property_type"),
        "monthly_rent_inr": ctx.get("monthly_rent"),
        "risk_score_0_100": ctx.get("risk_score"),
        "risk_category": ctx.get("risk_category"),
        "on_time_rate": ctx.get("on_time_rate"),
        "late_count_last_6_cycles": ctx.get("late_count_6m"),
        "missed_cycles": ctx.get("missed_count"),
        "avg_days_past_due": ctx.get("avg_days_past_due"),
        "days_since_last_payment": ctx.get("days_since_last_payment"),
        "deposit_months": ctx.get("deposit_months"),
        "tenure_months": ctx.get("tenure_months"),
    }
    compact = {k: v for k, v in keep.items() if v is not None}
    return "Tenant data:\n" + json.dumps(compact, indent=2, default=str)


def _parse(raw: str) -> dict | None:
    """Extract and validate the model's JSON object; None if malformed."""
    if not raw:
        return None
    text = raw.strip()
    # Tolerate accidental markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None

    required = {"action", "reason", "urgency", "suggested_message"}
    if not required.issubset(data):
        return None
    if data["urgency"] not in URGENCY_LEVELS:
        # Snap to the nearest valid level rather than discard a good answer.
        data["urgency"] = "Medium"
    return {k: str(data[k]).strip() for k in required}


def _urgency_from_score(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def _rule_based(ctx: dict) -> dict:
    """Deterministic recommendation with the same schema as the LLM path."""
    name = ctx.get("tenant_name", "the tenant")
    score = float(ctx.get("risk_score", 0))
    late6 = int(ctx.get("late_count_6m", 0) or 0)
    missed = int(ctx.get("missed_count", 0) or 0)
    deposit = float(ctx.get("deposit_months", 0) or 0)
    rent = ctx.get("monthly_rent")
    urgency = _urgency_from_score(score)

    if score >= 75:
        action = "Call tenant today and review lease continuation"
        reason = (
            f"Risk score {int(score)}/100 with {missed} missed and {late6} late "
            "payments recently signals imminent default."
        )
    elif score >= 50:
        action = "Send a firm payment reminder and flag for follow-up"
        reason = f"Risk score {int(score)}/100 with {late6} late cycles in the last 6 months."
    elif score >= 25:
        action = "Send an early, friendly reminder before the due date"
        reason = f"Moderate risk ({int(score)}/100); proactive nudges keep collection on track."
    else:
        action = "No action needed; continue standard reminders"
        reason = f"Low risk ({int(score)}/100) with a healthy payment history."

    if deposit < 1 and score >= 50:
        action += "; review deposit adequacy"

    rent_txt = f" Your rent of INR {int(rent):,} for this month" if rent else " Your rent"
    if score >= 50:
        message = (
            f"Dear {name}, this is a reminder from your property manager."
            f"{rent_txt} appears overdue. Kindly clear the pending amount at the "
            "earliest or reply to discuss a payment plan. Thank you."
        )
    elif score >= 25:
        message = (
            f"Dear {name}, a friendly reminder that your rent is due shortly. "
            "Please ensure timely payment to avoid any late fees. Thank you."
        )
    else:
        message = ""

    return {
        "action": action,
        "reason": reason,
        "urgency": urgency,
        "suggested_message": message,
    }


class RecommendationEngine:
    """Generates per-tenant recommendations, with caching and fallback."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self._cache: dict[str, Recommendation] = {}

    @property
    def source_label(self) -> str:
        return self.client.label()

    def for_tenant(self, ctx: dict, *, use_cache: bool = True) -> Recommendation:
        tenant_id = str(ctx.get("tenant_id", ctx.get("tenant_name", "unknown")))
        if use_cache and tenant_id in self._cache:
            return self._cache[tenant_id]

        parsed = None
        if self.client.is_live:
            raw = self.client.generate(_SYSTEM, _build_prompt(ctx))
            parsed = _parse(raw) if raw else None

        source = self.client.provider if parsed else "fallback"
        fields = parsed or _rule_based(ctx)

        rec = Recommendation(tenant_id=tenant_id, source=source, **fields)
        if use_cache:
            self._cache[tenant_id] = rec
        return rec


def for_tenant(ctx: dict) -> dict:
    """Convenience one-shot used by scripts and smoke tests."""
    return RecommendationEngine().for_tenant(ctx).to_dict()
