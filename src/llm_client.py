"""Provider-agnostic LLM client for the recommendation engine.

The engine must run for anyone who clones the repo, including a reviewer with
no API key, so this client resolves a backend at construction time:

    TENANTEASE_LLM_PROVIDER = auto (default) | anthropic | openai | fallback

* **anthropic** - Anthropic SDK, default model `claude-haiku-5-5` (fast, cheap,
  well suited to short structured generations at per-tenant volume).
* **openai** - OpenAI SDK, default model `gpt-4o-mini` (Cynbit's documented
  stack lists OpenAI, so this mirrors their tooling).
* **fallback** - no network call; `generate()` returns None and the caller
  uses its deterministic rule-based generator.

In `auto` mode we pick Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI if
`OPENAI_API_KEY` is set, else fallback. Each backend uses its own official SDK
- the two are never mixed. Any runtime error degrades to fallback rather than
breaking the dashboard.
"""

from __future__ import annotations

import os

from . import config


class LLMClient:
    """Resolves and calls whichever LLM backend is configured."""

    def __init__(self) -> None:
        self.provider = self._resolve_provider()
        self.model = self._resolve_model()

    # -- resolution --------------------------------------------------------

    @staticmethod
    def _has(key: str) -> bool:
        return bool(os.getenv(key, "").strip())

    def _resolve_provider(self) -> str:
        choice = config.LLM_PROVIDER
        if choice in {"anthropic", "openai", "fallback"}:
            return choice
        # auto
        if self._has("ANTHROPIC_API_KEY"):
            return "anthropic"
        if self._has("OPENAI_API_KEY"):
            return "openai"
        return "fallback"

    def _resolve_model(self) -> str:
        if config.LLM_MODEL:
            return config.LLM_MODEL
        if self.provider == "anthropic":
            return config.DEFAULT_ANTHROPIC_MODEL
        if self.provider == "openai":
            return config.DEFAULT_OPENAI_MODEL
        return "rule-based-fallback"

    @property
    def is_live(self) -> bool:
        """True when a real model backs this client."""
        return self.provider in {"anthropic", "openai"}

    def label(self) -> str:
        """Human-readable provider/model badge for the dashboard."""
        if self.provider == "fallback":
            return "Rule-based fallback (no API key)"
        return f"{self.provider} · {self.model}"

    # -- generation --------------------------------------------------------

    def generate(self, system: str, user_prompt: str) -> str | None:
        """Return raw model text, or None to signal the caller to fall back."""
        try:
            if self.provider == "anthropic":
                return self._anthropic(system, user_prompt)
            if self.provider == "openai":
                return self._openai(system, user_prompt)
        except Exception:
            # Never let a transient API error break the dashboard.
            return None
        return None

    def _anthropic(self, system: str, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def _openai(self, system: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
