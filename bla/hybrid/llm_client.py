"""LLM client abstraction: real (Anthropic) + mock (offline tests).

The real client uses `claude-opus-4-7` per the project's standing default,
with prompt caching on the system prompt (since the conceptual-object-JEPA
system prompt is large and stable across a session). All API parameters
follow the claude-api skill's Opus 4.7 conventions: no temperature/top_p,
adaptive thinking, effort=high.

Credential lookup order (first hit wins):
  1. `api_key=` kwarg passed to AnthropicLLMClient()
  2. `ANTHROPIC_API_KEY` env var
  3. `.bla_secrets.json` at the repo root (chmod 600, gitignored)

The mock client is used in tests so the suite runs offline.
"""
from __future__ import annotations

import abc
import json
import os
from pathlib import Path
from typing import Any, Optional


_SECRETS_FILENAME = ".bla_secrets.json"


def _load_api_key_from_secrets_file() -> Optional[str]:
    """Read anthropic_api_key from .bla_secrets.json if present.

    Searches the file in the repo root (relative to this module) and the
    user's home directory. Returns None if not found or unreadable. This
    is a developer-convenience fallback — production deployments should
    set ANTHROPIC_API_KEY in the environment instead.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / _SECRETS_FILENAME,
        Path.home() / _SECRETS_FILENAME,
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = data.get("anthropic_api_key")
        if isinstance(key, str) and key.startswith("sk-ant-"):
            return key
    return None


class LLMClient(abc.ABC):
    """Minimal client surface the hybrid loop depends on."""

    @abc.abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        json_only: bool = False,
    ) -> str:
        """Run one completion; return assistant text.

        Args:
          system:    full system prompt.
          user:      user message text.
          max_tokens: response cap.
          json_only: if True, the prompt asks the model for raw JSON only
                     and the caller will `json.loads()` the result. The
                     real client also enables structured-output mode when
                     a schema is provided (not used in MVP).
        """


class AnthropicLLMClient(LLMClient):
    """Real client using the Anthropic SDK.

    Defaults follow claude-api skill recommendations for Opus 4.7:
      - model: claude-opus-4-7
      - thinking: adaptive
      - effort: high
      - no temperature/top_p (would 400 on 4.7)
      - prompt caching on the system block via cache_control
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
        effort: str = "high",
        thinking_adaptive: bool = True,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "AnthropicLLMClient requires the `anthropic` package. "
                "Install with `pip install anthropic`.") from e
        self._anthropic = anthropic
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = _load_api_key_from_secrets_file()
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in the "
                "environment, pass api_key= explicitly, or place a "
                f"{_SECRETS_FILENAME} file (chmod 600) at the repo root "
                "with key 'anthropic_api_key'.")
        self.model = model
        self.effort = effort
        self.thinking_adaptive = thinking_adaptive
        self._client = anthropic.Anthropic(api_key=api_key)
        # Per-call accounting, cumulative across the lifetime of this client.
        # Bench scripts read these to compute total token spend.
        self.last_usage: dict[str, int] = {}
        self.total_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "n_calls": 0,
        }

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        json_only: bool = False,
    ) -> str:
        # Cache the (large, stable) system prompt; keep user volatile.
        system_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        if json_only:
            user = (
                user.rstrip()
                + "\n\nRespond with raw JSON only, no markdown fences, no "
                "prose around it."
            )
        base_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user}],
        }
        # Optional modern kwargs — older SDKs don't accept these. Try with,
        # fall back without on TypeError. (Anthropic SDK >=0.80 accepts
        # output_config + adaptive thinking; 0.76 does not.)
        modern_kwargs: dict[str, Any] = {}
        if self.effort:
            modern_kwargs["output_config"] = {"effort": self.effort}
        if self.thinking_adaptive:
            modern_kwargs["thinking"] = {"type": "adaptive"}

        try:
            resp = self._client.messages.create(**base_kwargs, **modern_kwargs)
        except TypeError as e:
            # Older SDK — strip the unsupported kwargs and retry once.
            if "unexpected keyword argument" not in str(e):
                raise
            resp = self._client.messages.create(**base_kwargs)
        except Exception as e:
            # Model rejected one of the optional kwargs at the API level
            # (e.g. Haiku doesn't accept adaptive thinking; Opus 4.7
            # rejects effort=max if not opus-tier; etc). Retry once with
            # the base kwargs only.
            msg = str(e).lower()
            if "not supported on this model" not in msg and "thinking" not in msg:
                raise
            resp = self._client.messages.create(**base_kwargs)
        # Capture usage (best-effort; older SDKs always have .usage on Message)
        try:
            u = resp.usage
            self.last_usage = {
                "input_tokens": getattr(u, "input_tokens", 0) or 0,
                "output_tokens": getattr(u, "output_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(
                    u, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(
                    u, "cache_read_input_tokens", 0) or 0,
            }
            for k, v in self.last_usage.items():
                self.total_usage[k] += v
            self.total_usage["n_calls"] += 1
        except AttributeError:
            self.last_usage = {}
        text_parts = [b.text for b in resp.content if b.type == "text"]
        return "".join(text_parts).strip()


class MockLLMClient(LLMClient):
    """Scripted client for offline tests.

    Provide a list of canned responses; each `complete()` call pops the
    next one. Useful for end-to-end loop tests where we don't want a
    live API call.
    """

    def __init__(self, scripted_responses: list[str]):
        self._responses: list[str] = list(scripted_responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        json_only: bool = False,
    ) -> str:
        self.calls.append({
            "system": system, "user": user,
            "max_tokens": max_tokens, "json_only": json_only,
        })
        if not self._responses:
            raise RuntimeError(
                "MockLLMClient ran out of scripted responses "
                f"(after {len(self.calls)} calls).")
        return self._responses.pop(0)


def parse_json_response(text: str) -> Any:
    """Forgiving JSON parser for LLM output.

    Strips ```json fences and surrounding prose if the model ignored
    the json_only instruction. Returns the parsed object or raises
    ValueError with the offending text.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop opening fence (with or without language tag) + closing fence
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # If extraneous prose still wraps the JSON, find the outermost {...} or [...]
    if not (s.startswith("{") or s.startswith("[")):
        for opener, closer in [("{", "}"), ("[", "]")]:
            i = s.find(opener)
            j = s.rfind(closer)
            if i != -1 and j > i:
                s = s[i:j + 1]
                break
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {e}\n--- TEXT ---\n{text}"
        ) from e
