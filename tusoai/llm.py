from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import tempfile
import time

Provider = Literal["openai", "claude"]

GENERAL_CLIENT: Any = None
GENERAL_PROVIDER: Provider = "openai"
GENERAL_TEMP: float = 1.0
GENERAL_MODEL: str = ""
GENERAL_THINKING: bool = False
GENERAL_THINKING_TOKENS: int = 0
GENERAL_REASONING_MODE: Optional[str] = None
GENERAL_MAX_TOKENS: Optional[int] = None


def configure_default_client(
    client: Any,
    *,
    provider: Provider,
    model: str,
    temperature: float = 1.0,
    thinking: bool = False,
    thinking_tokens: int = 0,
    reasoning_mode: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> None:
    global GENERAL_CLIENT, GENERAL_PROVIDER, GENERAL_MODEL, GENERAL_TEMP, GENERAL_THINKING
    global GENERAL_THINKING_TOKENS, GENERAL_REASONING_MODE, GENERAL_MAX_TOKENS
    GENERAL_CLIENT = client
    GENERAL_PROVIDER = provider
    GENERAL_MODEL = model
    GENERAL_TEMP = temperature
    GENERAL_THINKING_TOKENS = int(thinking_tokens or 0)
    GENERAL_THINKING = bool(thinking and GENERAL_THINKING_TOKENS > 0)
    GENERAL_REASONING_MODE = reasoning_mode
    GENERAL_MAX_TOKENS = max_tokens


class Tusoai:
    def __init__(
        self,
        *,
        client: Any,
        provider: Provider,
        temperature: float = 1.0,
        model_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        model_settings = model_settings or {
            "pdf": {"model": "gpt-5.4-nano"},
            "construction": {"model": "gpt-5.4"},
            "optimization": {"model": "gpt-5.4-mini"},
        }

        required = {"pdf", "construction", "optimization"}
        missing = [k for k in required if k not in model_settings]
        if missing:
            raise ValueError(f"Missing model_settings entries: {missing}")

        allowed_reasoning = {None, "minimal", "low", "medium", "high", "xhigh"}

        def _norm(stage: str) -> Dict[str, Any]:
            cfg = dict(model_settings.get(stage) or {})
            model = str(cfg.get("model") or "").strip()
            if not model:
                raise ValueError(f"model_settings['{stage}'].model must be provided")
            thinking_tokens = int(cfg.get("thinking_tokens", 0) or 0)
            thinking = bool(cfg.get("thinking", False) and thinking_tokens > 0)
            reasoning_mode = cfg.get("reasoning_mode")
            if reasoning_mode not in allowed_reasoning:
                raise ValueError(
                    f"Invalid reasoning_mode for '{stage}': {reasoning_mode}. "
                    f"Use one of {sorted(x for x in allowed_reasoning if x is not None)} or None."
                )
            return {
                "model": model,
                "thinking": thinking,
                "thinking_tokens": thinking_tokens,
                "reasoning_mode": reasoning_mode,
            }

        pdf_cfg = _norm("pdf")
        construction_cfg = _norm("construction")
        optimization_cfg = _norm("optimization")

        def _model_matches_provider(model_name: str) -> bool:
            m = model_name.lower()
            if provider == "openai":
                return m.startswith("gpt") or m.startswith("o")
            if provider == "claude":
                return m.startswith("claude")
            return False

        for stage, cfg in (
            ("pdf", pdf_cfg),
            ("construction", construction_cfg),
            ("optimization", optimization_cfg),
        ):
            if not _model_matches_provider(cfg["model"]):
                raise ValueError(
                    f"model_settings['{stage}'].model='{cfg['model']}' is not valid for provider '{provider}'"
                )

        self.client = client
        self.provider = provider
        self.model = optimization_cfg["model"]
        self.pdf_model = pdf_cfg["model"]
        self.construction_model = construction_cfg["model"]
        self.optimization_model = optimization_cfg["model"]
        self.model_settings = {
            "pdf": pdf_cfg,
            "construction": construction_cfg,
            "optimization": optimization_cfg,
        }
        self.temperature = temperature
        self.thinking_tokens = int(optimization_cfg["thinking_tokens"])
        self.thinking = bool(optimization_cfg["thinking"])
        self.reasoning_mode = optimization_cfg["reasoning_mode"]

        self.pdf_thinking_tokens = int(pdf_cfg["thinking_tokens"])
        self.pdf_thinking = bool(pdf_cfg["thinking"])
        self.pdf_reasoning_mode = pdf_cfg["reasoning_mode"]

        self.construction_thinking_tokens = int(construction_cfg["thinking_tokens"])
        self.construction_thinking = bool(construction_cfg["thinking"])
        self.construction_reasoning_mode = construction_cfg["reasoning_mode"]

        self.optimization_thinking_tokens = int(optimization_cfg["thinking_tokens"])
        self.optimization_thinking = bool(optimization_cfg["thinking"])
        self.optimization_reasoning_mode = optimization_cfg["reasoning_mode"]
        self.max_tokens = max_tokens
        self._prev: Optional[Tuple[Any, Provider, str, float, bool, int, Optional[str], Optional[int]]] = None

    @classmethod
    def from_api_key(
        cls,
        *,
        api_key: str,
        provider: Provider,
        temperature: float = 1.0,
        model_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        openai_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Tusoai":
        client = init(
            api_key,
            provider=provider,
            openai_kwargs=openai_kwargs,
        )
        return cls(
            client=client,
            provider=provider,
            temperature=temperature,
            model_settings=model_settings,
            max_tokens=max_tokens,
        )

    def activate(self) -> None:
        self.activate_construction()

    def _activate_with_model(
        self,
        model: str,
        *,
        thinking: bool,
        thinking_tokens: int,
        reasoning_mode: Optional[str],
    ) -> None:
        self._prev = (
            GENERAL_CLIENT,
            GENERAL_PROVIDER,
            GENERAL_MODEL,
            GENERAL_TEMP,
            GENERAL_THINKING,
            GENERAL_THINKING_TOKENS,
            GENERAL_REASONING_MODE,
            GENERAL_MAX_TOKENS,
        )
        configure_default_client(
            self.client,
            provider=self.provider,
            model=model,
            temperature=self.temperature,
            thinking=thinking,
            thinking_tokens=thinking_tokens,
            reasoning_mode=reasoning_mode,
            max_tokens=self.max_tokens,
        )

    def activate_construction(self) -> None:
        self._activate_with_model(
            self.construction_model,
            thinking=self.construction_thinking,
            thinking_tokens=self.construction_thinking_tokens,
            reasoning_mode=self.construction_reasoning_mode,
        )

    def activate_optimization(self) -> None:
        self._activate_with_model(
            self.optimization_model,
            thinking=self.optimization_thinking,
            thinking_tokens=self.optimization_thinking_tokens,
            reasoning_mode=self.optimization_reasoning_mode,
        )

    @contextmanager
    def use_construction_model(self):
        self.activate_construction()
        try:
            yield self
        finally:
            self.deactivate()

    @contextmanager
    def use_optimization_model(self):
        self.activate_optimization()
        try:
            yield self
        finally:
            self.deactivate()

    def deactivate(self) -> None:
        if self._prev is None:
            return
        client, provider, model, temperature, thinking, thinking_tokens, reasoning_mode, max_tokens = self._prev
        configure_default_client(
            client,
            provider=provider,
            model=model,
            temperature=temperature,
            thinking=thinking,
            thinking_tokens=thinking_tokens,
            reasoning_mode=reasoning_mode,
            max_tokens=max_tokens,
        )
        self._prev = None

    def __enter__(self) -> "Tusoai":
        self.activate_construction()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.deactivate()

    def create_method_subtask(self, *args, **kwargs):
        from tusoai.subtasks import create_method_subtask

        return create_method_subtask(*args, tusoai=self, **kwargs)

    def create_data_subtask(self, *args, **kwargs):
        from tusoai.subtasks import create_data_subtask

        return create_data_subtask(*args, tusoai=self, **kwargs)

    def optimize(self, *args, **kwargs):
        from tusoai.optimization import optimize

        self.activate_optimization()
        try:
            return optimize(*args, **kwargs)
        finally:
            self.deactivate()


def init(
    api_key: str,
    provider: Provider = "openai",
    *,
    openai_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create and return an OpenAI or Claude SDK client."""
    openai_kwargs = openai_kwargs or {}

    if provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=api_key, **openai_kwargs)

    if provider == "claude":
        from anthropic import Anthropic

        return Anthropic(api_key=api_key)

    raise ValueError(f"Unknown provider: {provider}")


def _extract_claude_text(message: Any) -> str:
    """
    Extract a plain text response from an Anthropic Messages API result.

    The Anthropic SDK typically returns `message.content` as a list of content blocks
    (e.g., TextBlock objects), though some environments may provide dicts or strings.

    Supported shapes:
        - message.content: List[TextBlock-like] where blocks may have attributes:
            - block.type == "text" and block.text contains the string
        - message.content: List[dict] where dict["type"] == "text" and dict["text"] exists
        - message.content: str

    Args:
        message: The object returned by `client.messages.create(...)`.

    Returns:
        A single concatenated string of all text blocks, stripped.
    """
    parts: list[str] = []
    content = getattr(message, "content", None)

    if isinstance(content, list):
        for block in content:
            if (
                hasattr(block, "type")
                and getattr(block, "type") == "text"
                and hasattr(block, "text")
            ):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    elif isinstance(content, str):
        parts.append(content)

    return "".join(parts).strip()


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
    """Safely traverse nested attributes and/or dict keys using a dotted path."""
    cur: Any = obj
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
            continue
        if hasattr(cur, part):
            cur = getattr(cur, part)
            continue
        return default
    return cur


def _per_million_cost(tokens: int, rate_per_1m: float) -> float:
    return (float(tokens) / 1_000_000.0) * float(rate_per_1m)


def _get_rates_usd_per_1m(
    provider: Provider,
    model: str,
    *,
    input_tokens: Optional[int] = None,
) -> Dict[str, float]:
    m = model.lower().strip()

    if provider == "openai":
        if m == "gpt-4o-mini" or "gpt-4o-mini" in m:
            return {"input": 0.15, "cached_input": 0.075, "output": 0.60}

        if m == "gpt-5.4-nano" or "gpt-5.4-nano" in m:
            return {"input": 0.20, "cached_input": 0.020, "output": 1.25}

        if m.startswith("gpt-5.4-mini"):
            return {"input": 0.75, "cached_input": 0.075, "output": 4.50}

        if m.startswith("gpt-5.4"):
            return {"input": 2.50, "cached_input": 0.250, "output": 15.00}

        if m.startswith("gpt-5.1-codex"):
            return {"input": 1.25, "cached_input": 0.125, "output": 10.00}

        raise ValueError(f"Unsupported OpenAI model for cost calc: {model}")

    if provider == "claude":
        if m.startswith("claude-opus-4-6") or m.startswith("claude-opus-4-5"):
            return {"input": 5.00, "cached_input": 0.50, "output": 25.00}
        if m.startswith("claude-sonnet-4-6") or m.startswith("claude-sonnet-4-5"):
            return {"input": 3.00, "cached_input": 0.30, "output": 15.00}
        if m.startswith("claude-haiku-4-5"):
            return {"input": 1.00, "cached_input": 0.10, "output": 5.00}

        if m.startswith("claude-3-5-sonnet"):
            return {"input": 3.00, "cached_input": 0.30, "output": 15.00}
        if m.startswith("claude-3-5-haiku"):
            return {"input": 0.80, "cached_input": 0.08, "output": 4.00}
        if m.startswith("claude-3-opus"):
            return {"input": 15.00, "cached_input": 1.50, "output": 75.00}
        if m.startswith("claude-3-sonnet"):
            return {"input": 3.00, "cached_input": 0.30, "output": 15.00}
        if m.startswith("claude-3-haiku"):
            return {"input": 0.25, "cached_input": 0.03, "output": 1.25}

        raise ValueError(f"Unsupported Claude model for cost calc: {model}")

    raise ValueError(f"Unsupported provider for cost calc: {provider}")


def _estimate_cost_usd(
    *,
    provider: Provider,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    rates = _get_rates_usd_per_1m(provider, model, input_tokens=input_tokens)

    cached_tokens = min(int(cached_input_tokens or 0), int(input_tokens or 0))
    uncached_tokens = max(int(input_tokens or 0) - cached_tokens, 0)

    total = 0.0
    total += _per_million_cost(uncached_tokens, rates["input"])

    if cached_tokens and "cached_input" in rates:
        total += _per_million_cost(cached_tokens, rates["cached_input"])
    elif cached_tokens:
        total += _per_million_cost(cached_tokens, rates["input"])

    total += _per_million_cost(int(output_tokens or 0), rates["output"])
    return total


def run_prompt_full(
    *,
    prompt: str,
    client: Any,
    temperature: float,
    model: str,
    provider: Provider,
    max_tokens: Optional[int] = None,
    thinking: bool = False,
    thinking_tokens: int = 0,
    reasoning_mode: Optional[str] = None,
) -> Tuple[str, float]:
    if "gpt-5" in model or thinking:
        temperature = 1.0
    if thinking_tokens <= 0:
        thinking = False

    try:
        if provider == "openai":
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "input": prompt,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_output_tokens"] = max_tokens
                if reasoning_mode:
                    kwargs["reasoning"] = {"effort": str(reasoning_mode)}

                resp = client.responses.create(**kwargs)
                text = (getattr(resp, "output_text", "") or "").strip()

                usage = _safe_get(resp, "usage", None)
                in_tok = int(_safe_get(usage, "input_tokens", 0) or 0)
                out_tok = int(_safe_get(usage, "output_tokens", 0) or 0)
                cached_in = int(
                    _safe_get(usage, "input_tokens_details.cached_tokens", 0) or 0
                )

                cost = (
                    _estimate_cost_usd(
                        provider=provider,
                        model=model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cached_input_tokens=cached_in,
                    )
                    if (in_tok or out_tok)
                    else 0.0
                )
                return text, cost

            kwargs = {
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()

            usage = _safe_get(resp, "usage", None)
            in_tok = int(_safe_get(usage, "prompt_tokens", 0) or 0)
            out_tok = int(_safe_get(usage, "completion_tokens", 0) or 0)
            cached_in = int(_safe_get(usage, "prompt_tokens_details.cached_tokens", 0) or 0)

            cost = (
                _estimate_cost_usd(
                    provider=provider,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cached_input_tokens=cached_in,
                )
                if (in_tok or out_tok)
                else 0.0
            )
            return text, cost

        if provider == "claude":
            if thinking:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens or 1024,
                    "thinking": {"type": "enabled", "budget_tokens": int(thinking_tokens)},
                }
            else:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens or 1024,
                }
            if temperature is not None:
                kwargs["temperature"] = temperature

            msg = client.messages.create(**kwargs)
            text = _extract_claude_text(msg)

            usage = _safe_get(msg, "usage", None)
            in_tok = int(_safe_get(usage, "input_tokens", 0) or 0)
            out_tok = int(_safe_get(usage, "output_tokens", 0) or 0)

            cost = (
                _estimate_cost_usd(
                    provider=provider,
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cached_input_tokens=0,
                )
                if (in_tok or out_tok)
                else 0.0
            )
            return text, cost

        raise ValueError(f"Unknown provider: {provider}")

    except Exception as e:  # noqa: BLE001
        print("LLM call failed:", e)
        return "", 0.0



def _extract_openai_response_text(resp_body: Any) -> str:
    try:
        out = resp_body.get("output", []) if isinstance(resp_body, dict) else []
        texts: List[str] = []
        for item in out or []:
            for c in (item.get("content", []) if isinstance(item, dict) else []):
                t = c.get("text") if isinstance(c, dict) else None
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        if texts:
            return "\n".join(texts).strip()
    except Exception:
        pass
    return (str((resp_body or {}).get("output_text", "")) if isinstance(resp_body, dict) else "").strip()


def submit_prompt_batch(
    prompts: List[str],
    *,
    client: Any,
    provider: Provider,
    model: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    thinking: bool = False,
    thinking_tokens: int = 0,
    reasoning_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if not prompts:
        raise ValueError("prompts must be non-empty")

    custom_ids = [f"req_{i}" for i in range(len(prompts))]

    if provider == "openai":
        lines = []
        for cid, prompt in zip(custom_ids, prompts):
            body: Dict[str, Any] = {"model": model, "input": prompt, "temperature": temperature}
            if max_tokens is not None:
                body["max_output_tokens"] = max_tokens
            if reasoning_mode:
                body["reasoning"] = {"effort": str(reasoning_mode)}
            lines.append({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body})

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
            for row in lines:
                tf.write(json.dumps(row) + "\n")
            temp_path = tf.name

        try:
            with open(temp_path, "rb") as fh:
                input_file = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(input_file_id=getattr(input_file, "id", None), endpoint="/v1/responses", completion_window="24h")
        finally:
            try:
                import os
                os.unlink(temp_path)
            except Exception:
                pass
        return {
            "provider": provider,
            "batch_id": getattr(batch, "id", None),
            "custom_ids": custom_ids,
            "submitted_at": time.time(),
            "poll_after": time.time() + 2.0,
        }

    if provider == "claude":
        reqs: List[Dict[str, Any]] = []
        for cid, prompt in zip(custom_ids, prompts):
            params: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens or 1024,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if thinking and thinking_tokens > 0:
                params["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_tokens)}
            reqs.append({"custom_id": cid, "params": params})

        b = client.messages.batches.create(requests=reqs)
        return {
            "provider": provider,
            "batch_id": getattr(b, "id", None),
            "custom_ids": custom_ids,
            "submitted_at": time.time(),
            "poll_after": time.time() + 2.0,
        }

    raise ValueError(f"Unknown provider: {provider}")


def poll_prompt_batch(
    handle: Dict[str, Any],
    *,
    client: Any,
    provider: Provider,
    model: str,
) -> Tuple[bool, Dict[str, Tuple[str, float]]]:
    if not handle:
        return False, {}
    if time.time() < float(handle.get("poll_after", 0.0)):
        return False, {}
    handle["poll_after"] = time.time() + 2.0

    batch_id = handle.get("batch_id")
    if not batch_id:
        return True, {}

    # returns mapping custom_id -> (text, cost)
    out: Dict[str, Tuple[str, float]] = {}

    if provider == "openai":
        b = client.batches.retrieve(batch_id)
        status = str(getattr(b, "status", ""))
        if status not in {"completed", "failed", "expired", "cancelled"}:
            return False, {}
        if status != "completed":
            return True, {}

        output_file_id = getattr(b, "output_file_id", None)
        if not output_file_id:
            return True, {}
        content_obj = client.files.content(output_file_id)
        text_blob = getattr(content_obj, "text", None)
        if text_blob is None:
            raw = getattr(content_obj, "content", b"")
            text_blob = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        for line in str(text_blob).splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("custom_id", ""))
            body = (((row.get("response") or {}).get("body") or {}) if isinstance(row, dict) else {})
            msg = _extract_openai_response_text(body)
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            in_tok = int((usage.get("input_tokens") or 0))
            out_tok = int((usage.get("output_tokens") or 0))
            cached = int((((usage.get("input_tokens_details") or {}).get("cached_tokens")) or 0))
            cost = _estimate_cost_usd(provider=provider, model=model, input_tokens=in_tok, output_tokens=out_tok, cached_input_tokens=cached) * 0.5
            out[cid] = (msg, float(cost))
        return True, out

    if provider == "claude":
        b = client.messages.batches.retrieve(batch_id)
        status = str(getattr(b, "processing_status", ""))
        if status not in {"ended", "failed", "canceled", "expired"}:
            return False, {}
        if status != "ended":
            return True, {}

        try:
            results_iter = client.messages.batches.results(batch_id)
        except Exception:
            results_iter = []

        for item in results_iter:
            cid = str(getattr(item, "custom_id", ""))
            result = getattr(item, "result", None)
            if getattr(result, "type", "") != "succeeded":
                continue
            msg_obj = getattr(result, "message", None)
            text = _extract_claude_text(msg_obj)
            usage = _safe_get(msg_obj, "usage", None)
            in_tok = int(_safe_get(usage, "input_tokens", 0) or 0)
            out_tok = int(_safe_get(usage, "output_tokens", 0) or 0)
            cost = _estimate_cost_usd(provider=provider, model=model, input_tokens=in_tok, output_tokens=out_tok, cached_input_tokens=0) * 0.5
            out[cid] = (text, float(cost))
        return True, out

    return True, {}


def submit_default_prompt_batch(prompts: List[str]) -> Dict[str, Any]:
    if GENERAL_CLIENT is None:
        raise RuntimeError("Default LLM client is not configured. Call configure_default_client().")
    return submit_prompt_batch(
        prompts,
        client=GENERAL_CLIENT,
        provider=GENERAL_PROVIDER,
        model=GENERAL_MODEL,
        temperature=GENERAL_TEMP,
        max_tokens=GENERAL_MAX_TOKENS,
        thinking=GENERAL_THINKING,
        thinking_tokens=GENERAL_THINKING_TOKENS,
        reasoning_mode=GENERAL_REASONING_MODE,
    )


def poll_default_prompt_batch(handle: Dict[str, Any]) -> Tuple[bool, Dict[str, Tuple[str, float]]]:
    if GENERAL_CLIENT is None:
        raise RuntimeError("Default LLM client is not configured. Call configure_default_client().")
    return poll_prompt_batch(handle, client=GENERAL_CLIENT, provider=GENERAL_PROVIDER, model=GENERAL_MODEL)

def run_prompt(message: str) -> Tuple[str, float]:
    if GENERAL_CLIENT is None:
        raise RuntimeError("Default LLM client is not configured. Call configure_default_client().")
    return run_prompt_full(
        prompt=message,
        client=GENERAL_CLIENT,
        temperature=GENERAL_TEMP,
        model=GENERAL_MODEL,
        provider=GENERAL_PROVIDER,
        thinking=GENERAL_THINKING,
        thinking_tokens=GENERAL_THINKING_TOKENS,
        reasoning_mode=GENERAL_REASONING_MODE,
        max_tokens=GENERAL_MAX_TOKENS,
    )


__all__ = [
    "Tusoai",
    "Provider",
    "configure_default_client",
    "init",
    "run_prompt",
    "run_prompt_full",
    "submit_default_prompt_batch",
    "poll_default_prompt_batch",
    "submit_prompt_batch",
    "poll_prompt_batch",
]
