#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

_DEFAULT_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_completions_url() -> str:
    """OPENROUTER_BASE_URL: .../v1 -> .../v1/chat/completions."""
    base = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    if not base:
        return _DEFAULT_CHAT_COMPLETIONS_URL
    b = base.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return f"{b}/chat/completions"


def resolve_llm_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


SYSTEM_MESSAGE = (
    "You are an expert in computational chemistry."
)


@dataclass(frozen=True)
class ModelPreset:
    id: str
    model: str
    temperature: float
    reasoning_effort: str
    reasoning_kind: str = "effort"
    supported_params: frozenset[str] | None = None


SUPPORTED_PARAMS: dict[str, frozenset[str]] = {
    "anthropic/claude-opus-5": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "stop",
        "structured_outputs", "temperature", "tool_choice", "tools", "verbosity",
    }),
    "anthropic/claude-sonnet-5": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "stop",
        "structured_outputs", "temperature", "tool_choice", "tools", "verbosity",
    }),
    "openai/gpt-6-astra": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "seed",
        "structured_outputs", "tool_choice", "tools",
    }),
    "openai/gpt-5.6-sol": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "seed",
        "structured_outputs", "tool_choice", "tools",
    }),
    "google/gemini-3.6-flash": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "reasoning_effort",
        "response_format", "seed", "stop", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_p",
    }),
    "qwen/qwen3.7-max": frozenset({
        "include_reasoning", "logprobs", "max_tokens", "presence_penalty",
        "reasoning", "response_format", "seed", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_logprobs", "top_p",
    }),
    "qwen/qwen3.5-plus-20260420": frozenset({
        "include_reasoning", "logprobs", "max_tokens", "presence_penalty",
        "reasoning", "response_format", "seed", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_logprobs", "top_p",
    }),
    "google/gemini-3-flash-preview": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "reasoning_effort",
        "response_format", "seed", "stop", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_p",
    }),
    "anthropic/claude-sonnet-4.5": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "response_format", "stop", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_k", "top_p",
    }),
    "openai/gpt-5": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "seed",
        "structured_outputs", "tool_choice", "tools",
    }),
    "google/gemini-2.5-flash": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "response_format",
        "seed", "stop", "structured_outputs", "temperature", "tool_choice",
        "tools", "top_p",
    }),
    "moonshotai/kimi-k3": frozenset({
        "include_reasoning", "max_completion_tokens", "max_tokens",
        "reasoning", "reasoning_effort", "response_format", "stop",
        "structured_outputs", "temperature", "tool_choice", "tools", "verbosity",
    }),
    "moonshotai/kimi-k2-thinking": frozenset({
        "frequency_penalty", "include_reasoning", "logprobs", "max_tokens",
        "presence_penalty", "reasoning", "repetition_penalty", "response_format",
        "seed", "stop", "structured_outputs", "temperature", "tool_choice",
        "tools", "top_k", "top_logprobs", "top_p",
    }),
    "openai/o4-mini": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "response_format",
        "seed", "structured_outputs", "tool_choice", "tools",
    }),
    "deepseek/deepseek-v4-flash-0731": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "reasoning_effort",
        "response_format", "seed", "stop", "structured_outputs",
        "temperature", "tool_choice", "tools", "top_p",
    }),
    "deepseek/deepseek-v3.2": frozenset({
        "frequency_penalty", "include_reasoning", "logit_bias", "logprobs",
        "max_tokens", "min_p", "presence_penalty", "reasoning",
        "repetition_penalty", "response_format", "seed", "stop",
        "structured_outputs", "temperature", "tool_choice", "tools",
        "top_k", "top_logprobs", "top_p",
    }),
    "anthropic/claude-sonnet-4": frozenset({
        "include_reasoning", "max_tokens", "reasoning", "stop",
        "temperature", "tool_choice", "tools", "top_k", "top_p",
    }),
}

REASONING_EFFORTS: dict[str, list[str]] = {
    "anthropic/claude-opus-5": ["max", "xhigh", "high", "medium", "low"],
    "anthropic/claude-sonnet-5": ["max", "xhigh", "high", "medium", "low"],
    "openai/gpt-5.6-sol": ["xhigh", "high", "medium", "low", "none"],
    "openai/gpt-6-astra": ["max","xhigh","high","medium","low"],
    "google/gemini-3.6-flash": ["high", "medium", "low", "minimal"],
    "qwen/qwen3.7-max": [],
    "qwen/qwen3.5-plus-20260420": [],
    "google/gemini-3-flash-preview": ["high", "medium", "low", "minimal"],
    "anthropic/claude-sonnet-4.5": [],
    "openai/gpt-5": ["high", "medium", "low", "minimal"],
    "google/gemini-2.5-flash": [],
    "moonshotai/kimi-k3": ["high", "medium", "low"],
    "moonshotai/kimi-k2-thinking": [],
    "openai/o4-mini": [],
    "deepseek/deepseek-v4-flash-0731": ["high", "medium", "low", "minimal"],
    "deepseek/deepseek-v3.2": [],
    "anthropic/claude-sonnet-4": [],
}


_MODELS_GROUP_2: frozenset[str] = frozenset({
    "qwen/qwen3.7-max",
    "qwen/qwen3.5-plus-20260420",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "moonshotai/kimi-k2-thinking",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "anthropic/claude-sonnet-4",
})

_MODELS_GROUP_3: frozenset[str] = frozenset({
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.6-sol",
    "google/gemini-3.6-flash",
    "google/gemini-3-flash-preview",
    "openai/gpt-5",
    "moonshotai/kimi-k3",
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-6-astra"
})


def model_supports(preset: ModelPreset, param: str) -> bool:
    if preset.supported_params is None:
        return True
    return param in preset.supported_params


def get_model_reasoning_kind(model_id: str) -> str | None:
    """Return reasoning_kind for a model: 'effort', 'enabled', or None."""
    mid = model_id.strip()
    if mid in _MODELS_GROUP_2:
        return "enabled"
    if mid in _MODELS_GROUP_3:
        return "effort"
    sp = SUPPORTED_PARAMS.get(mid)
    if sp is None:
        return "effort"
    if "reasoning_effort" in sp:
        return "effort"
    if "reasoning" in sp:
        return "enabled"
    return None


def get_default_reasoning_effort(model_id: str) -> str:
    """Return default reasoning effort for a model (only meaningful for group 3)."""
    mid = model_id.strip()
    if mid in _MODELS_GROUP_3:
        efforts = REASONING_EFFORTS.get(mid, [])
        if "high" in efforts:
            return "high"
        if efforts:
            return efforts[0]
    return "high"


def _json_dumps_debug(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def call_openrouter(
    api_key: str,
    preset: ModelPreset,
    user_content: str,
    timeout: float = 120.0,
    *,
    use_system_prompt: bool = True,
    system_message: str | None = None,
    omit_reasoning: bool = False,
    max_tokens: int | None = None,
) -> tuple[str | None, str | None, str | None, dict | None]:
    """
    Returns (assistant_text, error_message, raw_response, api_data).

    On success: (text, None, None, data).
    On failure: (None, error, raw, data) where raw is response body or parsed JSON string when available.
    data is the full parsed JSON response from OpenRouter (contains id, usage, choices, etc.)

    If use_system_prompt is True: uses system_message when given, otherwise SYSTEM_MESSAGE.
    """
    data: dict | None = None
    if use_system_prompt:
        sys_text = system_message if system_message is not None else SYSTEM_MESSAGE
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]
    body: dict[str, Any] = {
        "model": preset.model,
        "messages": messages,
    }
    if model_supports(preset, "temperature"):
        body["temperature"] = preset.temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if not omit_reasoning:
        if getattr(preset, "reasoning_kind", "effort") == "enabled":
            if model_supports(preset, "reasoning"):
                body["reasoning"] = {"enabled": True}
        else:
            if model_supports(preset, "reasoning"):
                body["reasoning"] = {"effort": preset.reasoning_effort}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/benchmark_ranking",
        "X-Title": "Conformer ranking benchmark",
    }
    try:
        r = requests.post(
            chat_completions_url(), headers=headers, json=body, timeout=timeout
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raw = None
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                raw = resp.text
            except (OSError, UnicodeDecodeError):
                pass
        return None, str(e), raw, data
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON from API: {e}", r.text, data
    err = data.get("error")
    if err:
        return None, json.dumps(err), _json_dumps_debug(data), data
    choices = data.get("choices") or []
    if not choices:
        return None, "No choices in response", _json_dumps_debug(data), data
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        return None, "Empty message content", _json_dumps_debug(data), data
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "".join(text_parts)
    text = str(content).strip()
    if not text:
        return None, "Empty message content", _json_dumps_debug(data), data
    return text, None, None, data
