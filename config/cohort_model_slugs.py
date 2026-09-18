"""OpenRouter / Modelgrep slug registry for geom-cohort models."""

from __future__ import annotations

COHORT_MODEL_SLUGS: dict[str, str | None] = {
    "Claude Opus 5": "anthropic/claude-opus-5",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5",
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Claude Sonnet 4": "anthropic/claude-sonnet-4",
    "GPT-5.6 Sol": "openai/gpt-5.6-sol",
    "GPT-6 Astra": "openai/gpt-6-astra",
    "GPT-5": "openai/gpt-5",
    "O4 Mini": "openai/o4-mini",
    "Gemini 3.6 Flash": "google/gemini-3.6-flash",
    "Gemini 3 Flash": "google/gemini-3-flash-preview",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemma 3 27B": "google/gemma-3-27b-it",
    "Gemma 4 31B": "google/gemma-4-31b-it",
    "Kimi K3": "moonshotai/kimi-k3",
    "Kimi K2 Thinking": "moonshotai/kimi-k2-thinking",
    "Qwen 3.8 27B": "qwen/qwen3.8-27b",
    "Qwen 3.7 Max": "qwen/qwen3.7-max",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
    "Qwen 3.5 27B": "qwen/qwen3.5-27b",
    "Qwen 3.5 Plus": None,
    "Qwen 2.5 32B": None,
    "Qwen 3 32B": "qwen/qwen3-32b",
    "Qwen 3.5 35B A3B": "qwen/qwen3.5-35b-a3b",
    "Qwen 3.6 35B A3B": "qwen/qwen3.6-35b-a3b",
    "DeepSeek V4 Flash 0731": "deepseek/deepseek-v4-flash-0731",
    "DeepSeek-R1 32B": None,
    "DeepSeek V3.2": "deepseek/deepseek-v3.2",
    "Muse Glimmer 30B": "meta/muse-glimmer-30b",
}
