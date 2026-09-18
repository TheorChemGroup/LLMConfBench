#!/usr/bin/env python3

"""Unit tests for models.yaml helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import (
    MARKER_DEFAULT,
    MARKER_REASONING,
    MARKER_SIZE,
    MARKER_SIZE_LOCAL,
    MARKER_SIZE_PARAMS_MAX,
    MARKER_SIZE_PARAMS_MIN,
    arch_for_display,
    class_for_display,
    color_for_display,
    default_models_table,
    family_for_display,
    is_closed_display,
    is_local_display,
    is_os_display,
    is_reasoning_model_id,
    is_reasoning_or_thinking,
    marker_for_arch,
    marker_for_display,
    marker_for_model_id,
    models_by_id,
    models_in_group,
    params_active_for_display,
    params_total_for_display,
    scatter_size_for_params,
    scatter_style_for_display,
)


class TestReasoningMarkers(unittest.TestCase):
    def test_geom_benchmark_markers_follow_reasoning_flag(self) -> None:
        for m in models_in_group("geom_benchmark"):
            expected_reasoning = (
                m.reasoning
                if m.reasoning is not None
                else is_reasoning_or_thinking(m.display)
            )
            self.assertEqual(is_reasoning_model_id(m.id), expected_reasoning, msg=m.id)
            expected_marker = MARKER_REASONING if expected_reasoning else MARKER_DEFAULT
            self.assertEqual(marker_for_model_id(m.id), expected_marker, msg=m.id)

    def test_marker_for_display(self) -> None:
        self.assertEqual(marker_for_display("Claude Opus 5"), MARKER_REASONING)
        self.assertEqual(marker_for_display("Claude Sonnet 5"), MARKER_REASONING)

    def test_short_names(self) -> None:
        by_id = models_by_id()
        self.assertEqual(by_id["claudesonnet5"].short, "claude-sonnet-5")
        self.assertEqual(by_id["gpt56sol"].short, "gpt-5.6-sol")

    def test_colors_from_yaml(self) -> None:
        self.assertEqual(color_for_display("Claude Opus 5"), "#c2410c")
        self.assertEqual(color_for_display("GPT-5.6 Sol"), "#34d399")
        for m in models_in_group("geom_benchmark"):
            self.assertEqual(color_for_display(m.display), m.color)

    def test_gemini_and_gemma_are_distinct_families(self) -> None:
        self.assertEqual(family_for_display("Gemini 3.6 Flash"), "Gemini")
        self.assertEqual(family_for_display("Gemma 3 27B"), "Gemma")
        self.assertEqual(family_for_display("Muse Glimmer 30B"), "Meta")
        self.assertEqual(family_for_display("O4 Mini"), "GPT")
        self.assertNotEqual(
            color_for_display("Gemini 3.6 Flash"),
            color_for_display("Gemma 3 27B"),
        )

    def test_os_and_closed_classes(self) -> None:
        self.assertEqual(class_for_display("Qwen 2.5 32B"), "os")
        self.assertTrue(is_os_display("Qwen 2.5 32B"))
        self.assertTrue(is_local_display("DeepSeek-R1 32B"))
        self.assertFalse(is_os_display("Claude Opus 5"))
        self.assertTrue(is_closed_display("Claude Opus 5"))

        self.assertTrue(is_os_display("Qwen 3.5 Plus"))
        self.assertFalse(is_closed_display("Qwen 3.5 Plus"))
        self.assertFalse(is_closed_display("Kimi K3"))
        self.assertEqual(scatter_style_for_display("Qwen 2.5 32B")[0], MARKER_SIZE_LOCAL)
        self.assertEqual(scatter_style_for_display("Claude Opus 5")[0], MARKER_SIZE)
        self.assertEqual(MARKER_SIZE / MARKER_SIZE_LOCAL, 4.0)

    def test_params_and_arch_from_yaml(self) -> None:
        by_id = models_by_id()
        gemma = by_id["gemma327b"]
        self.assertEqual(gemma.params_active_b, 27.0)
        self.assertEqual(gemma.params_total_b, 27.0)
        self.assertEqual(gemma.arch, "dense")
        moe = by_id["qwen3535ba3b"]
        self.assertEqual(moe.params_total_b, 35.0)
        self.assertEqual(moe.params_active_b, 3.0)
        self.assertEqual(moe.arch, "moe")
        unpublished = by_id["claudeopus5"]
        self.assertIsNone(unpublished.params_active_b)
        self.assertIsNone(unpublished.arch)
        self.assertEqual(params_active_for_display("Gemma 3 27B"), 27.0)
        self.assertEqual(params_total_for_display("Qwen 3.5 35B A3B"), 35.0)
        self.assertEqual(params_active_for_display("Qwen 3.5 35B A3B"), 3.0)
        self.assertEqual(arch_for_display("Qwen 3.5 35B A3B"), "moe")
        self.assertIsNone(params_active_for_display("Claude Opus 5"))
        self.assertIsNone(params_total_for_display("Claude Opus 5"))

    def test_arch_markers_and_param_sizes(self) -> None:
        self.assertEqual(marker_for_arch("dense"), "o")
        self.assertEqual(marker_for_arch("moe"), "^")
        self.assertEqual(marker_for_arch("distill"), "D")
        self.assertEqual(marker_for_arch(None), "^")
        self.assertEqual(scatter_size_for_params(None), MARKER_SIZE)
        self.assertEqual(scatter_size_for_params(10.0), MARKER_SIZE_PARAMS_MIN)
        self.assertEqual(scatter_size_for_params(3000.0), MARKER_SIZE_PARAMS_MAX)
        self.assertLess(
            scatter_size_for_params(30.0),
            scatter_size_for_params(300.0),
        )
        self.assertLess(
            scatter_size_for_params(300.0),
            scatter_size_for_params(3000.0),
        )

    def test_invalid_arch_is_rejected(self) -> None:
        from config.models_config import _optional_arch

        self.assertIsNone(_optional_arch(None))
        self.assertIsNone(_optional_arch("null"))
        self.assertEqual(_optional_arch("Dense"), "dense")
        with self.assertRaises(ValueError):
            _optional_arch("transformer")

    def test_default_models_table_includes_random_and_benchmark(self) -> None:
        rows = default_models_table()
        benchmark = models_in_group("geom_benchmark")
        self.assertEqual(rows[0], ("Random", None))
        self.assertEqual(
            rows[1:],
            [(model.display, model.answers) for model in benchmark],
        )

    def test_floor_stem_keeps_params_suffix(self) -> None:
        sys.path.insert(0, str(_REPO / "viz_helper"))
        from plot_tau_vs_release_square import with_floor_stem

        path = Path("figures/release_scatter/r2scan3c/tau_vs_release_square_w95_params.png")
        self.assertEqual(
            with_floor_stem(path, 2.0).name,
            "tau_vs_release_square_w95_2_params.png",
        )
        path = Path("figures/release_scatter/r2scan3c/tau_vs_params_square_w95.png")
        self.assertEqual(
            with_floor_stem(path, 2.0).name,
            "tau_vs_params_square_w95_2.png",
        )
        path = Path("figures/release_scatter/r2scan3c/tau_vs_release_square_w95_os.png")
        self.assertEqual(
            with_floor_stem(path, 2.0).name,
            "tau_vs_release_square_w95_2_os.png",
        )
        path = Path(
            "figures/release_scatter/r2scan3c/tau_vs_release_square_w95_closed.png"
        )
        self.assertEqual(
            with_floor_stem(path, 2.0).name,
            "tau_vs_release_square_w95_2_closed.png",
        )

        from plot_tau_vs_release_square import _include_model

        self.assertTrue(
            _include_model("Gemma 3 27B", size_by_params=False, only="os")
        )
        self.assertFalse(
            _include_model("Claude Opus 5", size_by_params=False, only="os")
        )
        self.assertTrue(
            _include_model("Claude Opus 5", size_by_params=False, only="closed")
        )
        self.assertFalse(
            _include_model("Kimi K3", size_by_params=False, only="closed")
        )


if __name__ == "__main__":
    unittest.main()
