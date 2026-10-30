#!/usr/bin/env python3

"""Tests for authoritative three-seed visualization cohort discovery."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import ModelSpec
from viz.cohort import (
    complete_answer_files,
    complete_geom_models,
    warn_unregistered_complete_models,
)


def _spec(model_id: str, filename: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        display=model_id.title(),
        short=model_id,
        answers=filename,
        color="#123456",
        model_class="os",
    )


class TestVizCohort(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.seeds = [root / seed for seed in ("42", "43", "44")]
        metadata = {
            "molecule": "mol",
            "letter_to_deltaE_kcal_mol": {"1": 0.0, "2": 1.0},
            "public_label_to_conf_index": {"1": 0, "2": 1},
        }
        for seed in self.seeds:
            seed.mkdir()
            (seed / "prompt_metadata.jsonl").write_text(
                json.dumps(metadata) + "\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_answers(self, filename: str, ranking: list[str], *, seeds=None) -> None:
        row = json.dumps({"molecule": "mol", "ranking": ranking}) + "\n"
        for seed in seeds or self.seeds:
            (seed / filename).write_text(row, encoding="utf-8")

    def test_complete_registered_model_is_returned(self) -> None:
        filename = "answers_registered.jsonl"
        self._write_answers(filename, ["1", "2"])
        spec = _spec("registered", filename)
        with (
            patch("viz.cohort.models_in_group", return_value=[spec]),
            patch("viz.cohort.models_by_id", return_value={spec.id: spec}),
        ):
            result = complete_geom_models(seed_dirs=self.seeds)
        self.assertEqual(result, [(spec.display, filename)])

    def test_incomplete_registered_model_is_excluded(self) -> None:
        filename = "answers_incomplete.jsonl"
        self._write_answers(filename, ["1", "2"], seeds=self.seeds[:2])
        spec = _spec("incomplete", filename)
        with (
            patch("viz.cohort.models_in_group", return_value=[spec]),
            patch("viz.cohort.models_by_id", return_value={spec.id: spec}),
        ):
            result = complete_geom_models(seed_dirs=self.seeds)
        self.assertEqual(result, [])

    def test_complete_unregistered_filename_warns(self) -> None:
        filename = "answers_newmodel.jsonl"
        self._write_answers(filename, ["1", "2"])
        complete = complete_answer_files(seed_dirs=self.seeds)
        stderr = io.StringIO()
        with (
            patch("viz.cohort.models_by_id", return_value={}),
            patch("viz.cohort.models_in_group", return_value=[]),
            redirect_stderr(stderr),
        ):
            warn_unregistered_complete_models(complete)
        self.assertIn(filename, stderr.getvalue())
        self.assertIn("color", stderr.getvalue())

    def test_registered_model_outside_group_warns(self) -> None:
        filename = "answers_outside.jsonl"
        spec = _spec("outside", filename)
        stderr = io.StringIO()
        with (
            patch("viz.cohort.models_by_id", return_value={spec.id: spec}),
            patch("viz.cohort.models_in_group", return_value=[]),
            redirect_stderr(stderr),
        ):
            warn_unregistered_complete_models({filename})
        self.assertIn("omitted from groups.geom_benchmark", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
