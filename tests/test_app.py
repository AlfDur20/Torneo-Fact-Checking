"""
tests/test_app.py — Unit tests for the FastAPI inference server.

These tests use a lightweight stub model so they run without the full
fine-tuned checkpoint (useful in CI where the model artefact is absent).
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers for patching app internals
# ---------------------------------------------------------------------------
ID2LABEL = {0: "REFUTED", 1: "SUPPORTED"}


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        # Return dict-like object whose values support .to(device)
        class _T:
            def to(self, device):
                return self

        return {"input_ids": _T(), "attention_mask": _T()}


class FakeModel:
    def __init__(self, return_label_id=1):
        self._label_id = return_label_id
        self.config = types.SimpleNamespace(id2label={0: "REFUTED", 1: "SUPPORTED"})

    def __call__(self, **kwargs):
        class _Logits:
            pass

        class _Out:
            pass

        out = _Out()
        out.logits = _Logits()
        return out

    def eval(self):
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Base test class that sets up the fake infrastructure
# ---------------------------------------------------------------------------
class _Base(unittest.TestCase):
    _label_id = 1  # override in subclasses

    def setUp(self):
        # Remove cached app module between tests
        for mod in list(sys.modules.keys()):
            if mod in ("app",):
                del sys.modules[mod]

        fake_tensor = MagicMock()
        fake_tensor.item.return_value = self._label_id

        self._patches = [
            patch("app._load_model", return_value=(
                FakeTokenizer(),
                FakeModel(self._label_id),
                "cpu",
                ID2LABEL,
            )),
            patch("torch.no_grad", return_value=MagicMock(
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
            )),
            patch("torch.argmax", return_value=fake_tensor),
        ]
        for p in self._patches:
            p.start()

        from fastapi.testclient import TestClient
        from app import app

        # Use context-manager form so the lifespan (startup/shutdown) runs
        self._ctx = TestClient(app)
        self._ctx.__enter__()
        self.client = self._ctx

    def tearDown(self):
        self._ctx.__exit__(None, None, None)
        for p in self._patches:
            p.stop()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestHealthEndpoint(_Base):
    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


class TestPredictSupported(_Base):
    _label_id = 1

    def test_returns_valid_label(self):
        resp = self.client.post("/predict", json={"claim": "Hà Nội là thủ đô của Việt Nam."})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predicted_label", data)
        self.assertIn(data["predicted_label"], ("SUPPORTED", "REFUTED"))


class TestPredictRefuted(_Base):
    _label_id = 0

    def test_with_evidence(self):
        resp = self.client.post("/predict", json={
            "claim": "Hà Nội không phải là thủ đô.",
            "evidence": "Hà Nội là thủ đô của Việt Nam.",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predicted_label", data)
        self.assertIn(data["predicted_label"], ("SUPPORTED", "REFUTED"))


class TestPredictValidation(_Base):
    def test_empty_claim_raises_422(self):
        resp = self.client.post("/predict", json={"claim": "  "})
        self.assertEqual(resp.status_code, 422)

    def test_missing_claim_raises_422(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 422)

    def test_output_json_format(self):
        resp = self.client.post("/predict", json={"claim": "test claim"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predicted_label", data)
        self.assertIn(data["predicted_label"], ("SUPPORTED", "REFUTED"))


class TestLabelNormalisation(unittest.TestCase):
    """Tests for label normalisation logic in train.py."""

    def setUp(self):
        import importlib.util
        # Stub kagglehub so import doesn't fail when training deps are absent
        sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))
        spec = importlib.util.spec_from_file_location(
            "train_module",
            os.path.join(os.path.dirname(__file__), "..", "train.py"),
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_supported_variants(self):
        fn = self.mod._normalise_label
        for raw in ("SUPPORTED", "Support", "true", "TRUE", "1", "CORRECT"):
            with self.subTest(raw=raw):
                self.assertEqual(fn(raw), "SUPPORTED")

    def test_refuted_variants(self):
        fn = self.mod._normalise_label
        for raw in ("REFUTED", "Refute", "false", "FALSE", "0", "INCORRECT"):
            with self.subTest(raw=raw):
                self.assertEqual(fn(raw), "REFUTED")

    def test_unknown_returns_none(self):
        fn = self.mod._normalise_label
        self.assertIsNone(fn("MAYBE"))
        self.assertIsNone(fn("NEI"))


if __name__ == "__main__":
    unittest.main()
