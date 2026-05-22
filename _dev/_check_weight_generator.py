"""Sanity check for llm_weight_generator.py (no Anthropic API needed)."""
import json
import sys
from pathlib import Path

DSC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DSC_ROOT))

from dsc_framework.llm_weight_generator import (  # noqa: E402
    DEFAULT_WEIGHT_PROFILES,
    SUM_TOLERANCE,
    WEIGHT_BOUNDS,
    WeightGenerationResult,
    WeightGenerator,
)

print("=== import OK ===")
print("cells:", list(DEFAULT_WEIGHT_PROFILES.keys()))
for key, w in DEFAULT_WEIGHT_PROFILES.items():
    print(f"  {key}: {len(w)} metrics, sum={sum(w.values()):.4f}")

print("\n=== fallback test (LLM call fails) ===")


class FailingGen(WeightGenerator):
    def _call_llm(self, user_message):
        raise RuntimeError("intentional test failure")


g = FailingGen("tabular", "classification")
result = g.generate({"n_rows": 100, "schema": {"a": "int"}})
print("used_fallback:", result.used_fallback)
print("reason:", result.fallback_reason)
print("weights sum:", round(sum(result.weights.values()), 4))
assert result.used_fallback
assert "llm_call_failed" in result.fallback_reason

print("\n=== valid response test ===")


class FakeOKGen(WeightGenerator):
    def _call_llm(self, user_message):
        return json.dumps({"weights": dict(DEFAULT_WEIGHT_PROFILES[self.cell_key])})


g2 = FakeOKGen("tabular", "classification")
r2 = g2.generate({"n_rows": 100})
print("used_fallback:", r2.used_fallback)
print("weights sum:", round(sum(r2.weights.values()), 4))
assert not r2.used_fallback, f"expected pass, got: {r2.fallback_reason}"

print("\n=== invalid sum response test (sum != 1) ===")


class InvalidSumGen(WeightGenerator):
    def _call_llm(self, user_message):
        return json.dumps({"weights": {k: 0.5 for k in self.metric_names}})


g3 = InvalidSumGen("tabular", "classification")
r3 = g3.generate({"n_rows": 100})
print("used_fallback:", r3.used_fallback)
print("reason:", r3.fallback_reason)
assert r3.used_fallback
assert "validation_failed" in r3.fallback_reason

print("\n=== invalid bound response test (w > 0.60) ===")


class OutOfBoundsGen(WeightGenerator):
    def _call_llm(self, user_message):
        names = self.metric_names
        # one metric gets 0.95, others split remainder
        w = {names[0]: 0.95}
        rest = (1.0 - 0.95) / (len(names) - 1)
        for n in names[1:]:
            w[n] = rest
        return json.dumps({"weights": w})


g4 = OutOfBoundsGen("tabular", "classification")
r4 = g4.generate({"n_rows": 100})
print("used_fallback:", r4.used_fallback)
print("reason:", r4.fallback_reason)
assert r4.used_fallback
assert "outside" in r4.fallback_reason

print("\n=== markdown code block parse test ===")


class MarkdownGen(WeightGenerator):
    def _call_llm(self, user_message):
        weights = dict(DEFAULT_WEIGHT_PROFILES[self.cell_key])
        body = json.dumps({"weights": weights})
        return f"Here is the answer:\n```json\n{body}\n```\nDone."


g5 = MarkdownGen("tabular", "regression")
r5 = g5.generate({"n_rows": 100})
print("used_fallback:", r5.used_fallback)
print("reason:", r5.fallback_reason)
assert not r5.used_fallback, f"markdown parse failed: {r5.fallback_reason}"

print("\n=== all checks passed ===")
