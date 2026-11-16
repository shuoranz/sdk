"""Ready-made metrics.

``ExactMatch`` shows what a DeepEval metric is underneath: a class with
``measure()`` that sets ``score``, ``success`` and ``reason``. It needs no
judge model, so it's free, instant, and deterministic.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase

try:  # DeepEval 4.x name
    from deepeval.test_case import SingleTurnParams as _Params
except ImportError:  # older DeepEval
    from deepeval.test_case import LLMTestCaseParams as _Params


class ExactMatch(BaseMetric):
    """1.0 if actual_output equals expected_output (whitespace/case-insensitive)."""

    # Lets evaluate_run check records have these fields before scoring.
    _required_params = [_Params.ACTUAL_OUTPUT, _Params.EXPECTED_OUTPUT]

    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False) -> None:
        super().__init__()  # keep any per-instance setup newer DeepEval versions add
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.evaluation_model = None
        self.include_reason = True
        self.async_mode = True
        self.strict_mode = False

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        actual = (test_case.actual_output or "").strip()
        expected = (test_case.expected_output or "").strip()
        if not self.case_sensitive:
            actual, expected = actual.lower(), expected.lower()
        self.score = 1.0 if actual == expected else 0.0
        self.success = self.score >= self.threshold
        self.reason = "exact match" if self.success else f"expected {expected!r}, got {actual!r}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self) -> str:
        return "Exact Match"


class ExpectedTools(BaseMetric):
    """Fraction of ``expected_tools`` the agent actually called (order-insensitive).

    Free and deterministic, like ExactMatch: a quick agent check that needs
    no judge. For order and argument checks use DeepEval's ToolCorrectnessMetric.
    """

    _required_params = [_Params.TOOLS_CALLED, _Params.EXPECTED_TOOLS]

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.evaluation_model = None
        self.include_reason = True
        self.async_mode = True
        self.strict_mode = False

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        called = {t.name for t in test_case.tools_called or []}
        expected = {t.name for t in test_case.expected_tools or []}
        self.score = len(expected & called) / len(expected) if expected else 1.0
        self.success = self.score >= self.threshold
        missing = sorted(expected - called)
        self.reason = "all expected tools called" if not missing else f"never called: {', '.join(missing)}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self) -> str:
        return "Expected Tools"


def correctness(model: Any = None, threshold: float = 0.7) -> GEval:
    """A G-Eval metric: is the answer consistent with the reference answer?"""
    return GEval(
        name="Correctness",
        criteria=(
            "Determine whether the actual output is factually consistent with the "
            "expected output. Extra correct detail is fine; any contradiction or "
            "missing key fact from the expected output is not."
        ),
        evaluation_params=[
            _Params.INPUT,
            _Params.ACTUAL_OUTPUT,
            _Params.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )
