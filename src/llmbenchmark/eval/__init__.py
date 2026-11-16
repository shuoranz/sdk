"""DeepEval integration. Requires: pip install "llmbenchmark[eval]"."""

try:
    import deepeval  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'llmbenchmark.eval needs DeepEval. Install it with: pip install "llmbenchmark[eval]"'
    ) from exc

from .judges import DEFAULT_JUDGE_MODEL, ClaudeJudge, JudgeIncomplete, JudgeRefusal, judge
from .metrics import ExactMatch, ExpectedTools, correctness
from .adapters import spans_to_trace_dict, to_conversational_test_case, to_test_case
from .runner import EvalReport, evaluate_run

__all__ = [
    "evaluate_run",
    "EvalReport",
    "judge",
    "ClaudeJudge",
    "JudgeRefusal",
    "JudgeIncomplete",
    "DEFAULT_JUDGE_MODEL",
    "ExactMatch",
    "ExpectedTools",
    "correctness",
    "to_test_case",
    "to_conversational_test_case",
    "spans_to_trace_dict",
]
