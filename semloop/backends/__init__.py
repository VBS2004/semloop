"""Judge backends: something that answers typed questions about a state.

A backend takes one state (any JSON value) plus a mapping of question id to a
typed question, and returns one ``Answer`` per question id. Everything else in
this package is backend-agnostic, so the same metric can run on Jev, on a
chat model, or on a scripted fake in tests.

Question dicts follow the TypeSafe System One wire format:

* ``{"type": "noul", "instructions": str, "criteria": {"true": str, "false": str}}``
* ``{"type": "choice", "instructions": str, "criteria": {label: description}}``
* ``{"type": "score", "instructions": str, "criteria": [level0, level1, ...]}``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol


@dataclass(frozen=True)
class Answer:
    """One answer, normalised across backends.

    ``probability`` is the probability of "yes" for a noul, the probability of
    the winning label for a choice, and the probability mass of the winning
    level for a score. ``value`` is the yes/no probability, the chosen label or
    the numeric score respectively.
    """

    type: str
    value: Any
    probability: float
    probabilities: Dict[Any, float] = field(default_factory=dict)
    confidence: Optional[float] = None

    @property
    def is_yes(self) -> bool:
        if self.type != "noul":
            raise TypeError(f"is_yes is only defined for noul answers, got {self.type!r}")
        return float(self.value) >= 0.5

    def is_uncertain(self, low: float = 0.4, high: float = 0.6) -> bool:
        """True when the answer sits in the undecided band.

        For a noul that is a yes/no probability near the middle. For a choice or
        score it is a winning probability below ``high``: with more than two
        options, "undecided" means no option won clearly, so the low bound
        does not apply.
        """
        if self.type == "noul":
            return low <= float(self.value) <= high
        return self.probability < high


@dataclass(frozen=True)
class JudgeResponse:
    """A backend's answers plus what the call cost."""

    answers: Dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""


class Backend(Protocol):
    """Anything that can answer typed questions about a state."""

    name: str

    def ask(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> JudgeResponse: ...


class MockBackend:
    """A scripted backend for tests and offline runs.

    ``answers`` maps a question id to a value: a float for a noul, a label for
    a choice, an int for a score. ``default`` answers anything not listed, so a
    test only has to script the ids it cares about.
    """

    name = "mock"

    def __init__(
        self,
        answers: Optional[Mapping[str, Any]] = None,
        *,
        default: Any = 0.0,
        input_tokens: int = 0,
    ):
        self._answers = dict(answers or {})
        self._default = default
        self._input_tokens = input_tokens
        self.calls: list[Dict[str, Any]] = []

    def ask(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> JudgeResponse:
        self.calls.append({"state": state, "questions": dict(questions)})
        answers: Dict[str, Answer] = {}
        for qid, question in questions.items():
            raw = self._answers.get(qid, self._default)
            answers[qid] = _mock_answer(str(question.get("type", "noul")), raw, question)
        return JudgeResponse(
            answers=answers, input_tokens=self._input_tokens, model="mock"
        )


def _mock_answer(qtype: str, raw: Any, question: Mapping[str, Any]) -> Answer:
    if qtype == "noul":
        p = float(raw)
        return Answer(type="noul", value=p, probability=max(p, 1.0 - p), probabilities={True: p, False: 1.0 - p})
    if qtype == "choice":
        labels = list(question.get("criteria", {}) or {})
        label = raw if isinstance(raw, str) else (labels[0] if labels else "")
        probs = {l: (0.9 if l == label else 0.1 / max(len(labels) - 1, 1)) for l in labels}
        return Answer(
            type="choice",
            value=label,
            probability=probs.get(label, 1.0),
            probabilities=probs,
            confidence=0.9,
        )
    levels = list(question.get("criteria", []) or [])
    level = int(raw) if isinstance(raw, (int, float)) else 0
    probs = {i: (0.9 if i == level else 0.1 / max(len(levels) - 1, 1)) for i in range(len(levels))}
    return Answer(
        type="score",
        value=float(level),
        probability=probs.get(level, 1.0),
        probabilities=probs,
        confidence=0.9,
    )


from .jev import JevBackend  # noqa: E402  (re-exported for convenience)
from .llm import LLMJudgeBackend  # noqa: E402

__all__ = [
    "Answer",
    "Backend",
    "JevBackend",
    "JudgeResponse",
    "LLMJudgeBackend",
    "MockBackend",
]
