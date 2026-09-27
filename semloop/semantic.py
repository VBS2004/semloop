"""Semantic loop detection: ask a judge whether each step made progress.

The lexical metric asks "do these two steps share words?". This asks "did the
second step get anywhere?" — which is the question that was meant all along.

Three families of question, all yes/no, and all sent in one request per trace:

* ``progress_{i}_{j}`` for consecutive steps — did step j advance the goal
  beyond step i?
* ``revisit_{i}_{j}`` for steps ``lookahead`` apart — is step j retrying what
  step i already tried? This catches A-B-A-B cycles, which a consecutive-pairs
  comparison cannot see by construction.
* ``tool_repeat_{name}`` for a tool called at least ``repetition_threshold``
  times — are the calls the same underlying action despite different
  arguments? This catches a retry loop that changes one cosmetic argument
  (a sort flag, a rephrased path) each time, which defeats the lexical
  check's exact ``(name, sorted args)`` match.

Call-graph cycle detection is left to the deterministic check in ``lexical`` —
it is exact, free, and already correct. Everything else that needs judgement —
reasoning stagnation and, now, tool-call identity — goes through the backend.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .backends import Answer, Backend, JudgeResponse
from .lexical import (
    DEFAULT_REPETITION_THRESHOLD,
    LoopResult,
    combine_scores,
    score_call_graph_cycles,
    score_reasoning_stagnation,
    score_tool_repetition,
)
from .trace import Span, Trace

#: A pair this stagnant scores 0.0 on the stagnation sub-signal.
SEVERE_STAGNATION = 0.85
#: A pair this stagnant scores 0.5.
MILD_STAGNATION = 0.60
#: Answers whose yes/no probability lands in this band are treated as undecided.
UNCERTAIN_BAND = (0.40, 0.60)

PROGRESS_INSTRUCTIONS = (
    "Compare step {j} with step {i} of this agent trace. Step {j} makes real "
    "progress toward the goal: it moves to a different subtask, acts on "
    "information it did not have at step {i}, or changes approach after a "
    "failure. Answer no if it restates the same plan, retries the same action, "
    "or rephrases step {i} without getting anywhere new."
)

REVISIT_INSTRUCTIONS = (
    "Step {j} of this agent trace goes back to an attempt already made at step "
    "{i}, without new information to justify it — the agent is cycling between "
    "the same states rather than advancing. Ignore differences in wording."
)

STAGNATION_LEVELS = [
    "Every step advances the task. No repetition beyond ordinary retries.",
    "Mostly advancing, with one step that repeats earlier work.",
    "Repeatedly circling: several steps retry the same attempts with no new information.",
    "Completely stuck: the agent repeats the same attempt and cannot get past it.",
]

TOOL_REPEAT_INSTRUCTIONS = (
    "Look at `state.tool_calls.{name}`: {count} calls to the tool '{name}', with "
    "their distinct argument sets listed in the order they occurred. Are these "
    "calls all the same underlying action — the agent retrying or repeating "
    "itself — even though the arguments differ cosmetically (an added flag, a "
    "reordered key, a trivially different value)? Answer no if the different "
    "arguments make this a genuinely different action each time, not a retry."
)


@dataclass(frozen=True)
class PairFinding:
    """One stagnant pair the judge found."""

    kind: str  # "progress" or "revisit"
    step_i: int
    step_j: int
    stagnation: float  # probability that the pair is stagnant
    uncertain: bool

    def describe(self) -> str:
        if self.kind == "progress":
            return (
                f"step {self.step_j} made no progress over step {self.step_i} "
                f"(p={self.stagnation:.2f})"
            )
        return (
            f"step {self.step_j} revisits step {self.step_i} "
            f"(p={self.stagnation:.2f})"
        )


@dataclass(frozen=True)
class ToolFinding:
    """A tool the judge found repeated under cosmetic argument changes."""

    name: str
    count: int
    same_action: float  # probability the calls are one repeated action

    def describe(self) -> str:
        return (
            f"tool '{self.name}' called {self.count} times as apparently the "
            f"same action despite different arguments (p={self.same_action:.2f})"
        )


class SemanticLoopDetection:
    """Loop detection whose stagnation signal is a judged decision.

    The interface mirrors deepeval's ``AgentLoopDetectionMetric``: construct it
    with a threshold, call ``measure``, read ``score``, ``reason``,
    ``score_breakdown`` and ``success``.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        threshold: float = 0.5,
        repetition_threshold: int = 3,
        lookahead: int = 2,
        max_questions: int = 64,
        max_chars_per_step: int = 800,
        check_tool_repetition: bool = True,
        check_semantic_tool_repetition: bool = True,
        check_reasoning_stagnation: bool = True,
        check_call_graph_cycles: bool = True,
        fallback_to_lexical_when_uncertain: bool = True,
        severe_stagnation: float = SEVERE_STAGNATION,
        mild_stagnation: float = MILD_STAGNATION,
        uncertain_band: Tuple[float, float] = UNCERTAIN_BAND,
    ):
        if lookahead < 1:
            raise ValueError("lookahead must be at least 1")
        self.backend = backend
        self.threshold = threshold
        self.repetition_threshold = repetition_threshold
        self.lookahead = lookahead
        self.max_questions = max_questions
        self.max_chars_per_step = max_chars_per_step
        self.check_tool_repetition = check_tool_repetition
        self.check_semantic_tool_repetition = check_semantic_tool_repetition
        self.check_reasoning_stagnation = check_reasoning_stagnation
        self.check_call_graph_cycles = check_call_graph_cycles
        self.fallback_to_lexical_when_uncertain = fallback_to_lexical_when_uncertain
        self.severe_stagnation = severe_stagnation
        self.mild_stagnation = mild_stagnation
        self.uncertain_band = uncertain_band

        self.score: float = 1.0
        self.reason: str = ""
        self.success: bool = True
        self.score_breakdown: Dict[str, float] = {}
        self.findings: List[PairFinding] = []
        self.tool_findings: List[ToolFinding] = []
        self.last_response: Optional[JudgeResponse] = None
        self._tool_call_counts: Dict[str, int] = {}

    @property
    def __name__(self) -> str:  # matches deepeval's metric convention
        return "Semantic Loop Detection"

    def measure(self, trace: Trace) -> LoopResult:
        cycle_score, cycle_reason = 1.0, "Call graph cycles check skipped."
        if self.check_call_graph_cycles:
            cycle_score, cycle_reason = score_call_graph_cycles(trace.root)

        lex_rep_score, lex_rep_reason = 1.0, "Tool repetition check skipped."
        if self.check_tool_repetition:
            lex_rep_score, lex_rep_reason = score_tool_repetition(
                trace.tool_spans, self.repetition_threshold
            )

        # Build every question about this trace up front, so it is answered in
        # exactly one request regardless of how many checks are enabled. Tool
        # questions are counted against the budget first — there are rarely
        # more than a handful of repeated tool names — leaving the remainder
        # for the pairwise reasoning questions, which can otherwise dominate a
        # long trace.
        tool_questions: Dict[str, Dict[str, Any]] = {}
        self._tool_call_counts = {}
        if self.check_semantic_tool_repetition:
            tool_questions, self._tool_call_counts = build_tool_questions(
                trace.tool_spans, repetition_threshold=self.repetition_threshold
            )

        steps = reasoning_steps(trace) if self.check_reasoning_stagnation else []
        stag_budget = max(1, self.max_questions - len(tool_questions))
        stag_questions = (
            build_questions(steps, lookahead=self.lookahead, max_questions=stag_budget)
            if self.check_reasoning_stagnation
            else {}
        )
        questions: Dict[str, Dict[str, Any]] = {**tool_questions, **stag_questions}

        response: Optional[JudgeResponse] = None
        stag_score, stag_reason, uncertain, self.findings = 1.0, "Reasoning stagnation check skipped.", False, []
        sem_rep_score, sem_rep_reason, self.tool_findings = 1.0, "", []

        if questions:
            state = build_state(
                trace,
                steps,
                max_chars_per_step=self.max_chars_per_step,
                tool_spans=trace.tool_spans if tool_questions else None,
            )
            response = self.backend.ask(state, questions)

            if any(q.startswith(("progress_", "revisit_")) for q in questions):
                stag_score, stag_reason, uncertain, self.findings = self._score_stagnation(
                    response.answers
                )
                if uncertain and self.fallback_to_lexical_when_uncertain:
                    lex_stag_score, lex_stag_reason = score_reasoning_stagnation(trace.llm_spans)
                    stag_score = min(stag_score, lex_stag_score)
                    stag_reason = (
                        f"{stag_reason} Judge was undecided, so the lexical check "
                        f"was used as well: {lex_stag_reason}"
                    )
            elif self.check_reasoning_stagnation:
                stag_reason = "Not enough reasoning steps to check for stagnation."

            if any(q.startswith("tool_repeat_") for q in questions):
                sem_rep_score, sem_rep_reason, self.tool_findings = self._score_tool_answers(
                    response.answers
                )
        elif self.check_reasoning_stagnation:
            stag_reason = "Not enough reasoning steps to check for stagnation."

        # The more severe of the exact-match and judged checks wins: a semantic
        # miss never hides a repetition the deterministic check already caught.
        if sem_rep_score < lex_rep_score:
            rep_score, rep_reason = sem_rep_score, sem_rep_reason
        else:
            rep_score, rep_reason = lex_rep_score, lex_rep_reason

        self.last_response = response
        self.score_breakdown = {
            "tool_repetition": rep_score,
            "reasoning_stagnation": stag_score,
            "call_graph_cycles": cycle_score,
        }
        self.score = combine_scores(
            rep_score,
            stag_score,
            cycle_score,
            check_tool_repetition=self.check_tool_repetition or self.check_semantic_tool_repetition,
            check_reasoning_stagnation=self.check_reasoning_stagnation,
            check_call_graph_cycles=self.check_call_graph_cycles,
        )
        self.success = self.score >= self.threshold

        reasons = [
            reason
            for sub_score, reason in (
                (rep_score, rep_reason),
                (stag_score, stag_reason),
                (cycle_score, cycle_reason),
            )
            if sub_score < 1.0 and reason
        ]
        self.reason = " ".join(reasons) if reasons else "No loop patterns detected."

        return LoopResult(
            score=self.score,
            reason=self.reason,
            breakdown=dict(self.score_breakdown),
            uncertain=uncertain,
            cost_usd=response.cost_usd if response else 0.0,
            input_tokens=response.input_tokens if response else 0,
            latency_ms=response.latency_ms if response else 0.0,
        )

    def _score_stagnation(
        self, answers: Mapping[str, Answer]
    ) -> Tuple[float, str, bool, List[PairFinding]]:
        low, high = self.uncertain_band
        findings: List[PairFinding] = []

        for qid, answer in answers.items():
            if answer.type != "noul":
                continue
            kind, i, j = _parse_question_id(qid)
            if kind is None:
                continue
            p_yes = float(answer.value)
            # "no progress" is the negation of the progress question; the revisit
            # question already asks about stagnation directly.
            stagnation = 1.0 - p_yes if kind == "progress" else p_yes
            findings.append(
                PairFinding(
                    kind=kind,
                    step_i=i,
                    step_j=j,
                    stagnation=stagnation,
                    uncertain=low <= stagnation <= high,
                )
            )

        if not findings:
            return 1.0, "Judge returned no usable pair answers.", True, findings

        worst = max(findings, key=lambda f: f.stagnation)
        stagnant = [f for f in findings if f.stagnation >= self.mild_stagnation]

        if worst.stagnation >= self.severe_stagnation:
            score = 0.0
        elif worst.stagnation >= self.mild_stagnation:
            score = 0.5
        else:
            score = 1.0

        # Only call the verdict undecided when nothing cleared the bar and the
        # closest call sits in the band — an answer inside the band next to a
        # confident detection changes nothing.
        uncertain = score == 1.0 and worst.uncertain

        if score == 1.0:
            reason = "No reasoning stagnation: every step advanced the task."
        else:
            detail = "; ".join(f.describe() for f in sorted(
                stagnant, key=lambda f: -f.stagnation
            )[:3])
            reason = f"Reasoning stagnation — {detail}."
        return score, reason, uncertain, findings

    def _score_tool_answers(
        self, answers: Mapping[str, Answer]
    ) -> Tuple[float, str, List[ToolFinding]]:
        findings: List[ToolFinding] = []
        for qid, answer in answers.items():
            if answer.type != "noul" or not qid.startswith("tool_repeat_"):
                continue
            name = qid[len("tool_repeat_"):]
            p_same = float(answer.value)
            count = self._tool_call_counts.get(name, 0)
            if p_same >= 0.5:
                findings.append(ToolFinding(name=name, count=count, same_action=p_same))

        if not findings:
            return 1.0, "", []

        worst = max(findings, key=lambda f: f.same_action)
        if worst.count >= self.repetition_threshold * 2:
            score = 0.0
        elif worst.count >= self.repetition_threshold:
            score = 0.5
        else:
            score = 1.0
        reason = f"Semantic tool repetition — {worst.describe()}." if score < 1.0 else ""
        return score, reason, findings


def reasoning_steps(trace: Trace) -> List[Span]:
    """The spans whose progress is worth judging.

    LLM spans are the reasoning steps. When a trace has none — some harnesses
    only record tool calls — fall back to tool spans so the check still applies.
    """
    llm_spans = trace.llm_spans
    if len(llm_spans) >= 2:
        return llm_spans
    tool_spans = trace.tool_spans
    return tool_spans if len(tool_spans) >= 2 else llm_spans


def build_questions(
    steps: List[Span], *, lookahead: int = 2, max_questions: int = 64
) -> Dict[str, Dict[str, Any]]:
    """One progress question per consecutive pair, plus revisit questions.

    Questions are emitted consecutive-pairs-first so that truncation at
    ``max_questions`` drops the longer-range checks rather than the core ones.
    """
    if len(steps) < 2:
        return {}

    questions: Dict[str, Dict[str, Any]] = {}
    for i in range(len(steps) - 1):
        j = i + 1
        questions[f"progress_{i}_{j}"] = {
            "type": "noul",
            "instructions": PROGRESS_INSTRUCTIONS.format(i=i, j=j),
            "criteria": {
                "true": f"Step {j} advances the task beyond step {i}.",
                "false": f"Step {j} repeats or rephrases step {i} without advancing.",
            },
        }

    for distance in range(2, lookahead + 1):
        for i in range(len(steps) - distance):
            j = i + distance
            questions[f"revisit_{i}_{j}"] = {
                "type": "noul",
                "instructions": REVISIT_INSTRUCTIONS.format(i=i, j=j),
                "criteria": {
                    "true": f"Step {j} is a repeat of the attempt at step {i}.",
                    "false": f"Step {j} is a different attempt from step {i}.",
                },
            }

    questions["stagnation"] = {
        "type": "score",
        "instructions": "Overall, how stuck is this agent across the whole trace?",
        "criteria": list(STAGNATION_LEVELS),
    }

    if len(questions) <= max_questions:
        return questions
    kept = dict(list(questions.items())[: max_questions - 1])
    kept["stagnation"] = questions["stagnation"]
    return kept


def build_state(
    trace: Trace,
    steps: List[Span],
    *,
    max_chars_per_step: int = 800,
    tool_spans: Optional[List[Span]] = None,
) -> Dict[str, Any]:
    """The trace as judged state: the goal, numbered steps, and optionally the
    distinct argument variants seen per repeated tool name.

    Step text is truncated because Jev's context is 32k tokens; a long run with
    full tool output would not fit. Truncation is marked so the judge can see
    that it happened.
    """
    state: Dict[str, Any] = {
        "goal": trace.goal,
        "steps": [
            {
                "index": index,
                "type": span.type,
                "name": span.name,
                "input": _truncate(span.input_text, max_chars_per_step),
                "output": _truncate(span.output_text, max_chars_per_step),
            }
            for index, span in enumerate(steps)
        ],
    }
    if tool_spans:
        state["tool_calls"] = {
            name: variants
            for name, variants in _group_tool_variants(tool_spans).items()
        }
    return state


def _group_tool_variants(tool_spans: List[Span]) -> Dict[str, List[str]]:
    """Distinct argument strings per tool name, in first-seen order."""
    by_name: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, set] = defaultdict(set)
    for span in tool_spans:
        text = span.input_text
        if text not in seen[span.name]:
            seen[span.name].add(text)
            by_name[span.name].append(text)
    return dict(by_name)


def build_tool_questions(
    tool_spans: List[Span],
    *,
    repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """One question per tool name called at least ``repetition_threshold``
    times, asking whether the calls are the same action under different
    arguments. Returns the questions plus each name's raw call count, since the
    count (not the distinct-argument count) is what decides severity.
    """
    counts: Dict[str, int] = defaultdict(int)
    for span in tool_spans:
        counts[span.name] += 1

    questions: Dict[str, Dict[str, Any]] = {}
    for name, count in counts.items():
        if count < repetition_threshold:
            continue
        instructions = TOOL_REPEAT_INSTRUCTIONS.format(name=name, count=count)
        questions[f"tool_repeat_{name}"] = {
            "type": "noul",
            "instructions": instructions,
            "criteria": {
                "true": "Same underlying action, cosmetic argument differences only.",
                "false": "Genuinely different actions despite sharing a tool name.",
            },
        }
    return questions, dict(counts)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} chars]"


def _parse_question_id(qid: str) -> Tuple[Optional[str], int, int]:
    parts = qid.split("_")
    if len(parts) != 3 or parts[0] not in ("progress", "revisit"):
        return None, -1, -1
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None, -1, -1
