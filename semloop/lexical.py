"""A faithful port of deepeval's ``AgentLoopDetectionMetric`` scoring.

This exists so the benchmark can compare against the current lexical
implementation without installing deepeval, and so the port can be diffed
against upstream. Stop words, thresholds and sub-score weights are copied
from ``deepeval/metrics/agent_loop_detection/agent_loop_detection.py``
(read 2026-09-27).

Three sub-signals:

* tool repetition — identical ``(name, sorted args)`` calls
* reasoning stagnation — max(bigram Jaccard, SequenceMatcher) on consecutive
  LLM outputs, skipping any pair with fewer than 20 meaningful words
* call graph cycles — DFS for a repeated ``type:name:input_hash`` label on one
  root-to-leaf path
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from .trace import Span, Trace

# Copied verbatim from deepeval.
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "i", "will", "now", "based",
    "on", "information", "provided", "to", "of", "in", "and", "that", "this",
    "with", "for", "it", "my", "next", "step", "going", "so", "do", "be",
    "have", "has", "not", "but", "as", "or", "from", "at", "by", "about",
    "above", "below", "up", "its", "let",
}

MIN_WORDS_FOR_COMPARISON = 20
DEFAULT_REPETITION_THRESHOLD = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.85

WEIGHT_TOOL_REPETITION = 0.40
WEIGHT_REASONING_STAGNATION = 0.35
WEIGHT_CALL_GRAPH_CYCLES = 0.25


@dataclass
class LoopResult:
    """The outcome of one loop check."""

    score: float
    reason: str
    breakdown: Dict[str, float] = field(default_factory=dict)
    uncertain: bool = False
    cost_usd: float = 0.0
    input_tokens: int = 0
    latency_ms: float = 0.0

    def flags_loop(self, threshold: float = 0.5) -> bool:
        """True when the score falls below the pass threshold.

        deepeval treats ``score >= threshold`` as success, so a loop is
        flagged strictly below it.
        """
        return self.score < threshold


def clean_words(text: str) -> List[str]:
    return [
        w for w in str(text).lower().split() if w not in STOP_WORDS and len(w) > 2
    ]


def bigram_jaccard(words_a: List[str], words_b: List[str]) -> float:
    if len(words_a) < 2 or len(words_b) < 2:
        return 0.0
    bg_a = set(zip(words_a, words_a[1:]))
    bg_b = set(zip(words_b, words_b[1:]))
    union = bg_a | bg_b
    if not union:
        return 0.0
    return len(bg_a & bg_b) / len(union)


def sequence_ratio(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b).ratio()


def score_tool_repetition(
    tool_spans: List[Span], repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD
) -> Tuple[float, str]:
    if not tool_spans:
        return 1.0, "No tool spans found."

    counts: Dict[Tuple[str, Tuple[Any, ...]], int] = {}
    for span in tool_spans:
        input_val = span.input
        if isinstance(input_val, str):
            try:
                input_val = json.loads(input_val)
            except Exception:
                pass
        if isinstance(input_val, dict):
            args_tuple: Tuple[Any, ...] = tuple(
                sorted((str(k), str(v)) for k, v in input_val.items())
            )
        else:
            args_tuple = (str(input_val),)
        key = (span.name, args_tuple)
        counts[key] = counts.get(key, 0) + 1

    (tool_name, _), count = max(counts.items(), key=lambda kv: kv[1])
    if count >= repetition_threshold * 2:
        return 0.0, f"Tool '{tool_name}' called {count} times with identical arguments."
    if count >= repetition_threshold:
        return 0.5, f"Tool '{tool_name}' called {count} times with identical arguments."
    return 1.0, "Tool calls are within acceptable repetition limits."


def score_reasoning_stagnation(
    llm_spans: List[Span], similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Tuple[float, str]:
    if len(llm_spans) < 2:
        return 1.0, "Not enough LLM spans to check for reasoning stagnation."

    max_overlap = 0.0
    pair = (-1, -1)
    for i in range(len(llm_spans) - 1):
        out1, out2 = llm_spans[i].output, llm_spans[i + 1].output
        if not isinstance(out1, str) or not isinstance(out2, str):
            continue
        words1, words2 = clean_words(out1), clean_words(out2)
        if (
            len(words1) < MIN_WORDS_FOR_COMPARISON
            or len(words2) < MIN_WORDS_FOR_COMPARISON
        ):
            continue
        similarity = max(
            bigram_jaccard(words1, words2),
            sequence_ratio(" ".join(words1), " ".join(words2)),
        )
        if similarity > max_overlap:
            max_overlap, pair = similarity, (i, i + 1)

    if max_overlap >= similarity_threshold:
        if max_overlap > 0.95:
            return 0.0, f"Identical reasoning outputs at steps {pair[0]} and {pair[1]}."
        return (
            0.5,
            f"High reasoning overlap ({max_overlap:.2f}) at steps {pair[0]} and {pair[1]}.",
        )
    return 1.0, "No reasoning stagnation."


def score_call_graph_cycles(root: Span) -> Tuple[float, str]:
    cycle_path: List[str] = []

    def label(span: Span) -> str:
        raw = span.input
        if isinstance(raw, dict):
            try:
                raw = json.dumps(raw, sort_keys=True)
            except (TypeError, ValueError):
                raw = str(raw)
        return f"{span.type}:{span.name}:{str(raw)[:64]}"

    def dfs(span: Span, ancestors: List[str]) -> bool:
        current = label(span)
        if current in ancestors:
            cycle_path.extend(ancestors[ancestors.index(current):])
            cycle_path.append(current)
            return True
        ancestors.append(current)
        for child in span.children:
            if dfs(child, ancestors):
                return True
        ancestors.pop()
        return False

    if dfs(root, []):
        display = " -> ".join(":".join(p.split(":", 2)[:2]) for p in cycle_path)
        return 0.0, f"Cycle detected in execution path: {display}."
    return 1.0, "No execution cycles detected."


def combine_scores(
    rep_score: float,
    stag_score: float,
    cycle_score: float,
    *,
    check_tool_repetition: bool = True,
    check_reasoning_stagnation: bool = True,
    check_call_graph_cycles: bool = True,
) -> float:
    weights = 0.0
    total = 0.0
    if check_tool_repetition:
        weights += WEIGHT_TOOL_REPETITION
        total += rep_score * WEIGHT_TOOL_REPETITION
    if check_reasoning_stagnation:
        weights += WEIGHT_REASONING_STAGNATION
        total += stag_score * WEIGHT_REASONING_STAGNATION
    if check_call_graph_cycles:
        weights += WEIGHT_CALL_GRAPH_CYCLES
        total += cycle_score * WEIGHT_CALL_GRAPH_CYCLES
    if weights == 0.0:
        return 1.0
    return total / weights


def lexical_loop_score(
    trace: Trace,
    *,
    repetition_threshold: int = DEFAULT_REPETITION_THRESHOLD,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> LoopResult:
    """Score a trace exactly the way deepeval's metric does today."""
    rep_score, rep_reason = score_tool_repetition(
        trace.tool_spans, repetition_threshold
    )
    stag_score, stag_reason = score_reasoning_stagnation(
        trace.llm_spans, similarity_threshold
    )
    cycle_score, cycle_reason = score_call_graph_cycles(trace.root)

    score = combine_scores(rep_score, stag_score, cycle_score)
    reasons = [
        reason
        for sub_score, reason in (
            (rep_score, rep_reason),
            (stag_score, stag_reason),
            (cycle_score, cycle_reason),
        )
        if sub_score < 1.0
    ]
    return LoopResult(
        score=score,
        reason=" ".join(reasons) if reasons else "No loop patterns detected.",
        breakdown={
            "tool_repetition": rep_score,
            "reasoning_stagnation": stag_score,
            "call_graph_cycles": cycle_score,
        },
    )
