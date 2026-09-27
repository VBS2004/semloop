"""Tests for SemanticLoopDetection against a scripted MockBackend.

No network, no API key: every judge answer is scripted, so these tests check
the *wiring* (question building, answer parsing, score combination, single
request per trace) rather than whether a live Jev or LLM judge is actually
smart. `bench/run.py --judge jev` is what measures that.
"""

import json
from pathlib import Path

import pytest

from semloop import MockBackend, SemanticLoopDetection, Trace
from semloop.semantic import build_questions, build_state, build_tool_questions, reasoning_steps
from semloop.trace import Span

DATASET = Path(__file__).resolve().parents[1] / "bench" / "dataset.jsonl"


def _load(trace_id: str) -> Trace:
    with DATASET.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["trace_id"] == trace_id:
                return Trace.from_dict(row)
    raise KeyError(trace_id)


def _healthy_backend_for(trace: Trace) -> MockBackend:
    """Script every progress/revisit question as 'yes, healthy progress',
    every tool-repeat question as 'no, genuinely different actions' — the
    honest verdict for a clean run, used to prove no signal fires spuriously.
    """
    steps = reasoning_steps(trace)
    stag_qs = build_questions(steps, lookahead=2, max_questions=64)
    tool_qs, _ = build_tool_questions(trace.tool_spans)
    answers = {}
    for qid in stag_qs:
        if qid.startswith("progress_"):
            answers[qid] = 0.95  # yes, advanced
        elif qid.startswith("revisit_"):
            answers[qid] = 0.05  # no, not a revisit
    for qid in tool_qs:
        answers[qid] = 0.05  # no, different actions
    return MockBackend(answers=answers, default=0.5)


@pytest.mark.skipif(not DATASET.exists(), reason="run bench/make_dataset.py first")
class TestSemanticDetectionFixesTheBlindSpots:
    def test_catches_paraphrased_loop(self):
        """The one blind spot this whole package targets: same intent, reworded."""
        trace = _load("paraphrased_loop-00")
        steps = reasoning_steps(trace)
        qs = build_questions(steps, lookahead=2, max_questions=64)
        # Every consecutive pair here is the same intent restated — script the
        # judge accordingly, exactly as a competent judge should answer.
        answers = {qid: 0.03 for qid in qs if qid.startswith("progress_")}
        backend = MockBackend(answers=answers, default=0.03)
        metric = SemanticLoopDetection(backend, check_semantic_tool_repetition=False)
        result = metric.measure(trace)
        assert result.breakdown["reasoning_stagnation"] < 1.0
        assert result.flags_loop(0.5)

    def test_catches_alternating_loop_via_revisit_questions(self):
        """A-B-A-B: consecutive pairs look fine, but the revisit_i_j question
        (i, i+2) directly asks whether step j repeats step i."""
        trace = _load("alternating_loop-00")
        steps = reasoning_steps(trace)
        qs = build_questions(steps, lookahead=2, max_questions=64)
        assert any(qid.startswith("revisit_") for qid in qs), "no revisit questions built"
        answers = {qid: 0.95 for qid in qs if qid.startswith("progress_")}  # consecutive pairs look fine
        answers.update({qid: 0.9 for qid in qs if qid.startswith("revisit_")})  # but it's a revisit
        backend = MockBackend(answers=answers, default=0.9)
        metric = SemanticLoopDetection(backend, check_semantic_tool_repetition=False)
        result = metric.measure(trace)
        assert result.breakdown["reasoning_stagnation"] < 1.0

    def test_does_not_skip_terse_steps(self):
        """Unlike the lexical port, nothing here filters on word count."""
        trace = _load("terse_loop-00")
        steps = reasoning_steps(trace)
        qs = build_questions(steps, lookahead=2, max_questions=64)
        assert any(qid.startswith("progress_") for qid in qs)

    def test_catches_cosmetic_argument_tool_loop(self):
        """The gap the lexical exact-match check can't close: same tool,
        different arguments, same underlying retry.

        The sub-score goes to 0.0 — a full, confident detection. Whether that
        alone flips the *combined* verdict depends on the threshold, not on
        whether the signal fired: see the note below and
        ``test_a_lone_signal_needs_a_stricter_threshold_to_flip_the_verdict``.
        """
        trace = _load("cosmetic_args_loop-00")
        tool_qs, counts = build_tool_questions(trace.tool_spans)
        assert "tool_repeat_grep" in tool_qs
        assert counts["grep"] == 6
        # Isolate the tool signal: script every reasoning question as healthy
        # progress so only the tool check can flag this trace.
        steps = reasoning_steps(trace)
        stag_qs = build_questions(steps, lookahead=2, max_questions=64)
        healthy = {qid: 0.95 for qid in stag_qs if qid.startswith("progress_")}
        healthy.update({qid: 0.05 for qid in stag_qs if qid.startswith("revisit_")})
        backend = MockBackend(answers={**healthy, "tool_repeat_grep": 0.9}, default=0.5)
        metric = SemanticLoopDetection(backend)
        result = metric.measure(trace)
        assert result.breakdown["tool_repetition"] == 0.0
        assert result.breakdown["reasoning_stagnation"] == 1.0  # confirms isolation worked

        # At deepeval's own default weights (tool repetition is 0.40 of the
        # composite) and default threshold (0.5), one fully-confident signal
        # is not enough on its own — see test_lexical.py's
        # test_single_severe_signal_cannot_flip_the_default_verdict for the
        # general property. This is not something the semantic swap changes;
        # it is documented here so the "loop" label on this trace is not read
        # as something semloop's default settings will flag out of the box.
        assert result.score == pytest.approx(0.6)
        assert not result.flags_loop(0.5)
        # A caller who wants any single confirmed signal to flip the verdict
        # sets the threshold above 1 - that signal's weight (here, > 0.60).
        assert result.flags_loop(0.65)

    def test_healthy_paraphrased_progress_is_not_a_false_positive(self):
        """Five different files, same sentence shape: a judge that actually
        reads the content sees real progress, unlike the lexical check."""
        trace = _load("template_progress-00")
        backend = _healthy_backend_for(trace)
        metric = SemanticLoopDetection(backend)
        result = metric.measure(trace)
        assert result.breakdown["reasoning_stagnation"] == 1.0
        assert not result.flags_loop(0.5)

    def test_healthy_boilerplate_progress_is_not_a_false_positive(self):
        trace = _load("boilerplate_progress-00")
        backend = _healthy_backend_for(trace)
        metric = SemanticLoopDetection(backend)
        result = metric.measure(trace)
        assert result.breakdown["reasoning_stagnation"] == 1.0

    def test_healthy_progress_trace_is_untouched(self):
        trace = _load("progress-00")
        backend = _healthy_backend_for(trace)
        metric = SemanticLoopDetection(backend)
        result = metric.measure(trace)
        assert result.score == 1.0
        assert not result.flags_loop(0.5)


def test_one_request_per_trace_regardless_of_pair_count():
    """The whole cost story rests on this: one call answers every question
    about a trace, however many pairs and tool names it has."""
    steps = [Span(type="llm", name="step", output="word " * 30) for _ in range(8)]
    trace = Trace.from_steps(goal="g", steps=[{"type": "llm", "output": s.output} for s in steps])
    backend = MockBackend(default=0.5)
    metric = SemanticLoopDetection(backend)
    metric.measure(trace)
    assert len(backend.calls) == 1


def test_undecided_answers_fall_back_to_lexical():
    """A judge answer sitting in the undecided band (0.4-0.6) triggers the
    lexical stagnation check as a second opinion, rather than silently
    resolving to 'no loop'."""
    trace = _load("verbatim_loop-00") if DATASET.exists() else None
    if trace is None:
        pytest.skip("run bench/make_dataset.py first")
    steps = reasoning_steps(trace)
    qs = build_questions(steps, lookahead=2, max_questions=64)
    # Every progress question answered exactly at the undecided midpoint.
    answers = {qid: 0.5 for qid in qs if qid.startswith("progress_")}
    backend = MockBackend(answers=answers, default=0.5)
    metric = SemanticLoopDetection(backend, check_semantic_tool_repetition=False)
    result = metric.measure(trace)
    assert result.uncertain
    # verbatim_loop-00 repeats the identical text, so the lexical fallback
    # should still catch it even though the judge itself was undecided.
    assert result.breakdown["reasoning_stagnation"] == 0.0


def test_no_reasoning_steps_short_circuits_without_a_call():
    trace = Trace.from_steps(goal="g", steps=[])
    backend = MockBackend(default=0.5)
    metric = SemanticLoopDetection(backend)
    result = metric.measure(trace)
    assert result.score == 1.0
    assert len(backend.calls) == 0
