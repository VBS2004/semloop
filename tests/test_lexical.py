"""Tests for the lexical port, checked against the dataset's own labels.

These pin down the *documented* blind spots: each assertion cites the exact
deepeval behaviour it is reproducing (see semloop/lexical.py's docstring and
module comments), so a change to the port that silently fixes or breaks one of
these is caught immediately.
"""

import json
from pathlib import Path

import pytest

from semloop import Trace
from semloop.lexical import (
    MIN_WORDS_FOR_COMPARISON,
    clean_words,
    lexical_loop_score,
    score_reasoning_stagnation,
)

DATASET = Path(__file__).resolve().parents[1] / "bench" / "dataset.jsonl"


def _load(trace_id: str) -> Trace:
    with DATASET.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["trace_id"] == trace_id:
                return Trace.from_dict(row)
    raise KeyError(trace_id)


@pytest.mark.skipif(not DATASET.exists(), reason="run bench/make_dataset.py first")
class TestDocumentedBlindSpots:
    def test_paraphrased_loop_is_missed(self):
        """Same intent, different words: bigram Jaccard and SequenceMatcher both
        stay low, so nothing crosses the 0.85 similarity threshold."""
        trace = _load("paraphrased_loop-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["reasoning_stagnation"] == 1.0

    def test_terse_steps_are_skipped_outright(self):
        """Steps under MIN_WORDS_FOR_COMPARISON meaningful words are never
        compared at all — not "compared and found dissimilar", skipped."""
        trace = _load("terse_loop-00")
        for span in trace.llm_spans:
            assert len(clean_words(span.output_text)) < MIN_WORDS_FOR_COMPARISON
        result = lexical_loop_score(trace)
        assert result.breakdown["reasoning_stagnation"] == 1.0

    def test_alternating_loop_is_missed(self):
        """A-B-A-B: only consecutive pairs are compared, so a period-2 cycle
        is invisible by construction, whatever the similarity threshold is."""
        trace = _load("alternating_loop-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["reasoning_stagnation"] == 1.0

    def test_cosmetic_argument_change_resets_tool_repetition(self):
        """Tool repetition keys on the exact (name, sorted args) tuple, so
        changing one argument value resets the count to 1 for that tuple."""
        trace = _load("cosmetic_args_loop-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["tool_repetition"] == 1.0

    def test_healthy_paraphrased_progress_is_a_false_positive(self):
        """Five different files described in one sentence shape: the lexical
        check reads this as repetition even though the work is real."""
        trace = _load("template_progress-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["reasoning_stagnation"] < 1.0

    def test_identical_reasoning_is_still_caught(self):
        """What today's check is good at: verbatim repetition. A replacement
        must not regress this."""
        trace = _load("verbatim_loop-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["reasoning_stagnation"] == 0.0

    def test_identical_tool_calls_are_still_caught(self):
        trace = _load("identical_tool_loop-00")
        result = lexical_loop_score(trace)
        assert result.breakdown["tool_repetition"] == 0.0

    def test_single_severe_signal_cannot_flip_the_default_verdict(self):
        """At the default weights (0.40 / 0.35 / 0.25) and threshold 0.5, no
        single sub-score of 0.0 can push the composite below threshold alone:
        the best a lone signal can do is pull the total to 1 - its own weight.
        This is a property of deepeval's own weighting, not something a
        semantic swap changes — documented here so it isn't mistaken for a bug
        in this port."""
        trace = _load("identical_tool_loop-00")  # tool_repetition alone hits 0.0
        result = lexical_loop_score(trace)
        assert result.breakdown["tool_repetition"] == 0.0
        assert result.breakdown["reasoning_stagnation"] == 0.0  # this trace repeats text too
        # Isolate the claim directly, independent of the dataset:
        from semloop.lexical import combine_scores

        assert combine_scores(0.0, 1.0, 1.0) == pytest.approx(0.60)
        assert combine_scores(0.0, 1.0, 1.0) >= 0.5  # "success" despite a severe signal


def test_score_reasoning_stagnation_skips_short_pairs():
    from semloop.trace import Span

    spans = [Span(type="llm", name="a", output="short reply"), Span(type="llm", name="b", output="also short")]
    score, reason = score_reasoning_stagnation(spans)
    assert score == 1.0
    assert "Not enough" in reason or "No reasoning stagnation" in reason


def test_score_reasoning_stagnation_needs_two_spans():
    from semloop.trace import Span

    score, reason = score_reasoning_stagnation([Span(type="llm", name="a", output="x" * 100)])
    assert score == 1.0
    assert "Not enough" in reason
