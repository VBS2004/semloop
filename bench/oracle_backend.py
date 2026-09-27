"""A scripted stand-in for a competent judge, keyed by the dataset's own labels.

This is **not** a benchmark of Jev, an LLM, or anything else real — it answers
using knowledge of which family generated the trace, which no real backend
has. Its only purpose is to confirm that ``SemanticLoopDetection``'s wiring
(question building, answer parsing, score combination) reaches the correct
verdict when its questions are answered the way a judge that actually read the
content should answer them. `bench/run.py --judge jev` or `--judge llm` is
what measures real backend accuracy; this measures only "does the mechanism
work", which is why its results are reported in a clearly separate column
labelled ``oracle`` and never compared against cost or latency.

Every one of the ten families gets one row: how a correct judge should answer
a consecutive-progress question, a revisit question, and a tool-repeat
question. The comment on each row names the trace property it reflects, so a
change to the family generators in ``make_dataset.py`` can be checked against
it directly.
"""

from __future__ import annotations

from typing import Dict, Mapping

from semloop import Answer, JudgeResponse, Trace
from semloop.semantic import build_questions, build_tool_questions, reasoning_steps

# progress: P(step j advances beyond step i) — high for real progress, low for restated intent.
# revisit: P(step j repeats step i) — high only for the alternating A-B-A-B family.
# tool: P(repeated tool calls are the same action despite different args) — high only when the
#       trace's tool calls are genuinely the same retried action, whether or not the arguments differ.
FAMILY_ORACLE: Dict[str, Dict[str, float]] = {
    "paraphrased_loop":     {"progress": 0.04, "revisit": 0.50, "tool": 0.90},
    "terse_loop":           {"progress": 0.04, "revisit": 0.50, "tool": 0.90},
    "alternating_loop":     {"progress": 0.85, "revisit": 0.90, "tool": 0.50},
    "cosmetic_args_loop":   {"progress": 0.90, "revisit": 0.05, "tool": 0.90},
    "verbatim_loop":        {"progress": 0.02, "revisit": 0.50, "tool": 0.90},
    "identical_tool_loop":  {"progress": 0.02, "revisit": 0.50, "tool": 0.95},
    "progress":             {"progress": 0.95, "revisit": 0.03, "tool": 0.05},
    "template_progress":    {"progress": 0.95, "revisit": 0.03, "tool": 0.05},
    "boilerplate_progress": {"progress": 0.95, "revisit": 0.03, "tool": 0.05},
    "retry_then_recover":   {"progress": 0.80, "revisit": 0.15, "tool": 0.05},
}


class OracleBackend:
    """Answers every question for one trace using its family's script."""

    name = "oracle"

    def __init__(self, family: str):
        if family not in FAMILY_ORACLE:
            raise KeyError(f"no oracle script for family {family!r}")
        self.family = family
        self.calls = 0

    def ask(self, state, questions: Mapping[str, Mapping]) -> JudgeResponse:
        self.calls += 1
        script = FAMILY_ORACLE[self.family]
        answers: Dict[str, Answer] = {}
        for qid, question in questions.items():
            if qid.startswith("progress_"):
                p = script["progress"]
            elif qid.startswith("revisit_"):
                p = script["revisit"]
            elif qid.startswith("tool_repeat_"):
                p = script["tool"]
            else:  # the auxiliary "stagnation" score question — not used for scoring
                continue
            answers[qid] = Answer(type="noul", value=p, probability=max(p, 1 - p))
        return JudgeResponse(answers=answers, model=f"oracle:{self.family}")


def oracle_scorer(trace: Trace, family: str, **metric_kwargs):
    """Build a fresh OracleBackend-backed metric and score one trace."""
    from semloop import SemanticLoopDetection

    backend = OracleBackend(family)
    metric = SemanticLoopDetection(backend, **metric_kwargs)
    return metric.measure(trace)
