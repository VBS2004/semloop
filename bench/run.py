"""Run the benchmark.

    python bench/run.py                      # lexical baseline only, no key needed
    python bench/run.py --judge jev          # adds Jev, needs TYPESAFE_API_KEY
    python bench/run.py --judge llm --model gpt-4o-mini   # adds an LLM judge

Two numbers are reported for each system:

* **stagnation signal** — did it detect reasoning stagnation where the label says
  it should? This is the like-for-like comparison, since the stagnation
  sub-signal is the only thing a semantic judge replaces.
* **metric verdict** — did the whole metric fail the trace at its threshold?
  Worth watching, because the sub-scores are weighted 0.40 / 0.35 / 0.25, so a
  stagnation score of 0.0 on its own only drags the total to 0.65 and still
  passes a 0.5 threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semloop import (  # noqa: E402
    JevBackend,
    LLMJudgeBackend,
    LoopResult,
    SemanticLoopDetection,
    Trace,
    lexical_loop_score,
)

DATASET = Path(__file__).with_name("dataset.jsonl")


@dataclass
class Row:
    trace: Trace
    family: str
    expect_loop: bool
    expect_stagnation: bool


@dataclass
class Tally:
    name: str
    stagnation_hits: int = 0
    stagnation_total: int = 0
    verdict_hits: int = 0
    verdict_total: int = 0
    uncertain: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    latency_ms: float = 0.0
    latencies: int = 0
    per_family: Dict[str, List[bool]] = None  # type: ignore[assignment]
    per_family_verdict: Dict[str, List[bool]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_family is None:
            self.per_family = {}
        if self.per_family_verdict is None:
            self.per_family_verdict = {}


def load(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                Row(
                    trace=Trace.from_dict(raw),
                    family=raw["family"],
                    expect_loop=bool(raw["loop"]),
                    expect_stagnation=bool(raw["stagnation"]),
                )
            )
    return rows


def evaluate(
    name: str, scorer: Callable[[Row], LoopResult], rows: List[Row], threshold: float
) -> Tally:
    tally = Tally(name=name)
    for row in rows:
        try:
            result = scorer(row)
        except Exception as exc:  # a judge failure is reported, never swallowed
            tally.errors += 1
            print(f"  ! {row.trace.trace_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        detected = result.breakdown.get("reasoning_stagnation", 1.0) < 1.0
        correct = detected == row.expect_stagnation
        tally.stagnation_total += 1
        tally.stagnation_hits += int(correct)
        tally.per_family.setdefault(row.family, []).append(correct)

        verdict_correct = result.flags_loop(threshold) == row.expect_loop
        tally.verdict_total += 1
        tally.verdict_hits += int(verdict_correct)
        tally.per_family_verdict.setdefault(row.family, []).append(verdict_correct)

        tally.uncertain += int(result.uncertain)
        tally.cost_usd += result.cost_usd
        tally.input_tokens += result.input_tokens
        if result.latency_ms:
            tally.latency_ms += result.latency_ms
            tally.latencies += 1
    return tally


def pct(hits: int, total: int) -> str:
    return "n/a" if not total else f"{hits}/{total} ({100 * hits / total:.0f}%)"


def report(tallies: List[Tally], rows: List[Row]) -> None:
    families = sorted({row.family for row in rows})
    width = max(len(f) for f in families) + 2

    print("\nStagnation signal, correct calls per family (like-for-like: reasoning check only)")
    header = "family".ljust(width) + "".join(t.name.center(16) for t in tallies)
    print(header)
    print("-" * len(header))
    for family in families:
        line = family.ljust(width)
        for tally in tallies:
            results = tally.per_family.get(family, [])
            cell = "n/a" if not results else f"{sum(results)}/{len(results)}"
            line += cell.center(16)
        print(line)

    print("\nMetric verdict, correct calls per family (whole combined score vs threshold)")
    print(header)
    print("-" * len(header))
    for family in families:
        line = family.ljust(width)
        for tally in tallies:
            results = tally.per_family_verdict.get(family, [])
            cell = "n/a" if not results else f"{sum(results)}/{len(results)}"
            line += cell.center(16)
        print(line)

    print("\nTotals")
    for tally in tallies:
        mean_latency = (
            f"{tally.latency_ms / tally.latencies:.0f} ms" if tally.latencies else "n/a"
        )
        print(
            f"  {tally.name}: stagnation {pct(tally.stagnation_hits, tally.stagnation_total)}"
            f" · metric verdict {pct(tally.verdict_hits, tally.verdict_total)}"
            f" · undecided {tally.uncertain}"
            f" · errors {tally.errors}"
            f" · ${tally.cost_usd:.4f} ({tally.input_tokens} input tokens)"
            f" · {mean_latency} mean"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        choices=["jev", "llm", "oracle"],
        help=(
            "add a judged system; repeatable. 'oracle' needs no API key or network "
            "access — it scripts answers from the dataset's own family labels, purely "
            "to confirm the scoring mechanism, not to measure real judge accuracy "
            "(see bench/oracle_backend.py)."
        ),
    )
    parser.add_argument("--model", default=None, help="model id for the chosen judge")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument(
        "--input-usd-per-mtok",
        type=float,
        default=0.0,
        help="input price for the LLM judge, per million tokens",
    )
    parser.add_argument(
        "--output-usd-per-mtok",
        type=float,
        default=0.0,
        help="output price for the LLM judge, per million tokens",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"{args.dataset} not found — run `python bench/make_dataset.py` first."
        )

    rows = load(args.dataset)
    print(f"{len(rows)} traces from {args.dataset.name}")

    tallies = [
        evaluate("lexical", lambda row: lexical_loop_score(row.trace), rows, args.threshold)
    ]

    for judge in args.judge:
        if judge == "oracle":
            from oracle_backend import OracleBackend

            def score_oracle(row: Row, _threshold=args.threshold, _lookahead=args.lookahead) -> LoopResult:
                metric = SemanticLoopDetection(
                    OracleBackend(row.family), threshold=_threshold, lookahead=_lookahead
                )
                return metric.measure(row.trace)

            tallies.append(evaluate("oracle", score_oracle, rows, args.threshold))
            continue

        if judge == "jev":
            backend: Any = JevBackend(model=args.model or "jev-latest", base_url=args.base_url)
            label = f"jev:{backend.model}"
        else:
            backend = LLMJudgeBackend(
                model=args.model or "gpt-4o-mini",
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                input_usd_per_token=args.input_usd_per_mtok / 1_000_000,
                output_usd_per_token=args.output_usd_per_mtok / 1_000_000,
            )
            label = f"llm:{backend.model}"
        metric = SemanticLoopDetection(
            backend, threshold=args.threshold, lookahead=args.lookahead
        )
        tallies.append(evaluate(label, lambda row, m=metric: m.measure(row.trace), rows, args.threshold))

    report(tallies, rows)


if __name__ == "__main__":
    main()
