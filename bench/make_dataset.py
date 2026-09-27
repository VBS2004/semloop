"""Build the benchmark dataset.

Ten families, each isolating one behaviour. Six are loops the metric should
catch, four are healthy runs it should leave alone. Families 5 and 6 are the
cases today's lexical check already handles — they are here so a replacement
has to prove it does not regress them.

The traces are written by hand from templates, not sampled from production.
That is enough to show a failure mode exists and is reproducible; it says
nothing about how often each family occurs in real traffic. Run the harness on
your own traces for that.

    python bench/make_dataset.py            # writes bench/dataset.jsonl
    python bench/make_dataset.py --per 10   # more instances per family
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

OUT_PATH = Path(__file__).with_name("dataset.jsonl")

TASKS = [
    {
        "goal": "Find why /api/orders returns 500 for guest checkout and fix it",
        "subject": "the guest checkout handler",
        "artifacts": ["orders_controller.py", "guest_session.py", "checkout_serializer.py", "cart_totals.py", "webhook_retry.py"],
        "target": "the missing session guard",
    },
    {
        "goal": "Make the nightly export job finish under ten minutes",
        "subject": "the export pipeline",
        "artifacts": ["export_job.py", "row_batcher.py", "s3_writer.py", "manifest_builder.py", "retention_sweep.py"],
        "target": "the per-row query in the batcher",
    },
    {
        "goal": "Add pagination to the customers list endpoint",
        "subject": "the customers endpoint",
        "artifacts": ["customers_view.py", "paginator.py", "list_serializer.py", "query_builder.py", "api_docs.md"],
        "target": "the unbounded query",
    },
    {
        "goal": "Track down the flaky test in the billing suite",
        "subject": "the billing test suite",
        "artifacts": ["test_invoices.py", "clock_fixture.py", "proration.py", "conftest.py", "test_refunds.py"],
        "target": "the frozen-clock fixture",
    },
    {
        "goal": "Upgrade the service to the new auth library without breaking sessions",
        "subject": "the auth integration",
        "artifacts": ["auth_client.py", "session_store.py", "middleware.py", "token_refresh.py", "login_view.py"],
        "target": "the token refresh path",
    },
]

# Ten ways to say "I am going to search the codebase for the handler" — long
# enough that the lexical check does not skip the pair, worded differently
# enough that bigram overlap and SequenceMatcher both stay low.
PARAPHRASES = [
    "Right, the sensible opening move here is to hunt through {subject} until I locate whichever function actually assembles the response, because without that anchor every later guess would be speculation rather than evidence.",
    "What I need first is a foothold: by grepping across the repository for the entry point behind {subject}, I can establish where control flow begins and stop reasoning in the abstract about code I have not read.",
    "My plan begins with locating the definition that drives {subject}. Until that file is open in front of me, any hypothesis about the failure would rest on assumption instead of the actual implementation.",
    "The obvious starting point is a repository-wide search aimed at {subject}, since pinning down its real implementation gives me something concrete to reason about rather than a mental model I invented.",
    "Before anything else I should trace {subject} back to its source. Finding that function means later conclusions rest on code I have inspected, not on a guess about how it probably behaves.",
    "Step one is orientation. If I scan the project tree for whatever powers {subject}, I get a factual base to work from, which beats theorising about internals I have never opened.",
    "To make headway I want the definition behind {subject} in view, right there on screen. Searching the sources for it converts this from guesswork into an investigation grounded in the implementation itself.",
    "I will start by combing the entire codebase for the routine responsible for {subject}, because reading the real thing is the only way to replace assumptions with actual evidence about what happens at runtime.",
    "First things first: find where {subject} is implemented. A search over the project gives me the actual code path, so my later reasoning is anchored in fact rather than in a plausible story.",
    "The natural first action is tracking down {subject} in the sources. Having the genuine implementation open means every subsequent step builds on observed behaviour instead of imagined behaviour.",
]

# Two alternating states for the A-B-A-B family: each step differs plainly from
# its neighbour, so consecutive comparison sees healthy variation.
ALTERNATING_A = [
    "Reading {artifact} now to understand how the request is validated before it reaches the database layer, since the stack trace points somewhere inside this file and I want the exact line.",
    "Opening {artifact} again to follow the validation path end to end, because the traceback lands in this module and I still need the precise statement that raises.",
]
ALTERNATING_B = [
    "The test suite is the fastest way to confirm this, so I am running the billing tests to see whether the error reproduces with the fixtures the CI machine uses.",
    "Let me execute the suite once more and watch for the failure, since a live reproduction tells me far more than continuing to read code in isolation.",
]

PROGRESS_STEPS = [
    "Searching the repository for the handler behind {subject} so I know which module actually builds the response before I start forming any theory about the fault.",
    "Found it in {artifact}. Reading the function now: it builds the payload from a session object that earlier middleware is supposed to populate on every request.",
    "The session object is optional in the guest path, so the attribute access raises. That explains the failure and points at {target} as the thing to change.",
    "Writing a regression test that drives the guest path with no session, so the fix is pinned by something executable rather than by my reading of the code.",
    "Test fails as expected, so the diagnosis holds. Applying the guard in {artifact} and rerunning the suite to confirm nothing else depended on the old behaviour.",
    "Suite is green, twenty-two passed. Recording what changed and why in the commit message so the next person does not have to rediscover {target}.",
]

# Same sentence shape, different file each time: real progress that the lexical
# check reads as repetition. Kept above 20 meaningful words so the pair is
# actually compared rather than skipped as too short.
TEMPLATE_PROGRESS = (
    "Now checking {artifact} for the deprecated call, reading through its imports "
    "and every function it exports one by one, then carefully noting whether "
    "anything in this particular file needs migrating before I move on to the "
    "next file waiting in the list."
)

# A long, near-fixed preamble with only a short unique tail: the boilerplate
# alone carries enough meaningful-word overlap to push consecutive pairs over
# the 0.85 similarity threshold, even though every step names a different file
# and does different real work. Paired with BOILERPLATE_BODIES below, not with
# PROGRESS_STEPS, whose bodies are long enough to dilute the effect.
BOILERPLATE = (
    "Based on the information provided so far, and having now carefully "
    "reviewed everything available up to this point in the investigation, "
    "weighing each of the alternatives in turn before committing to anything "
    "further, and keeping the overall goal firmly in mind throughout, "
    "the next concrete step I am therefore going to take right now, "
    "after all of that consideration, is as follows: "
)

BOILERPLATE_BODIES = [
    "check {artifact} directly.",
    "the middleware runs too late.",
    "add a guard right here.",
    "tests pass, twenty two cases.",
]

TERSE_LOOP = [
    "Retrying the search.",
    "Searching again.",
    "Let me search once more.",
    "Trying the search again.",
]

RETRY_THEN_RECOVER = [
    "Running the test suite to reproduce the reported failure and see the traceback with my own eyes rather than trusting the summary in the ticket.",
    "That run died on a missing environment variable rather than the bug itself, so the same command again with DATABASE_URL exported should get past setup.",
    "Setup passes now and the real failure appears in {artifact}, a different error from the one the ticket described, which changes what I need to look at.",
    "Reading {artifact} around the failing assertion to see which fixture supplies the timestamp, since the mismatch suggests {target} is the culprit.",
]


def _tool(name: str, args: Dict[str, Any], output: str) -> Dict[str, Any]:
    return {"type": "tool", "name": name, "input": args, "output": output}


def _llm(output: str, name: str = "reason") -> Dict[str, Any]:
    return {"type": "llm", "name": name, "output": output}


def paraphrased_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    picks = rng.sample(PARAPHRASES, 4)
    steps: List[Dict[str, Any]] = []
    for text in picks:
        steps.append(_llm(text.format(subject=task["subject"])))
        steps.append(
            _tool("grep", {"pattern": task["subject"], "path": "."}, "no matches")
        )
    return steps


def terse_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for text in rng.sample(TERSE_LOOP, 4):
        steps.append(_llm(text))
        steps.append(_tool("grep", {"pattern": task["target"], "path": "."}, "no matches"))
    return steps


def alternating_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    artifact = rng.choice(task["artifacts"])
    a_texts, b_texts = list(ALTERNATING_A), list(ALTERNATING_B)
    steps: List[Dict[str, Any]] = []
    for round_index in range(3):
        steps.append(_llm(a_texts[round_index % len(a_texts)].format(artifact=artifact)))
        steps.append(_tool("read_file", {"path": artifact}, "120 lines"))
        steps.append(_llm(b_texts[round_index % len(b_texts)]))
        steps.append(_tool("run_tests", {"suite": "billing"}, "1 failed"))
    return steps


def cosmetic_args_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Same search six times, with an argument tweaked so the counter resets."""
    picks = rng.sample(PARAPHRASES, 6)
    variations = [
        {"pattern": task["target"], "path": "."},
        {"pattern": task["target"], "path": "./"},
        {"pattern": task["target"], "path": ".", "case_sensitive": False},
        {"pattern": task["target"], "path": ".", "max_results": 50},
        {"pattern": task["target"], "path": ".", "max_results": 51},
        {"pattern": task["target"], "path": ".", "sort": "asc"},
    ]
    steps: List[Dict[str, Any]] = []
    for text, args in zip(picks, variations):
        steps.append(_llm(text.format(subject=task["subject"])))
        steps.append(_tool("grep", args, "no matches"))
    return steps


def verbatim_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """The case the lexical check is good at: identical reasoning, repeated."""
    text = rng.choice(PARAPHRASES).format(subject=task["subject"])
    steps: List[Dict[str, Any]] = []
    for _ in range(4):
        steps.append(_llm(text))
        steps.append(_tool("grep", {"pattern": task["subject"], "path": "."}, "no matches"))
    return steps


def identical_tool_loop(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Also caught today: the same tool call with byte-identical arguments."""
    text = rng.choice(PARAPHRASES).format(subject=task["subject"])
    steps: List[Dict[str, Any]] = []
    for _ in range(6):
        steps.append(_llm(text))
        steps.append(_tool("run_tests", {"suite": "billing", "k": "invoice"}, "1 failed"))
    return steps


def progress(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    artifact = rng.choice(task["artifacts"])
    steps: List[Dict[str, Any]] = []
    for index, text in enumerate(PROGRESS_STEPS):
        steps.append(
            _llm(
                text.format(
                    subject=task["subject"], artifact=artifact, target=task["target"]
                )
            )
        )
        if index == 0:
            steps.append(_tool("grep", {"pattern": task["subject"], "path": "."}, f"{artifact}:41"))
        elif index == 1:
            steps.append(_tool("read_file", {"path": artifact}, "120 lines"))
        elif index == 3:
            steps.append(_tool("write_file", {"path": f"test_{artifact}"}, "written"))
        elif index == 4:
            steps.append(_tool("run_tests", {"suite": "orders"}, "22 passed"))
    return steps


def template_progress(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Five files migrated, described in the same sentence shape each time."""
    steps: List[Dict[str, Any]] = []
    for artifact in task["artifacts"]:
        steps.append(_llm(TEMPLATE_PROGRESS.format(artifact=artifact)))
        steps.append(_tool("read_file", {"path": artifact}, "ok"))
    return steps


def boilerplate_progress(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Real progress buried under an identical preamble on every step.

    The unique content per step is deliberately terse — a genuine one-line
    update, the kind a real agent actually writes — so the fixed preamble in
    front of it dominates the word overlap between consecutive steps.
    """
    artifact = rng.choice(task["artifacts"])
    steps: List[Dict[str, Any]] = []
    for body in BOILERPLATE_BODIES:
        steps.append(_llm(BOILERPLATE + body.format(artifact=artifact)))
    return steps


def retry_then_recover(task: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Two attempts at the same command, then a genuine change of direction."""
    artifact = rng.choice(task["artifacts"])
    steps: List[Dict[str, Any]] = []
    for text in RETRY_THEN_RECOVER:
        steps.append(_llm(text.format(artifact=artifact, target=task["target"])))
    steps.insert(1, _tool("run_tests", {"suite": "billing"}, "error: DATABASE_URL unset"))
    steps.insert(3, _tool("run_tests", {"suite": "billing"}, "1 failed"))
    return steps


FAMILIES = [
    # name, builder, is a loop, stagnation should be detected, what it isolates
    ("paraphrased_loop", paraphrased_loop, True, True,
     "same intent reworded every step: bigram overlap and SequenceMatcher both stay low"),
    ("terse_loop", terse_loop, True, True,
     "steps under 20 meaningful words, which the lexical check skips outright"),
    ("alternating_loop", alternating_loop, True, True,
     "A-B-A-B cycle: consecutive pairs differ, so a consecutive-only comparison sees nothing"),
    ("cosmetic_args_loop", cosmetic_args_loop, True, True,
     "same search repeated with a cosmetic argument change, resetting the repetition counter"),
    ("verbatim_loop", verbatim_loop, True, True,
     "identical reasoning repeated: the lexical check already catches this"),
    ("identical_tool_loop", identical_tool_loop, True, True,
     "same tool and arguments six times: the lexical check already catches this"),
    ("progress", progress, False, False,
     "a healthy run that should not be flagged"),
    ("template_progress", template_progress, False, False,
     "five different files described in one sentence shape: lexical reads repetition, work is real"),
    ("boilerplate_progress", boilerplate_progress, False, False,
     "an identical preamble on every step, with real progress underneath"),
    ("retry_then_recover", retry_then_recover, False, False,
     "two goes at one command, then a change of approach: an ordinary retry, not a loop"),
]


def build(per_family: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for name, builder, is_loop, stagnation, isolates in FAMILIES:
        for instance in range(per_family):
            task = TASKS[instance % len(TASKS)]
            steps = builder(task, rng)
            rows.append(
                {
                    "trace_id": f"{name}-{instance:02d}",
                    "family": name,
                    "isolates": isolates,
                    "loop": is_loop,
                    "stagnation": stagnation,
                    "goal": task["goal"],
                    "root": {
                        "type": "agent",
                        "name": "engineer",
                        "input": task["goal"],
                        "output": "",
                        "children": [
                            {**step, "children": []} for step in steps
                        ],
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per", type=int, default=6, help="instances per family")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    rows = build(args.per, args.seed)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} traces across {len(FAMILIES)} families to {args.out}")


if __name__ == "__main__":
    main()
