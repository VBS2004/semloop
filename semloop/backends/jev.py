"""TypeSafe Jev backend (System One API).

One HTTP request carries the whole trace as state plus every question. The
questions are answered in parallel and cannot see each other's answers, which
is what makes a per-pair fan-out affordable: the trace is billed once.

Wire format follows the official SDK (`typesafe_sdk`, read 2026-09-27):

    POST {base_url}/v1/systemone
    {"model": ..., "state": ..., "questions": {id: {"type": ..., ...}}}

    -> {"model": ..., "usage": {"input_tokens": n},
        "answers": {id: {"type": "noul", "noul": 0.93}
                    | {"type": "choice", "choice": ..., "confidence": ..., "probabilities": {...}}
                    | {"type": "score", "score": ..., "confidence": ..., "probabilities": {...}}}}

Only the standard library is used, so the package has no install-time deps.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional

from . import Answer, JudgeResponse

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
SYSTEM_ONE_PATH = "/v1/systemone"

#: Published price: $0.042 per million input tokens, output free.
#: https://typesafe.ai/blog/introducing-system-one-models-and-jev
INPUT_USD_PER_TOKEN = 0.042 / 1_000_000


class JevError(RuntimeError):
    """The API rejected the request or returned something unusable."""


class JevBackend:
    """Calls the TypeSafe System One API."""

    name = "jev"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        path: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise JevError(
                "No API key. Pass api_key= or set TYPESAFE_API_KEY in the environment."
            )
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", DEFAULT_MODEL)
        self.base_url = (
            base_url or os.environ.get("TYPESAFE_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        # Some hosts (e.g. an OpenRouter proxy) serve the System One protocol
        # at a path of their own rather than TypeSafe's own /v1/systemone —
        # pass path="" when base_url is already the full endpoint.
        self.path = SYSTEM_ONE_PATH if path is None else path
        self.timeout = timeout

    def ask(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> JudgeResponse:
        if not questions:
            return JudgeResponse(answers={}, model=self.model)

        payload = json.dumps(
            {"model": self.model, "state": state, "questions": dict(questions)}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{self.path}",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs a live API
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise JevError(f"HTTP {exc.code} from System One API: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - needs a network
            raise JevError(f"Could not reach {self.base_url}: {exc.reason}") from exc
        latency_ms = (time.monotonic() - started) * 1000

        return parse_system_one_response(body, latency_ms=latency_ms)


def parse_system_one_response(
    body: Mapping[str, Any], *, latency_ms: float = 0.0
) -> JudgeResponse:
    """Turn a System One response body into normalised answers.

    Kept separate from the HTTP call so it can be tested against recorded
    payloads without a key.
    """
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise JevError(f"Response has no answers object: {str(body)[:200]}")

    answers: Dict[str, Answer] = {}
    for qid, raw in raw_answers.items():
        if not isinstance(raw, Mapping):
            raise JevError(f"Answer {qid!r} is not an object")
        answers[str(qid)] = _parse_answer(str(qid), raw)

    usage = body.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return JudgeResponse(
        answers=answers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=input_tokens * INPUT_USD_PER_TOKEN,
        latency_ms=latency_ms,
        model=str(body.get("model", "")),
    )


def _parse_answer(qid: str, raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        p = float(raw["noul"])
        return Answer(
            type="noul",
            value=p,
            probability=max(p, 1.0 - p),
            probabilities={True: p, False: 1.0 - p},
        )
    if kind == "choice":
        probabilities = {
            str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()
        }
        label = str(raw["choice"])
        return Answer(
            type="choice",
            value=label,
            probability=probabilities.get(label, 1.0),
            probabilities=dict(probabilities),
            confidence=_optional_float(raw.get("confidence")),
        )
    if kind == "score":
        probabilities = {
            int(k): float(v) for k, v in (raw.get("probabilities") or {}).items()
        }
        score = float(raw["score"])
        nearest = min(probabilities, key=lambda lvl: abs(lvl - score), default=None)
        return Answer(
            type="score",
            value=score,
            probability=probabilities.get(nearest, 1.0) if nearest is not None else 1.0,
            probabilities=dict(probabilities),
            confidence=_optional_float(raw.get("confidence")),
        )
    raise JevError(f"Answer {qid!r} has unsupported type {kind!r}")


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def estimate_cost_usd(input_tokens: int) -> float:
    """Cost of a request at the published input price. Output tokens are free."""
    return input_tokens * INPUT_USD_PER_TOKEN
