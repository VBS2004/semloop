"""An LLM-as-judge backend, for comparison against Jev.

Talks to any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
OpenRouter, a local server). It asks for one JSON object holding every answer,
which is how an LLM judge is normally wired up, then validates the shape.

This exists so the benchmark can report three columns — lexical, LLM judge,
Jev — rather than asserting that a decision model is cheaper. Prices vary per
model and change often, so cost is computed from the per-token rates you pass
in, and is 0 until you do.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Mapping, Optional

from . import Answer, JudgeResponse

DEFAULT_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """You judge an agent execution trace. You will be given a state \
and a set of questions, each with an id.

Answer every question. Reply with one JSON object and nothing else:
{"answers": {"<question id>": <answer>, ...}}

Answer shapes by question type:
- noul: a number from 0 to 1, the probability that the statement is true.
- choice: the exact label string of the option you pick.
- score: an integer index into the listed levels, starting at 0.

Do not add commentary, explanations or markdown fences."""


class LLMJudgeError(RuntimeError):
    """The judge call failed or returned something unusable."""


class LLMJudgeBackend:
    """One chat-completion call per trace, returning all answers as JSON."""

    name = "llm-judge"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        input_usd_per_token: float = 0.0,
        output_usd_per_token: float = 0.0,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise LLMJudgeError(
                f"No API key. Pass api_key= or set {api_key_env} in the environment."
            )
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.input_usd_per_token = input_usd_per_token
        self.output_usd_per_token = output_usd_per_token

    def ask(
        self, state: Any, questions: Mapping[str, Mapping[str, Any]]
    ) -> JudgeResponse:
        if not questions:
            return JudgeResponse(answers={}, model=self.model)

        user_content = json.dumps(
            {"state": state, "questions": dict(questions)}, ensure_ascii=False
        )
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
            raise LLMJudgeError(f"HTTP {exc.code} from judge: {detail}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - needs a network
            raise LLMJudgeError(f"Could not reach {self.base_url}: {exc.reason}") from exc
        latency_ms = (time.monotonic() - started) * 1000

        answers = parse_judge_reply(_content_of(body), questions)
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        return JudgeResponse(
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(
                input_tokens * self.input_usd_per_token
                + output_tokens * self.output_usd_per_token
            ),
            latency_ms=latency_ms,
            model=str(body.get("model", self.model)),
        )


def _content_of(body: Mapping[str, Any]) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMJudgeError(f"Judge reply has no message content: {str(body)[:200]}") from exc


def parse_judge_reply(
    content: str, questions: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Answer]:
    """Parse the judge's JSON reply into normalised answers.

    Tolerates a ```json fence, since models add one even when told not to.
    A missing or unparseable answer raises: silently scoring it as "no loop"
    would flatter the judge.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMJudgeError(f"Judge reply is not JSON: {content[:200]!r}") from exc

    raw = parsed.get("answers") if isinstance(parsed, Mapping) else None
    if not isinstance(raw, Mapping):
        raise LLMJudgeError(f"Judge reply has no answers object: {content[:200]!r}")

    answers: Dict[str, Answer] = {}
    for qid, question in questions.items():
        if qid not in raw:
            raise LLMJudgeError(f"Judge omitted an answer for {qid!r}")
        answers[qid] = _coerce(qid, str(question.get("type", "noul")), raw[qid], question)
    return answers


def _coerce(
    qid: str, qtype: str, value: Any, question: Mapping[str, Any]
) -> Answer:
    if qtype == "noul":
        try:
            p = float(value)
        except (TypeError, ValueError) as exc:
            raise LLMJudgeError(f"Answer for {qid!r} is not a number: {value!r}") from exc
        p = min(max(p, 0.0), 1.0)
        return Answer(
            type="noul",
            value=p,
            probability=max(p, 1.0 - p),
            probabilities={True: p, False: 1.0 - p},
        )
    if qtype == "choice":
        labels = list((question.get("criteria") or {}))
        label = str(value)
        if label not in labels:
            raise LLMJudgeError(
                f"Answer for {qid!r} is not one of the labels {labels}: {value!r}"
            )
        return Answer(type="choice", value=label, probability=1.0, probabilities={label: 1.0})
    levels = list(question.get("criteria") or [])
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMJudgeError(f"Answer for {qid!r} is not an integer level: {value!r}") from exc
    if not 0 <= level < len(levels):
        raise LLMJudgeError(f"Answer for {qid!r} is outside the {len(levels)} levels: {level}")
    return Answer(type="score", value=float(level), probability=1.0, probabilities={level: 1.0})
