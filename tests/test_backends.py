"""Backend parsing tests, against recorded response shapes.

These don't call any API. ``parse_system_one_response`` and
``parse_judge_reply`` are pure functions specifically so their parsing can be
checked against fixed payloads, independent of network access.
"""

import pytest

from semloop.backends.jev import JevError, estimate_cost_usd, parse_system_one_response
from semloop.backends.llm import LLMJudgeError, parse_judge_reply


def test_parse_system_one_response_noul():
    body = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 512},
        "answers": {"progress_0_1": {"type": "noul", "noul": 0.92}},
    }
    result = parse_system_one_response(body)
    answer = result.answers["progress_0_1"]
    assert answer.type == "noul"
    assert answer.value == pytest.approx(0.92)
    assert answer.probability == pytest.approx(0.92)
    assert result.input_tokens == 512
    assert result.cost_usd == pytest.approx(estimate_cost_usd(512))


def test_parse_system_one_response_choice():
    body = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 100},
        "answers": {
            "category": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.81,
                "probabilities": {"billing": 0.81, "technical": 0.19},
            }
        },
    }
    result = parse_system_one_response(body)
    answer = result.answers["category"]
    assert answer.value == "billing"
    assert answer.probability == pytest.approx(0.81)
    assert answer.confidence == pytest.approx(0.81)


def test_parse_system_one_response_score():
    body = {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 100},
        "answers": {
            "stagnation": {
                "type": "score",
                "score": 1.8,
                "confidence": 0.7,
                "probabilities": {"0": 0.05, "1": 0.6, "2": 0.3, "3": 0.05},
            }
        },
    }
    result = parse_system_one_response(body)
    answer = result.answers["stagnation"]
    assert answer.value == pytest.approx(1.8)
    assert answer.type == "score"


def test_parse_system_one_response_missing_answers_raises():
    with pytest.raises(JevError):
        parse_system_one_response({"model": "jev", "usage": {}})


def test_parse_judge_reply_plain_json():
    questions = {"is_spam": {"type": "noul"}}
    content = '{"answers": {"is_spam": 0.87}}'
    answers = parse_judge_reply(content, questions)
    assert answers["is_spam"].value == pytest.approx(0.87)


def test_parse_judge_reply_strips_code_fence():
    questions = {"is_spam": {"type": "noul"}}
    content = '```json\n{"answers": {"is_spam": 0.2}}\n```'
    answers = parse_judge_reply(content, questions)
    assert answers["is_spam"].value == pytest.approx(0.2)


def test_parse_judge_reply_choice_must_match_a_label():
    questions = {"category": {"type": "choice", "criteria": {"billing": None, "technical": None}}}
    with pytest.raises(LLMJudgeError):
        parse_judge_reply('{"answers": {"category": "shipping"}}', questions)


def test_parse_judge_reply_missing_question_raises():
    questions = {"a": {"type": "noul"}, "b": {"type": "noul"}}
    with pytest.raises(LLMJudgeError):
        parse_judge_reply('{"answers": {"a": 0.5}}', questions)


def test_parse_judge_reply_score_out_of_range_raises():
    questions = {"severity": {"type": "score", "criteria": ["low", "high"]}}
    with pytest.raises(LLMJudgeError):
        parse_judge_reply('{"answers": {"severity": 5}}', questions)
