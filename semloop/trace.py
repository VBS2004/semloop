"""Trace and span model, plus an adapter for deepeval's nested trace dicts.

The shape mirrors what ``deepeval`` hands to a metric via
``LLMTestCase._trace_dict``: a tree of spans, each with ``type``, ``name``,
``input``, ``output`` and ``children``. Nothing here depends on deepeval
being installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

SPAN_TYPES = ("agent", "llm", "tool", "retriever", "unknown")


@dataclass
class Span:
    """One step in an agent trace."""

    type: str = "unknown"
    name: str = "unnamed"
    input: Any = ""
    output: Any = ""
    children: List["Span"] = field(default_factory=list)

    @property
    def input_text(self) -> str:
        return _as_text(self.input)

    @property
    def output_text(self) -> str:
        return _as_text(self.output)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> Optional["Span"]:
        if not raw:
            return None
        return cls(
            type=str(raw.get("type", "unknown")),
            name=str(raw.get("name", "unnamed")),
            input=raw.get("input", ""),
            output=raw.get("output", ""),
            children=[
                child
                for child in (
                    cls.from_dict(c) for c in raw.get("children", []) or []
                )
                if child is not None
            ],
        )


@dataclass
class Trace:
    """A whole agent run: the goal it was given, and the span tree it produced."""

    goal: str
    root: Span
    trace_id: str = ""

    def walk(self) -> Iterator[Span]:
        """Depth-first walk over every span, root first."""

        def _walk(span: Span) -> Iterator[Span]:
            yield span
            for child in span.children:
                yield from _walk(child)

        yield from _walk(self.root)

    def spans_of_type(self, span_type: str) -> List[Span]:
        return [s for s in self.walk() if s.type == span_type]

    @property
    def llm_spans(self) -> List[Span]:
        return self.spans_of_type("llm")

    @property
    def tool_spans(self) -> List[Span]:
        return self.spans_of_type("tool")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Trace":
        root = Span.from_dict(raw.get("root"))
        if root is None:
            raise ValueError("trace has no root span")
        return cls(
            goal=str(raw.get("goal", "")),
            root=root,
            trace_id=str(raw.get("trace_id", "")),
        )

    @classmethod
    def from_deepeval_trace_dict(
        cls, trace_dict: Optional[Dict[str, Any]], goal: str = "", trace_id: str = ""
    ) -> "Trace":
        """Build a Trace from a deepeval ``LLMTestCase._trace_dict``."""
        root = Span.from_dict(trace_dict)
        if root is None:
            raise ValueError("empty deepeval trace dict")
        return cls(goal=goal, root=root, trace_id=trace_id)

    @classmethod
    def from_steps(
        cls,
        goal: str,
        steps: List[Dict[str, Any]],
        *,
        root_name: str = "agent",
        trace_id: str = "",
    ) -> "Trace":
        """Convenience builder: a flat list of steps under one agent span."""
        children = [
            Span(
                type=str(step.get("type", "llm")),
                name=str(step.get("name", "step")),
                input=step.get("input", ""),
                output=step.get("output", ""),
            )
            for step in steps
        ]
        return cls(
            goal=goal,
            root=Span(type="agent", name=root_name, input=goal, children=children),
            trace_id=trace_id,
        )


def _as_text(value: Any) -> str:
    """Render a span input/output as text, keeping dicts readable and stable."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
