"""semloop — semantic loop detection for agent traces.

The deterministic parts of loop detection (identical tool calls, call-graph
cycles) are exact and free, so they are kept. The part that needs
understanding — has this agent actually got anywhere? — is asked of a decision
model instead of being approximated with word overlap.

    from semloop import Trace, SemanticLoopDetection, JevBackend

    trace = Trace.from_steps(goal="Fix the failing test", steps=[...])
    metric = SemanticLoopDetection(JevBackend())
    result = metric.measure(trace)
    print(result.score, result.reason)
"""

from .backends import Answer, Backend, JevBackend, JudgeResponse, LLMJudgeBackend, MockBackend
from .lexical import LoopResult, lexical_loop_score
from .semantic import (
    PairFinding,
    SemanticLoopDetection,
    ToolFinding,
    build_questions,
    build_state,
    build_tool_questions,
)
from .trace import Span, Trace

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Backend",
    "JevBackend",
    "JudgeResponse",
    "LLMJudgeBackend",
    "LoopResult",
    "MockBackend",
    "PairFinding",
    "SemanticLoopDetection",
    "Span",
    "ToolFinding",
    "Trace",
    "build_questions",
    "build_state",
    "build_tool_questions",
    "lexical_loop_score",
]
