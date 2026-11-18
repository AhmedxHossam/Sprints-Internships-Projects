"""Deterministic routing: one initial attempt and at most three retries.

This module has no LangGraph/model/network dependency. ``END`` is a route
label; the graph maps it to LangGraph's END sentinel (``__end__``).
"""

from typing import Final, Literal, NotRequired, TypedDict

MAX_RETRIES: Final[int] = 3
Route = Literal["agent", "END"]


class RoutingState(TypedDict):
    """The small subset of graph state that controls termination."""

    loop_count: int
    needs_more_info: bool
    fatal_error: NotRequired[str]


def should_continue(state: RoutingState) -> Route:
    """Decide after critique; never delegate the retry budget to an LLM.

    loop_count is incremented once per completed critique. A value of 1
    means the initial attempt has completed. At 3, one retry remains; at
    4, the initial attempt plus all three retries have been consumed.
    Invalid counters also fail closed.
    """
    count = state["loop_count"]
    if count > MAX_RETRIES:
        return "END"
    if count < 0 or state.get("fatal_error"):
        return "END"
    if not state["needs_more_info"]:
        return "END"
    return "agent"


if __name__ == "__main__":
    # Explicit safety assertions required by the assignment.
    assert should_continue({"loop_count": 1, "needs_more_info": True}) == "agent"
    assert should_continue({"loop_count": 3, "needs_more_info": True}) == "agent"
    assert should_continue({"loop_count": 4, "needs_more_info": True}) == "END"
    assert should_continue({"loop_count": 1, "needs_more_info": False}) == "END"
    assert should_continue({"loop_count": 3, "needs_more_info": False}) == "END"
    assert should_continue({"loop_count": 4, "needs_more_info": False}) == "END"
    assert should_continue({"loop_count": -1, "needs_more_info": True}) == "END"
    assert should_continue({"loop_count": 1, "needs_more_info": True,
                            "fatal_error": "Model request failed"}) == "END"
    print("All routing safety assertions passed.")
