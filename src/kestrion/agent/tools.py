from kestrion.core.errors import InputRequired
from kestrion.core.types import AgentState
from .decorators import tool


@tool
def ask_human(question: str, _state: AgentState | None = None) -> str:
    """
    Ask the human user a question and wait for their text response.
    
    Use this tool when you are missing critical information that only the user can provide
    (e.g. asking for preferences, confirmation of ambiguous choices, or missing credentials).
    """
    if _state is None:
        raise ValueError("ask_human requires engine state to be passed.")

    import hashlib
    key = hashlib.md5(question.encode()).hexdigest()

    inputs = _state.scratch.get("_human_inputs", {})
    if key in inputs:
        return inputs.pop(key)
    if "ask_human" in inputs:
        # Backward compatibility for existing checkpoints
        return inputs.pop("ask_human")

    # We don't have the answer yet. Pause the run.
    raise InputRequired("ask_human", kwargs={"question": question}, question=question)
