from pydantic import BaseModel


class State(BaseModel):
    held_object: str | None = None
    inventory: list[str] = []


class ToolOutput(BaseModel):
    success: bool
    output: State | float | bool | None = None
    feedback_msg: str = ""
