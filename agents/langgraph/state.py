import os
from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str
    summary : str = ""
    score : int

    @field_validator("score")
    def score_positive(cls, value):
        if value < 0:
            raise ValueError("score must be positive")


from dataclasses import dataclass, field

@dataclass
class State:
    topic: str
    summary: str = ""
    message: list = field(default_factory=list)

from langgraph.graph import MessagesState


class State(MessagesState):
    user_name: str
    language: str