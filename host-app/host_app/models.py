from enum import StrEnum
from typing import Any
import reflex as rx
from dataclasses import dataclass


class UpdateTypes(StrEnum):
    start = "start"
    preprocess = "preprocess"
    ai_delta = "ai-delta"
    ai_message_end = "ai-message-end"
    tool_start = "tool-start"
    tool_end = "tool-end"
    end = "end"


@dataclass
class Update:
    type_: UpdateTypes
    delta: str = ""
    name: str = ""
    data: Any = None


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
