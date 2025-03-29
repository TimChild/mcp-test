import reflex as rx
from dataclasses import dataclass


@dataclass
class Update:
    type_: str
    delta: str = ""


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str
