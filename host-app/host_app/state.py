import os
from typing import Any, AsyncIterator
import reflex as rx
from openai import OpenAI

from dotenv import load_dotenv
from reflex.event import EventType
from .process import get_response_updates
from .models import QA

load_dotenv()

SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""

# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    # New chat modal open state.
    modal_open: bool = False

    @rx.event
    def set_new_chat_name(self, name: str) -> None:
        """Set the name of the new chat.

        Args:
            form_data: A dict with the new chat name.
        """
        self.new_chat_name = name

    @rx.event
    def toggle_modal(self):
        """Toggle the modal."""
        self.modal_open = not self.modal_open

    @rx.event
    def create_chat(self):
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    @rx.event
    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    @rx.event
    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var(cache=True)
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    @rx.event
    async def process_question(
        self, form_data: dict[str, Any]
    ) -> AsyncIterator[EventType | None]:
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        # # The reflex default
        # question_processor = self.openai_process_question

        # My implementation
        question_processor = self.general_process_question

        async for event in question_processor(question):
            yield event

    async def general_process_question(
        self, question: str
    ) -> AsyncIterator[EventType | None]:
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        self.processing = True
        yield

        async for update in get_response_updates(
            question=question, message_history=self.chats[self.current_chat][:-1]
        ):
            match update.type_:
                case "ai-delta":
                    self.chats[self.current_chat][-1].answer += update.delta
                    self.chats = self.chats
                    yield
                case _:
                    yield rx.toast.error(f"Unknown update type: {update.type_}")

        self.processing = False

    async def openai_process_question(self, question: str):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """
        # NOTE: This is the Reflex default implementation

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,  # type: ignore
            stream=True,
        )  # type: ignore

        # Stream the results, yielding after every word.
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer_text = item.choices[0].delta.content
                # Ensure answer_text is not None before concatenation
                if answer_text is not None:
                    self.chats[self.current_chat][-1].answer += answer_text
                else:
                    # Handle the case where answer_text is None, perhaps log it or assign a default value
                    # For example, assigning an empty string if answer_text is None
                    answer_text = ""
                    self.chats[self.current_chat][-1].answer += answer_text
                self.chats = self.chats
                yield

        # Toggle the processing flag.
        self.processing = False
