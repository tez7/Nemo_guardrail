"""NeMo Guardrails wrapper for the chatbot."""

from pathlib import Path
from typing import Optional

from nemoguardrails import LLMRails, RailsConfig

from .model import ChatHistory


class GuardrailsChatbot:
    def __init__(self, config_path: str = "config"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Guardrails config not found at: {self.config_path}")

        self.config = RailsConfig.from_path(str(self.config_path))
        self.rails = LLMRails(self.config)

    def generate(self, user_message: str, history: Optional[ChatHistory] = None) -> str:
        messages = []
        if history and history.messages:
            for msg in history.messages:
                messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        response = self.rails.generate(messages=messages)

        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)

    async def generate_async(self, user_message: str, history: Optional[ChatHistory] = None) -> str:
        messages = []
        if history and history.messages:
            for msg in history.messages:
                messages.append({"role": msg.role.value, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        response = await self.rails.generate_async(messages=messages)

        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)