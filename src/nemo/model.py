"""Pydantic models for the chatbot."""

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, max_length=8000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role=Role.USER, content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role=Role.ASSISTANT, content=content))

    def to_openai_format(self) -> List[dict]:
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        self.messages = []


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    config_path: str = Field(default="config")

    @field_validator("groq_api_key")
    @classmethod
    def key_must_be_set(cls, v: str) -> str:
        if not v or v == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set. Create a .env file from .env.example "
                "and add your key from https://console.groq.com/"
            )
        return v