"""Model access: chat, embeddings, and an offline scripted model for tests and evals."""

from .client import OpenAICompatChat, post_json_with_retry
from .embeddings import HashingEmbedder, OpenAICompatEmbedder
from .scripted import ScriptedChat, text_reply, tool_reply

__all__ = [
    "HashingEmbedder",
    "OpenAICompatChat",
    "OpenAICompatEmbedder",
    "ScriptedChat",
    "post_json_with_retry",
    "text_reply",
    "tool_reply",
]
