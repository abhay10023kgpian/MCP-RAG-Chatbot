"""
memory.py — Token-Aware Conversation Summarization Manager
============================================================
Implements a token-budget memory strategy that:
  1. Estimates token count of the conversation history
  2. If under budget, sends everything as-is
  3. If over budget, summarizes older messages and keeps only
     recent messages that fit within the token budget

Optimized for Groq free tier (8000 TPM rate limit).

Architecture:
  [System Prompt] + [Summary of old turns] + [Recent messages within budget] → sent to LLM

Token estimation uses a simple char-based heuristic (~4 chars per token)
to avoid requiring tiktoken or model-specific tokenizers.
"""

import os
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

# ─── Configuration ───
# Max tokens to allocate for conversation history (excluding system prompt).
# With 8000 TPM on Groq free tier, we need to be conservative:
#   ~200 system prompt + ~2000 history + ~500 user+tools + ~1000 response = ~3700/turn
# This leaves headroom for summarization calls.
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))

# Max tokens for the summary itself. Keeps summaries compact.
MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "300"))

# Minimum number of recent messages to always keep (even if over budget).
# This ensures the LLM always sees at least the last exchange.
MIN_RECENT_MESSAGES = int(os.getenv("MIN_RECENT_MESSAGES", "4"))


# ─── Per-thread summary cache ───
# Maps thread_id -> running summary string
_summary_cache: dict[str, str] = {}


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count from text using a char-based heuristic.
    ~4 characters per token is a reasonable approximation for English text
    with LLM models (slightly conservative to avoid overflows).
    """
    return max(1, len(text) // 4)


def _message_tokens(msg: BaseMessage) -> int:
    """Estimate tokens for a single message (content + overhead)."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    # Add ~4 tokens overhead per message for role tags, formatting
    return _estimate_tokens(content) + 4


def _total_tokens(messages: list[BaseMessage]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(_message_tokens(m) for m in messages)


def _split_by_token_budget(
    messages: list[BaseMessage],
    max_tokens: int,
    min_recent: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    Split messages into (old_messages, recent_messages) based on token budget.

    Walks backwards from the end of the message list, accumulating tokens
    until the budget is exhausted. Always keeps at least `min_recent` messages.

    Returns:
        (old_messages, recent_messages) where recent_messages fit within max_tokens
    """
    if not messages:
        return [], []

    total = _total_tokens(messages)
    if total <= max_tokens:
        # Everything fits — no trimming needed
        return [], messages

    # Walk backwards, accumulating messages within the token budget
    recent_tokens = 0
    split_idx = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = _message_tokens(messages[i])
        messages_kept = len(messages) - i

        if recent_tokens + msg_tokens > max_tokens and messages_kept > min_recent:
            # This message would exceed budget and we have enough recent messages
            split_idx = i + 1
            break
        
        recent_tokens += msg_tokens
        split_idx = i

    # Ensure we don't split in the middle of a tool-call sequence.
    # If split_idx lands on a ToolMessage, move it back to include
    # the preceding AI message that triggered the tool call.
    while split_idx > 0 and isinstance(messages[split_idx], ToolMessage):
        split_idx -= 1

    return messages[:split_idx], messages[split_idx:]


async def _generate_summary(
    messages_to_summarize: list[BaseMessage],
    existing_summary: Optional[str],
    llm: ChatGroq,
) -> str:
    """
    Generate a concise summary of the given messages.
    If an existing summary is provided, it's incorporated as prior context.

    Designed to produce summaries under MAX_SUMMARY_TOKENS (~300 tokens / ~200 words).
    """
    # Build the content to summarize
    conversation_text = []
    for msg in messages_to_summarize:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content[:200]}")
        elif isinstance(msg, AIMessage) and msg.content:
            # Truncate long AI responses to save tokens in the summarization call
            conversation_text.append(f"Assistant: {msg.content[:200]}")
        # Skip ToolMessages and SystemMessages — they're implementation details

    if not conversation_text:
        return existing_summary or ""

    conversation_block = "\n".join(conversation_text)

    if existing_summary:
        prompt = (
            f"Update this conversation summary with the new messages below. "
            f"Be very concise — max 150 words. Capture key topics and answers only.\n\n"
            f"CURRENT SUMMARY:\n{existing_summary}\n\n"
            f"NEW MESSAGES:\n{conversation_block}\n\n"
            f"UPDATED SUMMARY:"
        )
    else:
        prompt = (
            f"Summarize this conversation in max 150 words. "
            f"Capture key topics discussed and answers given.\n\n"
            f"CONVERSATION:\n{conversation_block}\n\n"
            f"SUMMARY:"
        )

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


async def trim_and_summarize(
    messages: list[BaseMessage],
    thread_id: str,
    llm: ChatGroq,
) -> list[BaseMessage]:
    """
    Apply token-budget summarization to the message list.

    If the conversation fits within MAX_HISTORY_TOKENS, returns unchanged.

    If over budget:
      1. Splits into (old, recent) based on token budget
      2. Summarizes old messages into a compact SystemMessage
      3. Returns [summary] + [recent messages]

    Args:
        messages: Full message list from LangGraph state
        thread_id: Conversation thread ID (for caching summaries)
        llm: The Groq LLM instance to use for summarization

    Returns:
        Trimmed message list with optional summary prefix
    """
    total = _total_tokens(messages)

    if total <= MAX_HISTORY_TOKENS:
        return messages

    old_messages, recent_messages = _split_by_token_budget(
        messages, MAX_HISTORY_TOKENS, MIN_RECENT_MESSAGES
    )

    if not old_messages:
        # Even with budget exceeded, nothing to summarize
        # (all messages are in the "recent" minimum)
        return messages

    # Generate or update the summary
    existing_summary = _summary_cache.get(thread_id)
    new_summary = await _generate_summary(old_messages, existing_summary, llm)
    _summary_cache[thread_id] = new_summary

    # Build the summarized message list
    summary_message = SystemMessage(
        content=(
            f"[CONVERSATION SUMMARY — older messages summarized to save tokens]\n"
            f"{new_summary}\n"
            f"[END SUMMARY — recent messages follow]"
        )
    )

    return [summary_message] + recent_messages


def clear_summary(thread_id: str) -> None:
    """
    Clear the cached summary for a thread.
    Call this when a user resets their conversation.
    """
    _summary_cache.pop(thread_id, None)


def get_summary(thread_id: str) -> Optional[str]:
    """
    Get the current cached summary for a thread, if any.
    Useful for debugging.
    """
    return _summary_cache.get(thread_id)
