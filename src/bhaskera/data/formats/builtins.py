"""
bhaskera.data.formats.builtins
==============================
Built-in renderers for the most common SFT data layouts.

Imported lazily by ``formats.__init__._ensure_builtins_loaded`` so the
registry stays cheap to import.
"""
from __future__ import annotations

import json
import logging
from typing import Any, List

from . import register_format

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(x: Any) -> List:
    """Normalise numpy arrays / pyarrow lists to a plain Python list."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _to_dict(x: Any) -> dict:
    """Normalise a row entry that may arrive as numpy.void / dict-like."""
    if isinstance(x, dict):
        return x
    # numpy structured array element / pyarrow struct: fall back to dict()
    try:
        return dict(x)
    except Exception:
        return {"role": "user", "content": str(x)}


def _manual_chatml(messages: List[dict]) -> str:
    """
    Fallback ChatML rendering when the tokenizer has no chat_template.

    Format:
        <|im_start|>role
        content<|im_end|>
        <|im_start|>role
        content<|im_end|>
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def _apply_chat_template_safe(tokenizer: Any, messages: List[dict]) -> str:
    """
    Use tokenizer.apply_chat_template if a chat_template is available;
    otherwise fall back to manual ChatML. This is what makes the renderer
    portable across Llama-3, Qwen, Mistral, Param2, etc. — each tokenizer
    knows its own template.
    """
    has_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if not has_template:
        return _manual_chatml(messages)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        # A model's own chat template can *reject* otherwise-valid message
        # structures — the strict ones (Mistral, gpt-oss) call the Jinja
        # raise_exception() helper for things like non-alternating roles or
        # an unexpected role. Hit for real on Nemotron chat: two of six
        # models aborted a 528k-row tokenize run partway through, while the
        # other four tokenized the identical data fine.
        #
        # One malformed row should not kill the whole corpus, and silently
        # dropping rows would give each model a different training set and
        # break the cross-model comparison this pipeline exists for. Falling
        # back to manual ChatML keeps every model on the same rows; the
        # fallback differs in special tokens for that row only.
        logger.warning(
            "chat_template rejected a row (%s: %s) — falling back to manual "
            "ChatML for it. Roles: %s",
            type(exc).__name__,
            str(exc)[:120],
            [m.get("role") for m in messages][:12],
        )
        return _manual_chatml(messages)


# ---------------------------------------------------------------------------
# ChatML
# ---------------------------------------------------------------------------

@register_format("chatml")
def render_chatml(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``messages: [{role, content}, ...]`` (your data shape).

    Extra columns on the row (category, intent, flags, …) are simply ignored,
    so this works directly for the JSONL you posted.
    """
    messages_field = options.get("messages_field", "messages")
    messages = _to_list(row.get(messages_field, []))
    messages = [_to_dict(m) for m in messages]
    if not messages:
        return ""
    return _apply_chat_template_safe(tokenizer, messages)


# ---------------------------------------------------------------------------
# Agentic (tool use)
# ---------------------------------------------------------------------------

@register_format("agentic")
def render_agentic(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``messages`` (tool calls / tool results allowed) and ``tools``,
    rendered in the model's native tool-calling syntax (formats/agentic.py).

    Like chatml, a row the template rejects falls back to manual ChatML --
    tool schemas as JSON in a system turn, calls as JSON -- rather than
    aborting the corpus or silently dropping the row.
    """
    from .agentic import AgenticTemplateError, normalize_messages, normalize_tools, render_row

    try:
        return render_row(row, tokenizer, options)
    except AgenticTemplateError as exc:
        logger.warning("agentic: chat template unusable for a row (%s); manual ChatML", exc)
        msgs = normalize_messages(row.get(options.get("messages_field", "messages")) or [])
        tools = row.get(options.get("tools_field", "tools"))
        flat = []
        if tools is not None:
            flat.append({"role": "system",
                         "content": "Tools:\n" + json.dumps(normalize_tools(tools))})
        for m in msgs:
            content = m.get("content") or ""
            if m.get("tool_calls"):
                content += json.dumps([c["function"] for c in m["tool_calls"]])
            flat.append({"role": m.get("role", "user"), "content": content})
        return _manual_chatml(flat)


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


@register_format("alpaca")
def render_alpaca(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``instruction``, optional ``input``, and ``output`` columns.

    options:
      * use_chat_template: bool — if True and tokenizer has a chat_template,
        render as a 2-turn chat (user=instruction[+input], assistant=output)
        instead of the classic Alpaca prose template. Default False.
    """
    instruction = str(row.get("instruction", "") or "")
    inp         = str(row.get("input", "") or "")
    output      = str(row.get("output", "") or "")

    if options.get("use_chat_template", False):
        user_turn = instruction if not inp else f"{instruction}\n\n{inp}"
        messages = [
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": output},
        ]
        return _apply_chat_template_safe(tokenizer, messages)

    template = _ALPACA_WITH_INPUT if inp else _ALPACA_NO_INPUT
    return template.format(instruction=instruction, input=inp, output=output)


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

_SHAREGPT_ROLE_MAP = {
    "human":     "user",
    "user":      "user",
    "gpt":       "assistant",
    "assistant": "assistant",
    "chatgpt":   "assistant",
    "bard":      "assistant",
    "system":    "system",
    "tool":      "tool",
    "function":  "tool",
}


@register_format("sharegpt")
def render_sharegpt(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``conversations: [{from, value}, ...]`` (the ShareGPT layout).

    options:
      * conversations_field: str — column name (default "conversations")
      * role_map: dict[str, str] — override the from->role mapping
    """
    field    = options.get("conversations_field", "conversations")
    role_map = {**_SHAREGPT_ROLE_MAP, **(options.get("role_map") or {})}

    convs = _to_list(row.get(field, []))
    if not convs:
        return ""

    messages = []
    for c in convs:
        c = _to_dict(c)
        sender = str(c.get("from", "user")).lower()
        role = role_map.get(sender, "user")
        messages.append({"role": role, "content": c.get("value", "") or ""})

    return _apply_chat_template_safe(tokenizer, messages)
