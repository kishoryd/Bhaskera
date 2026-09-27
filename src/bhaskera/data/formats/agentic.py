"""
bhaskera.data.formats.agentic
=============================
Render tool-use (agentic) conversations with each model's own chat template.

Why the model's template: tool calling has no common wire format. Qwen3
wraps calls in <tool_call> JSON, Llama 3.x emits a bare JSON object after a
python/ipython header, Mistral uses [TOOL_CALLS] with 9-character call
ids. Training or evaluating an agentic model on any other serialization
teaches or tests the wrong thing, so rows are always rendered through
tokenizer.apply_chat_template(messages, tools=...).

Row shape (also accepted as JSON strings, which avoids Arrow merging every
row's nested JSON schema into one struct and null-filling missing keys):
    messages: [{role, content, tool_calls?, tool_call_id?}, ...]
    tools:    [{name, description, parameters} | {type: function, function: {...}}]
Assistant tool_calls may be {name, arguments} or OpenAI-style
{type, id, function: {name, arguments}}; arguments may be a dict or a JSON
string.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgenticTemplateError(ValueError):
    """The tokenizer's chat template cannot render tools / tool calls."""


def _load(x: Any) -> Any:
    if isinstance(x, (str, bytes)):
        return json.loads(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _strip_none(x: Any) -> Any:
    """Drop None-valued keys (Arrow fills absent struct fields with null)."""
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [_strip_none(v) for v in x]
    return x


def _json_schema(x: Any) -> Any:
    """BFCL writes type "dict"; JSON Schema (and chat templates) say "object"."""
    if isinstance(x, dict):
        out = {k: _json_schema(v) for k, v in x.items()}
        if out.get("type") == "dict":
            out["type"] = "object"
        return out
    if isinstance(x, list):
        return [_json_schema(v) for v in x]
    return x


def normalize_tools(tools: Any) -> list[dict]:
    """Any accepted tool list -> [{type: function, function: {...}}]."""
    tools = _strip_none(_load(tools) or [])
    out = []
    for t in tools:
        t = _load(t)
        fn = t.get("function", t) if t.get("type") == "function" else t
        out.append({"type": "function", "function": _json_schema(fn)})
    return out


def tool_call(name: str, arguments: Any, index: int = 0) -> dict:
    """One OpenAI-style call. ids are 9 alphanumerics, which Mistral requires."""
    if isinstance(arguments, (str, bytes)):
        arguments = json.loads(arguments)
    return {"type": "function", "id": f"call{index:05d}",
            "function": {"name": name, "arguments": arguments or {}}}


def normalize_messages(messages: Any) -> list[dict]:
    msgs = _strip_none(_load(messages) or [])
    out, n_calls = [], 0
    for m in msgs:
        m = dict(_load(m))
        calls = m.get("tool_calls")
        if calls:
            norm = []
            for c in _load(calls):
                fn = c.get("function", c)
                norm.append(tool_call(fn["name"], fn.get("arguments"), n_calls))
                n_calls += 1
            m["tool_calls"] = norm
            m.setdefault("content", "")
        out.append(m)
    return out


def render(tokenizer: Any, messages: Any, tools: Any = None,
           add_generation_prompt: bool = False,
           template_kwargs: dict | None = None) -> str:
    """messages (+ tools) through the model's own chat template.

    template_kwargs go to the template as variables, e.g.
    {"enable_thinking": False} for Qwen3 hybrid-thinking models; templates
    that do not use a variable ignore it.

    Raises AgenticTemplateError when the tokenizer has no chat template or
    its template silently drops the tool definitions.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise AgenticTemplateError("tokenizer has no chat_template")
    msgs = normalize_messages(messages)
    tool_list = normalize_tools(tools) if tools is not None else []
    try:
        text = tokenizer.apply_chat_template(
            msgs, tools=tool_list or None, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **(template_kwargs or {}),
        )
    except Exception as exc:  # Jinja raise_exception() from strict templates
        raise AgenticTemplateError(f"{type(exc).__name__}: {exc}") from exc
    if tool_list:
        missing = [t["function"]["name"] for t in tool_list
                   if t["function"]["name"] not in text]
        if missing:
            raise AgenticTemplateError(
                f"chat template ignored tool definitions (e.g. {missing[0]!r})")
    return text


def render_row(row: dict, tokenizer: Any, options: dict) -> str:
    """Format-registry entry point; see register_format("agentic")."""
    messages = row.get(options.get("messages_field", "messages"))
    tools = row.get(options.get("tools_field", "tools"))
    if messages is None:
        return ""
    return render(tokenizer, messages, tools, add_generation_prompt=False,
                  template_kwargs=options.get("template_kwargs"))
