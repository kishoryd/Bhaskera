"""
bhaskera.data.formats.builtins
==============================
Built-in renderers for the most common SFT data layouts.

Imported lazily by ``formats.__init__._ensure_builtins_loaded`` so the
registry stays cheap to import.
"""
from __future__ import annotations

import logging
from typing import Any, List

from . import register_format

logger = logging.getLogger(__name__)


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


# ChatML

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


# Alpaca

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


# ShareGPT

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


# Prompt and completion

@register_format("prompt_completion")
def render_prompt_completion(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with a prompt column and a completion column (the layout of OpenAI's legacy fine-tuning files, and of many question/answer sets).

    options:
      * prompt_field: str — column name of the prompt (default "prompt")
      * completion_field: str — column name of the completion (default "completion")
      * separator: str — text placed between them (default "\n")
      * use_chat_template: bool — render as a 2-turn chat (user=prompt, assistant=completion) with the model's chat template instead
        of joining the two with the separator. Default False.

    Other question/answer layouts need only the two column names, for example ``{prompt_field: question, completion_field: answer}``
    or ``{prompt_field: instruction, completion_field: response}``.
    """
    prompt     = str(row.get(options.get("prompt_field", "prompt"), "") or "")
    completion = str(row.get(options.get("completion_field", "completion"), "") or "")
    if not prompt and not completion:
        return ""
    if options.get("use_chat_template", False):
        return _apply_chat_template_safe(tokenizer, [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}])
    return prompt + str(options.get("separator", "\n")) + completion


# Helpers for the formats below

def _plain(x: Any) -> Any:
    """Plain Python (dicts, lists, strings, numbers) from numpy arrays, pyarrow values and the like, with empty (None) fields dropped."""
    if x is None:
        return None
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _json_or_self(x: Any) -> Any:
    """A JSON text cell (a CSV column, or a field stored as text because its shape differs between rows) as the value it holds."""
    if isinstance(x, str) and x.lstrip()[:1] in ("[", "{"):
        import json
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def _template(tokenizer: Any, messages: List[dict], **kwargs: Any) -> str:
    """The model's chat template, with extra arguments (``tools=``), falling back to _apply_chat_template_safe's ChatML when there is none
    or the template rejects the row."""
    if not (hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)):
        return _manual_chatml(messages)
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    except Exception as exc:
        logger.warning("chat_template rejected a row (%s: %s) - falling back to manual ChatML for it", type(exc).__name__, str(exc)[:120])
        return _manual_chatml(messages)


def _col(row: dict, options: dict, key: str, default: str) -> str:
    return str(row.get(options.get(key + "_field", default), "") or "")


# Instruction, context and response (Dolly)

@register_format("dolly")
def render_dolly(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``instruction``, an optional ``context`` and a ``response`` (Databricks Dolly).

    options:
      * instruction_field / context_field / response_field: str - other column names
      * use_chat_template: bool - render as a 2-turn chat (user = instruction + context, assistant = response) with the model's chat
        template instead of the ### Instruction / ### Context / ### Response layout. Default False.
    """
    instruction = _col(row, options, "instruction", "instruction")
    context     = _col(row, options, "context", "context")
    response    = _col(row, options, "response", "response")
    if not instruction and not response:
        return ""
    if options.get("use_chat_template", False):
        user = instruction if not context else f"{instruction}\n\n{context}"
        return _apply_chat_template_safe(tokenizer, [{"role": "user", "content": user}, {"role": "assistant", "content": response}])
    ctx = f"### Context:\n{context}\n\n" if context else ""
    return f"### Instruction:\n{instruction}\n\n{ctx}### Response:\n{response}"


# Question and answer

@register_format("qa")
def render_qa(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with ``question`` and ``answer`` and, optionally, a ``context`` (a passage to answer from: SQuAD, TriviaQA and the like).

    options:
      * question_field / answer_field / context_field: str - other column names
      * use_chat_template: bool - render as a 2-turn chat instead of "Question: ... Answer: ...". Default False.
    """
    question = _col(row, options, "question", "question")
    answer   = _col(row, options, "answer", "answer")
    context  = _col(row, options, "context", "context")
    if not question and not answer:
        return ""
    if options.get("use_chat_template", False):
        user = question if not context else f"{context}\n\n{question}"
        return _apply_chat_template_safe(tokenizer, [{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
    ctx = f"Context: {context}\n\n" if context else ""
    return f"{ctx}Question: {question}\nAnswer: {answer}"


# Scored responses (reward data)

@register_format("scored_response")
def render_scored_response(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Rows with a ``prompt``, a ``response`` and a ``score`` (rated or reward-model-scored answers). Only responses that score at least
    ``min_score`` are trained on: the others render to nothing and are dropped. This is fine-tuning on the good answers (rejection
    sampling, "best of n"), not training a reward model and not reinforcement learning.

    options:
      * prompt_field / response_field / score_field: str - other column names (defaults prompt, response, score)
      * min_score: number - keep rows with score >= min_score (without it every row is kept: nothing is filtered)
      * use_chat_template: bool - render as a 2-turn chat. Default False (prompt, a newline, response).
    """
    prompt   = _col(row, options, "prompt", "prompt")
    response = _col(row, options, "response", "response")
    if "min_score" in options:
        try:
            if float(row.get(options.get("score_field", "score"))) < float(options["min_score"]):
                return ""
        except (TypeError, ValueError):
            return ""                                    # no usable score: not trusted
    if not prompt and not response:
        return ""
    if options.get("use_chat_template", False):
        return _apply_chat_template_safe(tokenizer, [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
    return prompt + "\n" + response


# Tool and function calling

@register_format("tool_calling")
def render_tool_calling(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Chat rows in which the assistant may call tools: ``messages`` as in chatml, where an assistant message can carry ``tool_calls``
    (``[{"type": "function", "function": {"name": ..., "arguments": {...}}}]``, arguments as an object or as JSON text) and a tool's
    result is a message with role ``tool``. The tools themselves (JSON schemas) come from a ``tools`` column, as a list or as JSON text.

    The model's own chat template lays tools and calls out the way that model expects, so this works for models whose template supports
    tools (Qwen3, Llama 3.1+, Mistral, Hermes and others). Without a chat template, calls are written as <tool_call> JSON blocks.
    Store ``tools`` as a JSON string when the schemas differ from row to row: they cannot share one table column otherwise.

    options:
      * messages_field / tools_field: str - other column names (defaults messages, tools)
    """
    messages = [m for m in (_plain(_json_or_self(x)) for x in _to_list(row.get(options.get("messages_field", "messages"), []))) if m]
    if not messages:
        return ""
    for m in messages:
        calls = m.get("tool_calls")
        if isinstance(calls, str):
            m["tool_calls"] = calls = _json_or_self(calls)
        for c in calls or []:
            f = c.get("function") if isinstance(c, dict) else None
            if isinstance(f, dict) and isinstance(f.get("arguments"), str):
                f["arguments"] = _json_or_self(f["arguments"]) if f["arguments"].lstrip()[:1] == "{" else f["arguments"]
        m.setdefault("content", "")
    tools = _plain(_json_or_self(row.get(options.get("tools_field", "tools"))))
    if not (hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)):
        import json
        out = []
        for m in messages:
            body = str(m.get("content") or "")
            for c in m.get("tool_calls") or []:
                body += "\n<tool_call>\n" + json.dumps(c.get("function", c), ensure_ascii=False) + "\n</tool_call>"
            out.append({"role": m.get("role", "user"), "content": body})
        return _manual_chatml(out)
    return _template(tokenizer, messages, **({"tools": tools} if tools else {}))


# Reasoning (thinking) traces

_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


@register_format("reasoning")
def render_reasoning(row: dict, tokenizer: Any, options: dict) -> str:
    """
    Reasoning data: the model learns to think first and answer after. Two layouts are read.

    * ``messages`` as in chatml, where an assistant message has a ``reasoning_content`` (or ``reasoning`` or ``thinking``) field next to its
      ``content``;
    * one row per example, with a ``prompt``, a ``reasoning`` and an ``answer`` (a ``response`` column also works as the answer).

    The reasoning is written in front of the answer between ``<think>`` and ``</think>``, the layout Qwen3, DeepSeek-R1 and most
    reasoning models use, and the text goes through the model's chat template like any chat.

    options:
      * messages_field: str - the chat column (default messages)
      * prompt_field / reasoning_field / answer_field: str - the column layout's names (defaults prompt, reasoning, answer)
      * think_open / think_close: str - the markers (defaults "<think>\\n" and "\\n</think>\\n\\n")
    """
    open_, close = options.get("think_open", "<think>\n"), options.get("think_close", "\n</think>\n\n")
    messages = [m for m in (_plain(_json_or_self(x)) for x in _to_list(row.get(options.get("messages_field", "messages"), []))) if m]
    if not messages:
        prompt    = _col(row, options, "prompt", "prompt")
        reasoning = _col(row, options, "reasoning", "reasoning")
        answer    = _col(row, options, "answer", "answer") or str(row.get("response", "") or "")
        if not prompt and not answer:
            return ""
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer, "reasoning_content": reasoning}]
    out = []
    for m in messages:
        content = str(m.get("content") or "")
        thought = next((str(m[k]) for k in _REASONING_KEYS if m.get(k)), "")
        m = {k: v for k, v in m.items() if k not in _REASONING_KEYS}
        if thought and m.get("role") == "assistant" and "</think>" not in content:
            content = f"{open_}{thought.strip()}{close}{content}"
        out.append({**m, "content": content})
    return _apply_chat_template_safe(tokenizer, out)
