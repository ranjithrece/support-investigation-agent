"""Ollama chat model factory + structured-output helper with repair retry."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from support_agent.config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_override: BaseChatModel | None = None  # tests inject a fake model here


def set_llm_override(model: BaseChatModel | None) -> None:
    global _override
    _override = model


def get_llm(model: str | None = None, num_ctx: int = 8192) -> BaseChatModel:
    if _override is not None:
        return _override
    s = get_settings()
    return ChatOllama(
        model=model or s.chat_model,
        base_url=s.ollama_base_url,
        temperature=0,
        num_ctx=num_ctx,
        reasoning=False,  # qwen3 "thinking" is ~7x slower and unnecessary for these steps
        client_kwargs={"timeout": s.llm_timeout_s},
    )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()


def _extract_json(text: str) -> Any:
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start >= 0:
            text = text[start:]
    return json.loads(text)


def structured(
    schema: type[T], system: str, user: str, model: str | None = None, retries: int = 2
) -> T:
    """Call the LLM and return a validated `schema` instance (JSON-schema constrained)."""
    llm = get_llm(model)
    msgs: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
    last_err = ""
    for attempt in range(retries + 1):
        try:
            if _override is not None:
                raw = llm.invoke(msgs).content
                return schema.model_validate(_extract_json(str(raw)))
            bound = llm.with_structured_output(schema, method="json_schema")  # type: ignore[attr-defined]
            out = bound.invoke(msgs)
            if isinstance(out, schema):
                return out
            return schema.model_validate(out)
        except (ValidationError, ValueError, json.JSONDecodeError, TypeError) as e:
            last_err = str(e)[:300]
            log.warning("structured output attempt %d failed: %s", attempt + 1, last_err)
            msgs = msgs[:2] + [
                HumanMessage(
                    content=f"Your previous output was invalid ({last_err}). "
                    f"Reply again with ONLY valid JSON matching the schema."
                )
            ]
    raise RuntimeError(f"LLM failed to produce valid {schema.__name__}: {last_err}")
