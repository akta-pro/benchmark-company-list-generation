"""OpenRouter chat calls: benchmark requests, the akta.pro tool loop and structured
output, shared with the relevance judge (src.validate)."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import json_repair
from mcp.shared.exceptions import MCPError
from openrouter import OpenRouter
from openrouter import errors as or_errors
from openrouter.utils import BackoffStrategy, RetryConfig
from pydantic import BaseModel, ValidationError

from src.akta import akta_credits, call_akta_tool, tool_result_text
from src.config import (
    AKTA_CREDIT_USD,
    MAX_REPEATED_TOOL_CALLS,
    MAX_TOOL_CALLS,
    OPENROUTER_SDK_RETRY_MAX_ELAPSED,
    OPENROUTER_TIMEOUT,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    SYSTEM_PROMPT,
    Provider,
    Query,
)
from src.logger import logger
from src.results import (
    FETCH_RAW_SUBDIR,
    benchmark_metrics,
    final_text,
    finalize_raw_record,
    new_raw_record,
    normalize,
    record_error,
    write_normalized_result,
    write_raw_result,
)
from src.retry import retry_transient
from src.schemas import SCHEMAS
from src.usage import akta_usage_summary, finish_reason

# Transient failures worth retrying; everything else (4xx, auth, validation)
# is permanent and propagates immediately.
RETRYABLE_ERRORS = (
    or_errors.TooManyRequestsResponseError,
    or_errors.InternalServerResponseError,
    or_errors.BadGatewayResponseError,
    or_errors.ServiceUnavailableResponseError,
    or_errors.ProviderOverloadedResponseError,
    or_errors.RequestTimeoutResponseError,
    or_errors.EdgeNetworkTimeoutResponseError,
    or_errors.NoResponseError,
    httpx.TransportError,
)


def request_params(kwargs: dict) -> dict:
    """Everything one `chat.send` carried, compact enough to keep per call.
    A request's messages only ever grow, so they're recorded as a count (call N
    sent request.messages[:messages_count]); tools are recorded by name, with
    the full definitions stored once as request.tools."""
    tools = kwargs.get("tools")
    return {
        **{k: v for k, v in kwargs.items() if k not in ("messages", "tools")},
        "messages_count": len(kwargs["messages"]),
        "tools": [t["function"]["name"] for t in tools] if tools else None,
    }


async def chat_send(client: OpenRouter, tag: str, request_log: list[dict], **kwargs):
    """`client.chat.send_async` with retry on transient errors (retry_transient).
    Appends the call's parameters to `request_log` before sending, so a call
    that ultimately fails is still on record, with every failed attempt under
    its `failed_attempts`."""
    call_log = {**request_params(kwargs), "failed_attempts": []}
    request_log.append(call_log)
    return await retry_transient(
        lambda: client.chat.send_async(**kwargs), RETRYABLE_ERRORS, tag, call_log["failed_attempts"]
    )


def build_openrouter_client() -> OpenRouter:
    sdk_retry = RetryConfig(
        "backoff",
        BackoffStrategy(
            initial_interval=int(RETRY_BASE_DELAY * 1000),
            max_interval=int(RETRY_MAX_DELAY * 1000),
            exponent=2,
            max_elapsed_time=int(OPENROUTER_SDK_RETRY_MAX_ELAPSED * 1000),
        ),
        retry_connection_errors=True,
    )
    return OpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout_ms=int(OPENROUTER_TIMEOUT * 1000),
        retry_config=sdk_retry,
    )


def _strict_objects(node):
    """Recursively set additionalProperties:false on every object schema.
    OpenAI strict mode rejects the schema unless every nested object (incl.
    $defs) declares it; Anthropic tolerates it missing."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for value in node.values():
            _strict_objects(value)
    elif isinstance(node, list):
        for item in node:
            _strict_objects(item)
    return node


def json_schema_format(name: str, json_schema: dict) -> dict:
    """Build an OpenRouter `response_format` for strict JSON Schema structured
    output from a JSON Schema dict."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": _strict_objects(json_schema)},
    }


def web_plugins(query: Query) -> list[dict] | None:
    """OpenRouter `web` plugin config when the query opts into web search, else
    None. Uses the provider's native search engine, capped at `max_web_searches`
    invocations per model call. Searches don't count against MAX_TOOL_CALLS."""
    if not query.use_web_search:
        return None
    return [{"id": "web", "engine": "native", "max_uses": query.max_web_searches}]


def parse_structured(content: str, schema: type[BaseModel]) -> tuple[dict, bool]:
    """Validate `content` against `schema`. If `content` isn't valid JSON, retry
    once through json_repair. Returns (validated dict, json_repaired flag).

    Only syntax is repaired: JSON that parses but violates the schema (e.g. a
    missing field) still raises, so genuine non-conformance stays a failure."""
    try:
        return schema.model_validate_json(content).model_dump(), False
    except ValidationError:
        pass
    repaired = json_repair.loads(content)
    if not isinstance(repaired, dict) or not repaired:
        raise ValueError("json_repair could not recover a JSON object from the model output")
    return schema.model_validate(repaired).model_dump(), True


def structured_output_params(schema_name: str, json_schema: dict) -> dict:
    """chat.send parameters that make the model's answer follow `json_schema`."""
    return {
        "response_format": json_schema_format(schema_name, json_schema),
        # only route to a provider that actually enforces response_format
        "provider": {"require_parameters": True},
    }


def parse_structured_response(
    tag: str, record: dict, response: dict, output_model: type[BaseModel]
) -> dict:
    """Check and parse a structured-output response (a model_dump): refuse a
    cut-off reply, keep the raw text, and parse it with `output_model`. Fills
    the record's raw_response / json_repaired / structured_output and returns
    the parsed dict; raises on any failure."""
    if finish_reason(response) == "length":
        raise ValueError(
            "structured-output response truncated (finish_reason=length); "
            "refusing to trust a possibly-repaired partial object"
        )
    content = final_text([response]) or ""
    record["raw_response"] = content
    parsed, record["json_repaired"] = parse_structured(content, output_model)
    record["structured_output"] = parsed
    if record["json_repaired"]:
        logger.warning("%s structured output needed json_repair", tag)
    return parsed


async def request_structured_output(
    client: OpenRouter,
    tag: str,
    record: dict,
    model_responses: list[dict],
    schema_name: str,
    json_schema: dict,
    output_model: type[BaseModel],
    **kwargs,
) -> dict:
    """One structured-output call (web_search variant, relevance judge): send
    with `json_schema` enforced, then parse_structured_response. Returns the
    parsed dict; raises on any failure. `kwargs` go to chat.send (model,
    messages...)."""
    response = await chat_send(
        client, tag, record["request"]["calls"],
        **structured_output_params(schema_name, json_schema),
        **kwargs,
    )
    model_responses.append(response.model_dump(mode="json"))
    return parse_structured_response(tag, record, model_responses[-1], output_model)


def save_raw_request(
    record: dict, query: Query, messages: list[dict], model_responses: list[dict],
    tool_calls: list[dict], start: float,
) -> None:
    """Finalize a benchmark request's raw record and write it to raw/<run_id>/fetch/."""
    finalize_raw_record(record, model_responses, round((time.monotonic() - start) * 1000))
    # The full message list, including tool rounds; each call in request.calls
    # sent a prefix of it.
    record["request"]["messages"] = messages
    record["tool_calls"] = tool_calls
    record["model_responses"] = model_responses
    if query.use_akta:
        record["akta_usage"] = akta_usage_summary(tool_calls)
    write_raw_result(record, FETCH_RAW_SUBDIR)


async def send_chat(
    client: OpenRouter,
    provider: Provider,
    query: Query,
    run_dir: Path,
    akta_http=None,
    akta_tools: list[dict] | None = None,
) -> dict:
    """Run one provider x query-variant: the akta.pro tool loop or a single call,
    then write the raw record and normalized result. A failure is recorded on
    the result, not raised; an interrupted run writes only the raw record.
    Returns the raw record."""
    request_id = str(uuid.uuid4())
    schema = SCHEMAS[query.structured_output] if query.structured_output else None
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.text},
    ]
    record = new_raw_record(
        request_id, run_dir.name, provider.model_dump(), query.model_dump(), messages
    )
    tag = f"[{provider.id} x {query.id}/{query.variant}]"
    if query.use_akta:
        record["request"]["tools"] = akta_tools
    request_log = record["request"]["calls"]

    model_responses: list[dict] = []
    tool_calls: list[dict] = []
    stop_reason = None
    plugins = web_plugins(query)
    start = time.monotonic()

    try:
        if query.use_akta:
            # The JSON schema rides along on every tool round, so the model's
            # final answer is already the structured one: no extra call.
            structured = (
                structured_output_params(query.structured_output, schema.model_json_schema())
                if schema is not None
                else None
            )
            stop_reason = await tool_loop(
                client, provider, akta_http, akta_tools,
                messages, request_log, model_responses, tool_calls, tag, plugins, structured,
            )
            if not tool_calls:
                logger.warning("%s akta request finished without using any Akta tools", tag)
            if schema is not None:
                parse_structured_response(tag, record, model_responses[-1], schema)
        elif schema is not None:
            logger.info("%s structured-output call (schema=%s)", tag, query.structured_output)
            await request_structured_output(
                client, tag, record, model_responses,
                query.structured_output, schema.model_json_schema(), schema,
                model=provider.model,
                messages=messages,
                plugins=plugins,
            )
        else:
            logger.info("%s openrouter call (model=%s)", tag, provider.model)
            response = await chat_send(
                client, tag, request_log,
                model=provider.model, messages=messages, plugins=plugins,
            )
            model_responses.append(response.model_dump(mode="json"))

        record["status"] = stop_reason or "completed"
    except asyncio.CancelledError:
        # The run was stopped (e.g. Ctrl-C): keep the raw record of what this
        # request did so far, but write no normalized result, so a partial
        # request never reaches the pools.
        logger.warning("%s interrupted; saving the raw record so far", tag)
        record["status"] = "interrupted"
        save_raw_request(record, query, messages, model_responses, tool_calls, start)
        raise
    except Exception as e:
        logger.exception("%s failed: %s", tag, e)
        record_error(record, e)

    save_raw_request(record, query, messages, model_responses, tool_calls, start)

    output = record["structured_output"] if schema is not None else final_text(model_responses)
    metrics = benchmark_metrics(record)
    normalized = normalize(record, output, metrics)
    norm_path = write_normalized_result(normalized, run_dir)

    logger.info("%s %s -> %s", tag, record["status"], norm_path)
    return record


async def run_akta_tool_call(akta_http, call, tag: str, tool_calls: list[dict]) -> str:
    """Run one akta.pro tool call the model asked for and record it in `tool_calls`,
    whether it succeeds or not (failed calls count as tool calls). Returns the
    text to send back to the model.

    - Arguments that aren't a JSON object, or a call akta.pro rejects as invalid
      (unknown tool, bad arguments): not retried; the model is told so it can
      correct itself.
    - akta.pro unavailable even after retries (AktaUnavailableError): the call is
      recorded and the error propagates, failing the request.
    - An error result from the tool itself reaches the model like any result."""
    entry = {
        "id": call.id,
        "name": call.function.name,
        "arguments": call.function.arguments,  # the raw text until it parses
        "result": None,
        "credits": 0.0,
        "cost_usd": 0.0,
        "status": "completed",
        "error": None,
        "error_type": None,
        "failed_attempts": [],  # every failed try, incl. retried ones (src.retry)
    }
    tool_calls.append(entry)

    try:
        arguments = json.loads(call.function.arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError(f"expected a JSON object, got {type(arguments).__name__}")
    except ValueError as e:
        logger.warning("%s akta tool %s: invalid arguments: %s", tag, call.function.name, e)
        record_error(entry, e)
        return f"Error: arguments are not a valid JSON object: {e}"
    entry["arguments"] = arguments

    logger.info("%s akta tool call: %s(%s)", tag, call.function.name, arguments)
    try:
        result = await call_akta_tool(akta_http, call.function.name, arguments, tag, entry["failed_attempts"])
    except MCPError as e:
        logger.warning("%s akta tool %s: call rejected: %s", tag, call.function.name, e)
        record_error(entry, e)
        return f"Error: Akta rejected the call: {e}"
    except asyncio.CancelledError:
        entry["status"] = "interrupted"
        raise
    except Exception as e:
        record_error(entry, e)
        raise

    entry["result"] = result
    entry["credits"] = akta_credits(result)
    entry["cost_usd"] = entry["credits"] * AKTA_CREDIT_USD
    return tool_result_text(result)


def _call_signature(entry: dict) -> tuple[str, str]:
    """(tool name, arguments) of a recorded tool call, for spotting repeats."""
    args = entry["arguments"]
    return entry["name"], json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)


async def tool_loop(
    client, provider, akta_http, akta_tools, messages, request_log, model_responses,
    tool_calls, tag, plugins=None, structured: dict | None = None,
) -> str | None:
    """Send the prompt with akta.pro tools, run tool calls, feed results back, repeat
    until the model answers. Appends to messages, request_log, model_responses
    and tool_calls in place so partial progress survives an error. The final
    answer is model_responses[-1].

    Tools are withdrawn, so the next round must be the final answer, when
    MAX_TOOL_CALLS is reached or the model makes the same tool call
    MAX_REPEATED_TOOL_CALLS times in a row (loop guard). If the model still
    requests tools after that, the loop ends and the request fails on parsing.

    `plugins` (the OpenRouter `web` plugin, or None) is forwarded on every round;
    web search is provider-side and does not count against MAX_TOOL_CALLS.
    `structured` (structured_output_params, or None) is also sent on every
    round, so the final answer comes back in the query's JSON schema.

    Returns why tool use was cut short ("max_tool_calls" or
    "repeated_tool_call"), or None if the model finished on its own."""
    stop_reason = None
    repeats, last_call = 0, None
    for round_num in range(1, MAX_TOOL_CALLS + 3):
        if stop_reason is None and len(tool_calls) >= MAX_TOOL_CALLS:
            stop_reason = "max_tool_calls"
        offer_tools = stop_reason is None
        logger.info(
            "%s openrouter call (model=%s, round=%d, tools=%s)",
            tag, provider.model, round_num, "akta" if offer_tools else "none",
        )
        response = await chat_send(
            client, tag, request_log,
            model=provider.model,
            messages=messages,
            tools=akta_tools if offer_tools else None,
            plugins=plugins,
            **(structured or {}),
        )
        model_responses.append(response.model_dump(mode="json"))
        message = response.choices[0].message

        if not message.tool_calls:
            return stop_reason
        if not offer_tools:
            logger.warning("%s still requesting tools after they were withdrawn (%s)", tag, stop_reason)
            return stop_reason

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            if stop_reason is None and len(tool_calls) >= MAX_TOOL_CALLS:
                stop_reason = "max_tool_calls"
            if stop_reason is not None:
                logger.warning("%s tool use stopped (%s), skipping %s", tag, stop_reason, call.function.name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": f"Error: no more tool calls allowed ({stop_reason}). Give your final answer now.",
                    }
                )
                continue

            content = await run_akta_tool_call(akta_http, call, tag, tool_calls)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

            this_call = _call_signature(tool_calls[-1])
            repeats = repeats + 1 if this_call == last_call else 1
            last_call = this_call
            if repeats >= MAX_REPEATED_TOOL_CALLS:
                stop_reason = "repeated_tool_call"
                logger.warning(
                    "%s same tool call %d times in a row: %s(%s); withdrawing tools for the final answer",
                    tag, repeats, *this_call,
                )

    # Round budget expired with the model still requesting tools.
    return stop_reason or "max_tool_calls"
