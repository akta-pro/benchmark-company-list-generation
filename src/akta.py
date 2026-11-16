"""akta.pro MCP: a shared httpx client; each call opens its own short-lived session.

Tool-call failures fall into three groups (see src.openrouter.run_akta_tool_call):
  - akta.pro returns an error result (result.is_error): a normal result for the model.
  - The call is invalid (MCPError with an INVALID_CALL_CODES code): the model is told.
  - akta.pro is unavailable (AktaUnavailableError): retried; if it persists, the
    request fails.
"""

import json
import re

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, METHOD_NOT_FOUND

from src.config import AKTA_API_KEY, AKTA_MCP_URL, AKTA_READ_TIMEOUT
from src.logger import logger
from src.retry import retry_transient


def build_akta_http_client() -> httpx2.AsyncClient:
    headers = {"x-api-key": AKTA_API_KEY} if AKTA_API_KEY else {}
    return httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(10.0, read=AKTA_READ_TIMEOUT),
    )


async def list_akta_tools(http: httpx2.AsyncClient) -> list[dict]:
    """Fetch akta's tools as OpenRouter function-tool schemas."""
    if not AKTA_API_KEY:
        raise RuntimeError("AKTA_API_KEY is not set")
    transport = streamable_http_client(AKTA_MCP_URL, http_client=http)
    async with Client(transport, mode="legacy") as client:  # akta.pro needs legacy mode
        listed = await client.list_tools()
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in listed.tools
    ]
    logger.info("akta: %d tools listed", len(tools))
    return tools


class AktaUnavailableError(Exception):
    """A temporary akta.pro failure worth retrying: connection problem, timeout,
    server error or a cut-off response."""


# MCP errors that mean the model's call itself was invalid (unknown tool, bad
# arguments). Retrying can't fix these; every other MCP error is on akta.pro's side.
INVALID_CALL_CODES = (INVALID_PARAMS, METHOD_NOT_FOUND)


def _leaf_errors(error: BaseException) -> list[BaseException]:
    """The underlying errors: the MCP client can wrap them in an ExceptionGroup."""
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _leaf_errors(sub)]
    return [error]


def _is_akta_unavailable(error: BaseException) -> bool:
    def transient(leaf: BaseException) -> bool:
        if isinstance(leaf, MCPError):
            return leaf.code not in INVALID_CALL_CODES
        return isinstance(leaf, httpx2.TransportError)

    return all(transient(leaf) for leaf in _leaf_errors(error))


async def call_akta_tool(
    http: httpx2.AsyncClient, name: str, arguments: dict, tag: str, failed_attempts: list[dict]
) -> dict:
    """Call one akta.pro tool, retrying temporary failures (retry_transient; every
    failed attempt is appended to `failed_attempts`). Raises MCPError when akta.pro
    rejects the call as invalid, AktaUnavailableError when it is still failing
    after the last attempt."""

    async def attempt():
        transport = streamable_http_client(AKTA_MCP_URL, http_client=http)
        try:
            async with Client(transport, mode="legacy") as client:
                return await client.call_tool(name, arguments)
        except Exception as e:
            if _is_akta_unavailable(e):
                details = "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in _leaf_errors(e))
                raise AktaUnavailableError(details) from e
            invalid = [leaf for leaf in _leaf_errors(e) if isinstance(leaf, MCPError)]
            if invalid:
                raise invalid[0] from e
            raise

    result = await retry_transient(attempt, (AktaUnavailableError,), tag, failed_attempts)
    if result.is_error:
        logger.warning("%s akta tool %s returned an error", tag, name)
    return result.model_dump(mode="json")


# company_data (and other markdown-formatted tools) report credits in a
# trailing text footer rather than a JSON field, e.g.:
#   "_Sections included: location · Credits consumed: 0.5_"
_CREDITS_FOOTER_RE = re.compile(r"Credits consumed:\s*([\d.]+)")


def akta_credits(result: dict) -> float:
    """Credits billed for one akta.pro tool call. Most tools (e.g.
    generate_company_list) report a `credits_consumed` JSON field; markdown
    tools (e.g. company_data) instead print it in a text footer. Returns 0.0
    when absent or unparseable (free tools legitimately report 0.0)."""
    sc = result.get("structured_content")
    if isinstance(sc, dict) and isinstance(sc.get("credits_consumed"), (int, float)):
        return float(sc["credits_consumed"])
    for block in result.get("content") or []:
        if block.get("type") != "text":
            continue
        text = block.get("text", "")
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("credits_consumed"), (int, float)):
            return float(data["credits_consumed"])
        match = _CREDITS_FOOTER_RE.search(text)
        if match:
            return float(match.group(1))
    return 0.0


def tool_result_text(result: dict) -> str:
    """Flatten an MCP tool result's content blocks to the text fed back to the
    model."""
    parts = [
        block.get("text", "")
        for block in (result.get("content") or [])
        if block.get("type") == "text"
    ]
    return "\n".join(parts) if parts else json.dumps(result)
