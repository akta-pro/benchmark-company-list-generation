"""HasData Google AI-mode scrape client.

One call = one question in, one Google AI Mode answer out. src.factpack uses it
to build each company's fact pack for the relevance-validation judge
(src.validate). Each call's full request and response are written, with the API
key redacted, to raw/<run_id>/hasdata/<request_id>.json.
"""

import time
import uuid
from datetime import datetime, timezone

import httpx2

from src.config import (
    HASDATA_API_KEY,
    HASDATA_API_URL,
    HASDATA_CALL_COST_USD,
    HASDATA_CREDITS_PER_CALL,
    HASDATA_READ_TIMEOUT,
)
from src.logger import logger
from src.results import HASDATA_RAW_SUBDIR, record_error, redact_headers, write_raw_result
from src.retry import retry_transient

RETRYABLE_STATUS = {400, 429, 500, 502, 503, 504}


def build_hasdata_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(timeout=httpx2.Timeout(10.0, read=HASDATA_READ_TIMEOUT))


def _list_lines(items: list[dict], depth: int = 0) -> list[str]:
    """One "- item" line per list item; nested lists are indented."""
    lines = []
    for item in items:
        if item.get("snippet"):
            lines.append(f"{'  ' * depth}- {item['snippet'].strip()}")
        lines += _list_lines(item.get("list") or [], depth + 1)
    return lines


def answer_text(body: dict) -> str:
    """All of the AI-mode answer's text, in order, one block per line:
    paragraphs, headings and other snippet blocks as-is, list items as
    "- item" lines (nested ones indented), table rows with cells joined by
    " | ". Sources (references, links) are left out. Empty string if the
    response has no text."""
    lines = []
    for block in body.get("textBlocks") or []:
        if block.get("type") == "list":
            lines += _list_lines(block.get("list") or [])
        elif block.get("type") == "table":
            lines += [" | ".join(str(cell) for cell in row) for row in block.get("rows") or []]
        elif block.get("snippet"):
            lines.append(block["snippet"].strip())
    return "\n".join(line for line in lines if line.strip())


async def hasdata_get(http: httpx2.AsyncClient, params: dict, headers: dict) -> httpx2.Response:
    """One GET. A retryable status is raised as HTTPStatusError so
    retry_transient retries it; the error carries the response."""
    response = await http.get(HASDATA_API_URL, params=params, headers=headers)
    if response.status_code in RETRYABLE_STATUS:
        raise httpx2.HTTPStatusError(
            f"retryable status {response.status_code}", request=response.request, response=response
        )
    return response


async def call_hasdata(http: httpx2.AsyncClient, question: str, run_id: str, meta: dict) -> tuple[str, dict]:
    """Ask HasData's Google AI-mode one question. Returns (answer, raw_record).
    Raises only if HASDATA_API_KEY is unset; every other failure is captured
    in the returned/dumped record (status="error") and the answer is "".
    `meta` is merged into the raw record as-is (e.g. query_id, company,
    question_index)."""
    if not HASDATA_API_KEY:
        raise RuntimeError("HASDATA_API_KEY is not set")

    request_id = str(uuid.uuid4())
    tag = f"[hasdata {meta.get('query_id')} q{meta.get('question_index')}]"
    params = {"q": question}
    headers = {"x-api-key": HASDATA_API_KEY, "Content-Type": "application/json"}

    record = {
        "request_id": request_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **meta,
        "question": question,
        "request": {
            "method": "GET",
            "url": HASDATA_API_URL,
            "params": params,
            "headers": redact_headers(headers),
            "body": None,
        },
        "response": None,
        "status": None,
        "error": None,
        "error_type": None,
        "failed_attempts": [],  # every failed try, incl. retried ones (src.retry)
        "latency_ms": None,
        # Set once the outcome is known: HasData bills successful scrapes only,
        # so a failed call (incl. one that exhausted its retries) costs nothing.
        "credits": 0,
        "cost_usd": 0.0,
        "answer": None,
    }

    start = time.monotonic()
    answer = ""
    try:
        logger.info("%s asking: %s", tag, question)
        try:
            response = await retry_transient(
                lambda: hasdata_get(http, params, headers),
                (httpx2.TransportError, httpx2.HTTPStatusError),
                tag,
                record["failed_attempts"],
            )
        except httpx2.HTTPStatusError as e:
            response = e.response  # still a retryable status after the last attempt; recorded below
        try:
            body = response.json()
        except ValueError:
            body = response.text  # non-JSON reply (e.g. an HTML error page), kept as-is
        record["response"] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "cookies": dict(response.cookies),
            "body": body,
        }
        response.raise_for_status()
        if not isinstance(body, dict):
            raise ValueError("HasData response body is not a JSON object")
        answer = answer_text(body)
        record["answer"] = answer
        record["status"] = "completed"
        record["credits"] = HASDATA_CREDITS_PER_CALL
        record["cost_usd"] = HASDATA_CALL_COST_USD
    except Exception as e:
        logger.exception("%s failed: %s", tag, e)
        record_error(record, e)

    record["latency_ms"] = round((time.monotonic() - start) * 1000)
    write_raw_result(record, HASDATA_RAW_SUBDIR)
    return answer, record
