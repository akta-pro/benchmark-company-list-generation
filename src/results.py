"""The two benchmark artifacts and the normalization between them.

raw result   = full execution dump, for debugging/reproducibility (raw/<run_id>/<stage>/)
normalized result = compact, stable contract for judging/aggregation (results/<run_id>/)
"""

import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import simplejson

from src.usage import finish_reason, llm_usage_summary

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
# Stages under raw/<run_id>/: benchmark (list-generation) calls, HasData
# fact-pack calls, validation judge calls.
FETCH_RAW_SUBDIR = "fetch"
HASDATA_RAW_SUBDIR = "hasdata"
VALIDATION_RAW_SUBDIR = "validation"

# raw/<run_id>/costs.jsonl: one line per HasData or validation raw record
# (stage, query_id, request_id, status, cost_usd, latency_ms), appended as each
# record is written. Stage cost metrics are summed from it. Benchmark (fetch)
# costs come from the per-request results.
COST_LEDGER_NAME = "costs.jsonl"
COST_LEDGER_STAGES = (HASDATA_RAW_SUBDIR, VALIDATION_RAW_SUBDIR)

REDACTED = "<REDACTED>"
# Lowercase, compared case-insensitively. Headers that carry credentials.
SENSITIVE_HEADERS = frozenset({
    "x-api-key", "api-key", "authorization", "proxy-authorization", "cookie", "set-cookie",
})


def redact_headers(headers: dict) -> dict:
    """Request headers as stored in a raw record: credentials are never written."""
    return {k: (REDACTED if k.lower() in SENSITIVE_HEADERS else v) for k, v in headers.items()}


def new_raw_record(
    request_id: str, run_id: str, provider: dict, query_config: dict, messages: list[dict]
) -> dict:
    """The empty execution record shared by every OpenRouter call path (main
    benchmark and validation judge). Callers mutate status/error/output fields
    as the call proceeds, then pass it through `finalize_raw_record`.

    `request` is what was actually sent to the model: `messages` (the full
    message list) plus `calls` (per-call parameters). `query_config` is the
    configuration the execution ran under, never sent: the benchmark path stores
    the whole queries.yaml entry for that query-variant (text, tool flags,
    fact_questions, validation files); the validation judge stores {id, text:
    <un-filled judge prompt template>, stage}."""
    return {
        "request_id": request_id,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "query_config": query_config,
        "status": None,  # completed | max_tool_calls | repeated_tool_call | error | interrupted
        "finish_reason": None,  # the final model response's stop reason
        "error": None,
        "error_type": None,
        "structured_output": None,  # parsed schema instance, when the call asks for one
        "raw_response": None,  # the structured call's raw content, kept for debugging
        "json_repaired": False,  # True if structured output needed json_repair to parse
        "latency_ms": None,
        "request": {
            "messages": [dict(m) for m in messages],
            "tools": None,  # full function-tool definitions offered, if any
            "calls": [],  # per-call parameters, in order (see src.openrouter.request_params)
        },
    }


def record_error(record: dict, error: Exception) -> None:
    """Mark a raw record as failed with `error`."""
    record["status"] = "error"
    record["error"] = str(error)
    record["error_type"] = type(error).__name__


def finalize_raw_record(record: dict, model_responses: list[dict], latency_ms: int) -> None:
    """Fill the response-derived fields common to every call path."""
    record["latency_ms"] = latency_ms
    if model_responses:
        record["finish_reason"] = finish_reason(model_responses[-1])
        record["total_usage"] = llm_usage_summary(model_responses)
        record["response"] = model_responses[-1]


def slug(value: str) -> str:
    """Filesystem-safe form of an id (provider ids like 'anthropic/claude-5'
    carry slashes)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


class PlainDecimal(Decimal):
    """Decimal that stringifies in plain decimal notation."""

    def __str__(self) -> str:
        return format(Decimal(self), "f")


def plain(obj):
    """Recursively convert floats to PlainDecimal for plain-decimal JSON output."""
    if isinstance(obj, float):
        return PlainDecimal(str(obj))
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain(v) for v in obj]
    return obj


def read_json(path: Path):
    return simplejson.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    """Write `data` as indented JSON with floats in plain decimal notation."""
    path.write_text(simplejson.dumps(plain(data), indent=2), encoding="utf-8")


def write_raw_result(record: dict, stage: str) -> Path:
    """Write the full execution dump to raw/<run_id>/<stage>/<request_id>.json
    (stage: fetch | hasdata | validation)."""
    out = RAW_DIR / record["run_id"] / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{record['request_id']}.json"
    write_json(path, record)
    if stage in COST_LEDGER_STAGES:
        append_cost_entry(record, stage)
    return path


def cost_entry(record: dict, stage: str) -> dict:
    """The cost ledger line for one HasData or validation raw record."""
    if stage == HASDATA_RAW_SUBDIR:
        query_id = record["query_id"]
        # HasData bills successful scrapes only, so a failed call is recorded as 0.0.
        cost = record["cost_usd"]
    else:
        query_id = record["query_config"]["id"]
        cost = (record.get("total_usage") or {}).get("cost")
    return {
        "stage": stage,
        "query_id": query_id,
        "request_id": record["request_id"],
        "status": record["status"],
        "cost_usd": cost,
        "latency_ms": record["latency_ms"],
    }


def append_cost_entry(record: dict, stage: str) -> None:
    """Add one record to its run's cost ledger."""
    path = RAW_DIR / record["run_id"] / COST_LEDGER_NAME
    with path.open("a", encoding="utf-8") as f:
        f.write(simplejson.dumps(cost_entry(record, stage)) + "\n")


def drop_cost_entries(run_id: str, query_id: str, stages: tuple[str, ...]) -> int:
    """Remove this query's entries for `stages` from the run's cost ledger, so a
    re-run records only what it spends now instead of adding to superseded
    passes. Returns how many entries were dropped. The raw records themselves
    are kept: they are the audit trail."""
    path = RAW_DIR / run_id / COST_LEDGER_NAME
    if not path.exists():
        return 0
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        entry = simplejson.loads(line)
        if entry.get("query_id") == query_id and entry.get("stage") in stages:
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return dropped


def read_cost_ledger(run_id: str) -> list[dict]:
    """Every entry in raw/<run_id>/costs.jsonl; [] before any has been recorded."""
    path = RAW_DIR / run_id / COST_LEDGER_NAME
    if not path.exists():
        return []
    return [simplejson.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_result_path(run_dir: Path, provider_id: str, query_id: str, variant: str) -> Path:
    """results/<run_id>/<provider>__<query>__<variant>.json for one execution."""
    return run_dir / f"{slug(provider_id)}__{slug(query_id)}__{slug(variant)}.json"


def write_normalized_result(normalized: dict, run_dir: Path) -> Path:
    """Write one execution's normalized result under results/<run_id>/."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = normalized_result_path(
        run_dir, normalized["provider"]["id"], normalized["query"]["id"], normalized["query"]["variant"]
    )
    write_json(path, normalized)
    return path


def final_text(model_responses: list[dict]) -> str | None:
    """The content of the last model response's message, or None."""
    if not model_responses:
        return None
    choices = model_responses[-1].get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content")


def benchmark_metrics(record: dict) -> dict:
    """Benchmark metrics for one finalized raw record, read from the totals
    already on it (total_usage, akta_usage) so the raw and normalized results
    can't disagree. Token/cost fields are None when nothing reported them (vs.
    0 for a reported zero); call counts and latency are always known."""
    usage = record.get("total_usage") or {}
    totals = usage.get("totals") or {}
    akta = record.get("akta_usage") or {}
    details = totals.get("completion_tokens_details") or {}
    server_tools = totals.get("server_tool_use_details") or {}

    llm_cost = usage.get("cost")
    # None when the query made no akta.pro tool calls; a real 0.0 when it did but
    # every call was free.
    akta_cost = akta["total_cost"] if akta.get("tool_calls") else None
    total_cost = None
    if llm_cost is not None or akta_cost is not None:
        total_cost = (llm_cost or 0) + (akta_cost or 0)

    return {
        "latency_ms": record["latency_ms"],
        "total_cost": total_cost,
        "cost_breakdown": {"llm_cost": llm_cost, "akta_cost": akta_cost},
        "llm_calls": usage.get("calls", 0),
        "tool_calls": akta.get("tool_calls", 0),
        "prompt_tokens": totals.get("prompt_tokens"),
        "completion_tokens": totals.get("completion_tokens"),
        "total_tokens": totals.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        # Provider-side web searches (OpenRouter `web` plugin); None when the
        # response reported no server-tool usage.
        "web_search_requests": server_tools.get("web_search_requests"),
    }


def normalize(record: dict, output, metrics: dict) -> dict:
    """The stable, compact contract for judging/aggregation. Carries only what
    downstream needs; the full execution stays in the raw artifact. `count` is
    the number of companies in the output's companyList (None when there is no
    list, e.g. an errored request)."""
    companies = output.get("companyList") if isinstance(output, dict) else None
    return {
        "request_id": record["request_id"],
        "run_id": record["run_id"],
        "provider": {
            "id": record["provider"]["id"],
            "model": record["provider"]["model"],
        },
        "query": {"id": record["query_config"]["id"], "variant": record["query_config"]["variant"]},
        "status": record["status"],
        "error": record["error"],
        "count": len(companies) if isinstance(companies, list) else None,
        "output": output,
        "json_repaired": record["json_repaired"],
        "metrics": metrics,
    }
