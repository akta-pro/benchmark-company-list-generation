"""Aggregation of OpenRouter and akta.pro per-call usage into request-level totals."""

from src.config import AKTA_CREDIT_USD


def akta_usage_summary(tool_calls: list[dict]) -> dict:
    """Aggregate akta.pro tool-call billing (credits -> USD) for one request."""
    total_credits = sum(c.get("credits", 0) or 0 for c in tool_calls)
    return {
        "tool_calls": len(tool_calls),
        "total_credits": total_credits,
        "credit_usd_rate": AKTA_CREDIT_USD,
        "total_cost": total_credits * AKTA_CREDIT_USD,
        "per_call": [
            {
                "name": c.get("name"),
                "credits": c.get("credits", 0),
                "cost": c.get("cost_usd", 0),
            }
            for c in tool_calls
        ],
    }


def finish_reason(response: dict) -> str | None:
    return (response.get("choices") or [{}])[0].get("finish_reason")


def add_usage(acc: dict, usage: dict) -> None:
    """Recursively sum numeric usage fields into acc, excluding `cost`."""
    for key, value in usage.items():
        if key == "cost" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            acc[key] = acc.get(key, 0) + value
        elif isinstance(value, dict):
            acc.setdefault(key, {})
            add_usage(acc[key], value)


def llm_usage_summary(responses: list[dict]) -> dict:
    """Aggregate a request's OpenRouter calls into cost, token totals, and a
    per-call breakdown. cost is None when no call reported one (vs. 0.0 for a
    reported zero)."""
    totals: dict = {}
    reported_costs = []
    per_call = []
    for i, response in enumerate(responses, start=1):
        usage = response.get("usage") or {}
        add_usage(totals, usage)
        if usage.get("cost") is not None:
            reported_costs.append(usage["cost"])
        per_call.append(
            {
                "call": i,
                "id": response.get("id"),
                "finish_reason": finish_reason(response),
                "cost": usage.get("cost", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        )
    return {
        "calls": len(responses),
        "cost": sum(reported_costs) if reported_costs else None,
        "totals": totals,
        "per_call": per_call,
    }
