"""Relevance validation judge.

For a completed run, take each query's combined candidate pool
(results/<run_id>/combined/<query_id>_all_companies.json) and, per company, ask
the judge model whether that company matches the query - using ONLY that
company's fact pack (company["factpack"]), built by src.factpack from HasData
Q&A pairs and stored on the company in the combined file.

No web search. The verdict (`relevant`, `relevant_rationale`) is appended in
place onto each company in the combined file; the full per-call record is
dumped to raw/<run_id>/validation/.

Usage:  python -m src.validate [run_dir]   (defaults to the latest run)
"""

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from openrouter import OpenRouter

from src.combined import find_combined_files, load_combined, save_combined
from src.config import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    MAX_CONCURRENCY,
    VALIDATION_DIR,
    fill_template,
    latest_run_dir,
)
from src.logger import logger
from src.openrouter import build_openrouter_client, request_structured_output
from src.results import (
    VALIDATION_RAW_SUBDIR,
    drop_cost_entries,
    finalize_raw_record,
    new_raw_record,
    record_error,
    write_raw_result,
)
from src.schemas import RelevanceVerdict

# Synthetic provider row for the judge, matching the Provider dict shape.
JUDGE_PROVIDER = {"id": JUDGE_MODEL, "type": "openrouter", "model": JUDGE_MODEL}

INSUFFICIENT = "info_insufficient"


def render_fact_pack(factpack: list[dict]) -> str:
    """Render a company's factpack (src.factpack) as evidence for the judge:
    its HasData [{question, answer}, ...] entries as Q/A pairs in question
    order."""
    lines = []
    for i, entry in enumerate(factpack, start=1):
        lines.append(f"Q{i}: {entry.get('question', '')}")
        lines.append(f"A{i}: {entry.get('answer') or 'null'}")
    return "\n".join(lines)


async def judge_company(
    client: OpenRouter, sem, tag, run_id, query, template, schema, company: dict,
    fact_pack: str | None,
) -> tuple[dict, str | None]:
    """Returns (verdict, request_id of the judge call's raw record). request_id
    is None when no model call was made at all (no factpack). Per-call records
    go to raw/<run_id>/validation/ in the same shape as every other OpenRouter
    call (see src.results.new_raw_record)."""
    # No fact pack at all -> insufficient by the rubric, without a model call.
    if not fact_pack:
        verdict = {
            "relevant": INSUFFICIENT,
            "relevant_rationale": "No factpack available for this company (src.factpack was not run, or returned no answers).",
        }
        return verdict, None

    request_id = str(uuid.uuid4())
    prompt = fill_template(
        template,
        query=query.text,
        company_name=company.get("name"),
        website=company.get("website"),
        fact_pack=fact_pack,
    )
    messages = [{"role": "user", "content": prompt}]
    # query_config.text is the un-filled judge template (which rubric ran); the
    # filled prompt for this company is record["request"]["messages"][0]["content"].
    record = new_raw_record(
        request_id, run_id, JUDGE_PROVIDER,
        {"id": query.id, "text": template, "stage": "validation"}, messages,
    )

    model_responses: list[dict] = []
    async with sem:
        logger.info("%s judging %s", tag, company.get("name"))
        # Timed inside the semaphore: latency is the judge call itself, not the
        # wait for a free slot behind the other companies.
        start = time.monotonic()
        try:
            verdict = await request_structured_output(
                client, tag, record, model_responses,
                "company_relevance_verdict", schema, RelevanceVerdict,
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            record["status"] = "completed"
        except Exception as e:
            logger.exception("%s failed on %s: %s", tag, company.get("name"), e)
            record_error(record, e)
            verdict = {"relevant": INSUFFICIENT, "relevant_rationale": f"validation error: {e}"}
        latency_ms = round((time.monotonic() - start) * 1000)

    finalize_raw_record(record, model_responses, latency_ms)
    write_raw_result(record, VALIDATION_RAW_SUBDIR)
    return verdict, request_id


async def validate_query(client, sem, run_id, query, combined_path: Path) -> None:
    data = load_combined(combined_path)
    companies = data["companies"]
    template = (VALIDATION_DIR / query.validation_prompt).read_text(encoding="utf-8")
    schema = json.loads((VALIDATION_DIR / query.validation_schema).read_text(encoding="utf-8"))
    tag = f"[val {query.id}]"
    with_factpack = sum(1 for c in companies if c.get("factpack"))
    logger.info("%s %d companies, %d with a factpack", tag, len(companies), with_factpack)

    dropped = drop_cost_entries(run_id, query.id, (VALIDATION_RAW_SUBDIR,))
    if dropped:
        logger.info("%s cleared %d cost entries from earlier judge passes", tag, dropped)

    def pack_for(c: dict) -> str | None:
        factpack = c.get("factpack")
        return render_fact_pack(factpack) if factpack else None

    results = await asyncio.gather(
        *(
            judge_company(client, sem, tag, run_id, query, template, schema, c, pack_for(c))
            for c in companies
        )
    )
    verdicts = [v for v, _ in results]
    for company, (verdict, request_id) in zip(companies, results):
        company["relevant"] = verdict["relevant"]
        company["relevant_rationale"] = verdict["relevant_rationale"]
        # raw/<run_id>/validation/<request_id>.json; None when no judge call was made
        company["validation_request_id"] = request_id

    save_combined(combined_path, data, run_id, query.id)

    tally: dict[str, int] = {}
    for v in verdicts:
        tally[v["relevant"]] = tally.get(v["relevant"], 0) + 1
    logger.info(
        "%s done -> %s | %s | validation (all calls this run): %s",
        tag, combined_path, tally, data["metrics"].get(VALIDATION_RAW_SUBDIR),
    )


async def main(run_dir: Path) -> None:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    jobs = find_combined_files(run_dir)
    if not jobs:
        logger.warning("nothing to validate under %s", run_dir / "combined")
        return

    async with build_openrouter_client() as client:
        await asyncio.gather(
            *(validate_query(client, sem, run_dir.name, q, p) for q, p in jobs)
        )
    logger.info("validation done")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run_dir()
    logger.info("validation for run: %s", target)
    asyncio.run(main(target))
