"""Fact-pack layer.

Builds a fact pack for every company in each query's combined candidate pool
(results/<run_id>/combined/<query_id>_all_companies.json): each of the query's
`fact_questions` templates, filled with the company's name/website, is asked of
Google AI Mode through HasData (src.hasdata), one at a time.

The entries are stored on the company:

    company["factpack"] = [
        {"question": "<filled question>", "answer": "<answer>", "request_id": "<id>"}, ...
    ]

where request_id names the call's raw/<run_id>/hasdata/<id>.json.

With --only-missing, an entry is kept (no call) when it asks the same question
and has a non-empty answer. Everything else is fetched.

The combined file is updated in place. src.validate reads company["factpack"]
as the judge's only evidence.

Usage:  python -m src.factpack [run_dir] [--only-missing]   (defaults to the latest run)
"""

import argparse
import asyncio
from pathlib import Path

from src.combined import find_combined_files, load_combined, save_combined
from src.config import HASDATA_MAX_CONCURRENCY, Query, fill_template, latest_run_dir
from src.hasdata import build_hasdata_client, call_hasdata
from src.logger import logger
from src.results import HASDATA_RAW_SUBDIR, drop_cost_entries


def reusable(entry: dict | None, question: str) -> bool:
    """An existing fact-pack entry answers `question` and needn't be asked again."""
    return bool(entry and entry.get("question") == question and entry.get("answer"))


async def ask_fact_questions(
    http, sem, run_id: str, query: Query, company: dict, existing: list[dict], only_missing: bool
) -> tuple[list[dict], int]:
    """Ask the query's fact_questions about one company, one after another;
    `sem` bounds HasData calls across all companies. With only_missing, reusable
    entries are kept as they are. Returns (Q/A entries, number of HasData calls
    made)."""
    factpack: list[dict] = []
    asked = 0
    for i, template in enumerate(query.fact_questions, start=1):
        question = fill_template(template, company_name=company.get("name"), website=company.get("website"))
        previous = existing[i - 1] if isinstance(existing, list) and i <= len(existing) else None
        if only_missing and reusable(previous, question):
            factpack.append(previous)
            continue
        meta = {
            "query_id": query.id,
            "question_index": i,
            "company": {"name": company.get("name"), "website": company.get("website")},
        }
        async with sem:
            answer, record = await call_hasdata(http, question, run_id, meta)
        factpack.append({"question": question, "answer": answer, "request_id": record["request_id"]})
        asked += 1
    return factpack, asked


async def factpack_query(http, sem, run_id: str, query: Query, combined_path: Path, only_missing: bool) -> None:
    data = load_combined(combined_path)
    companies = data["companies"]
    tag = f"[factpack {query.id}]"
    logger.info(
        "%s %d companies x %d questions%s", tag, len(companies), len(query.fact_questions),
        " (only missing)" if only_missing else "",
    )

    # A clean run re-pays for the whole pack, so the earlier passes' costs are
    # superseded and cleared. A top-up keeps the answers it already has, so its
    # calls are added to what those passes recorded instead.
    if not only_missing:
        dropped = drop_cost_entries(run_id, query.id, (HASDATA_RAW_SUBDIR,))
        if dropped:
            logger.info("%s cleared %d cost entries from earlier fact-pack passes", tag, dropped)

    results = await asyncio.gather(
        *(
            ask_fact_questions(http, sem, run_id, query, c, c.get("factpack") or [], only_missing)
            for c in companies
        )
    )
    asked = 0
    for company, (factpack, company_asked) in zip(companies, results):
        company["factpack"] = factpack
        asked += company_asked

    save_combined(combined_path, data, run_id, query.id)
    logger.info(
        "%s done -> %s | asked %d, kept %d | hasdata: %s",
        tag, combined_path, asked, len(companies) * len(query.fact_questions) - asked,
        data["metrics"].get(HASDATA_RAW_SUBDIR),
    )


async def main(run_dir: Path, only_missing: bool) -> None:
    sem = asyncio.Semaphore(HASDATA_MAX_CONCURRENCY)
    jobs = []
    for query, combined_path in find_combined_files(run_dir):
        if not query.fact_questions:
            logger.warning("no fact_questions configured for %s, skipping", query.id)
            continue
        jobs.append((query, combined_path))

    if not jobs:
        logger.warning("nothing to fact-pack under %s", run_dir / "combined")
        return

    async with build_hasdata_client() as http:
        await asyncio.gather(
            *(factpack_query(http, sem, run_dir.name, q, p, only_missing) for q, p in jobs)
        )
    logger.info("factpack done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HasData fact packs for a run's combined pools.")
    parser.add_argument("run_dir", nargs="?", type=Path, help="results/<run_id> (default: latest run)")
    parser.add_argument(
        "--only-missing", action="store_true",
        help="keep entries that already answer the same question; ask only empty or missing ones",
    )
    args = parser.parse_args()
    target = args.run_dir or latest_run_dir()
    logger.info("factpack for run: %s", target)
    asyncio.run(main(target, args.only_missing))
