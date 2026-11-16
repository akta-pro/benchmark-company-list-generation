"""Run the benchmark: every provider in providers.yaml x every query-variant in
queries.yaml, then build the per-query combined pools.

Usage:  python -m src.fetch                              (new timestamped run)
        python -m src.fetch --run-dir results/<run_id>   (add to an existing run)
        python -m src.fetch --query <query_id>           (only that query; combine with either)

With --run-dir, results are written into that run (same run_id, so raw records
land in raw/<run_id>/fetch/) and the combined pools are rebuilt from every
per-request result in it, keeping existing fact packs (see
src.combined.build_combined). It refuses to run if any provider x query-variant
it would run already has a result there, so nothing is overwritten.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from src.akta import build_akta_http_client, list_akta_tools
from src.combined import build_combined
from src.config import MAX_CONCURRENCY, Config, create_run_dir, load_config
from src.logger import logger
from src.openrouter import build_openrouter_client, send_chat
from src.results import normalized_result_path


def existing_results(run_dir: Path, config: Config) -> list[Path]:
    """Result files in run_dir that this config would overwrite."""
    paths = (
        normalized_result_path(run_dir, provider.id, query.id, query.variant)
        for provider in config.providers
        for query in config.queries
    )
    return [p for p in paths if p.exists()]


async def main(run_dir: Path | None, query_id: str | None = None) -> None:
    config = load_config(query_id)
    if run_dir is None:
        run_dir = create_run_dir()
    else:
        if not run_dir.is_dir():
            sys.exit(f"--run-dir {run_dir} does not exist")
        clashes = existing_results(run_dir, config)
        if clashes:
            sys.exit(
                "refusing to overwrite existing results in "
                f"{run_dir}: {', '.join(p.name for p in clashes)}"
            )
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    logger.info("run dir: %s", run_dir)

    n_requests = len(config.providers) * len(config.queries)
    logger.info(
        "%d providers x %d query-variants = %d requests",
        len(config.providers), len(config.queries), n_requests,
    )

    need_akta = any(q.use_akta for q in config.queries)

    async def run(client, akta_http, akta_tools, provider, query) -> dict:
        async with semaphore:
            return await send_chat(client, provider, query, run_dir, akta_http, akta_tools)

    async with build_openrouter_client() as client, build_akta_http_client() as akta_http:
        akta_tools = await list_akta_tools(akta_http) if need_akta else None
        await asyncio.gather(
            *(
                run(client, akta_http, akta_tools, provider, query)
                for provider in config.providers
                for query in config.queries
            )
        )

    build_combined(run_dir)
    logger.info("done, %d requests written to %s", n_requests, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the list-generation benchmark.")
    parser.add_argument(
        "--run-dir", type=Path,
        help="add results to this existing run (results/<run_id>) instead of creating a new one",
    )
    parser.add_argument("--query", help="run only this query id from queries.yaml (default: all)")
    args = parser.parse_args()
    asyncio.run(main(args.run_dir, args.query))
