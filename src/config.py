"""Benchmark settings, the provider and query config loaded from config/, and
run directories under results/."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
PROVIDERS_PATH = CONFIG_DIR / "providers.yaml"
QUERIES_PATH = CONFIG_DIR / "queries.yaml"
VALIDATION_DIR = CONFIG_DIR / "validation"  # per-query validation_prompt / validation_schema files
RESULTS_DIR = REPO_ROOT / "results"

# Relevance-validation judge model (src.validate), which judges each company
# against its fact pack (src.factpack).
JUDGE_MODEL = "google/gemini-3.5-flash"
# Output-token cap per judge call, reasoning included. It also bounds what
# OpenRouter reserves per in-flight call. A response that hits it is recorded as
# an error (finish_reason=length), never parsed.
JUDGE_MAX_TOKENS = 10000

# System prompt prepended to every benchmark request.
SYSTEM_PROMPT = (CONFIG_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()

MAX_CONCURRENCY = 5
MAX_TOOL_CALLS = 2000
# Loop guard: after this many identical tool calls in a row (same tool, same
# arguments), tools are withdrawn and the model must give its final answer.
MAX_REPEATED_TOOL_CALLS = 3

# Default retries on transient errors (OpenRouter, HasData, akta.pro MCP): total
# attempts, full-jitter exponential backoff between them.
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 30.0

# OpenRouter SDK client. The timeout is per network read, not per call (OpenRouter
# keeps long calls alive). The SDK's own retry (5xx, connection errors) is capped
# at this total time per attempt of ours; its default is 1 hour.
OPENROUTER_TIMEOUT = 300.0
OPENROUTER_SDK_RETRY_MAX_ELAPSED = 60.0

LOG_FILE = REPO_ROOT / "logs" / "pipeline.log"

AKTA_MCP_URL = "https://mcp.akta.pro/mcp"
AKTA_API_KEY = os.environ.get("AKTA_API_KEY")
AKTA_READ_TIMEOUT = 300.0
# akta.pro bills tool calls in credits (reported per result as `credits_consumed`).
# The USD rate depends on the plan; the default is the Enterprise rate.
AKTA_CREDIT_USD = float(os.environ.get("AKTA_CREDIT_USD") or 0.04)

# HasData Google AI-mode scrape: one call per fact-pack question, used by
# src.factpack to build each company's fact pack for the validation judge.
# HasData's account-level rate limit is much lower than OpenRouter/akta.pro's, so
# it gets its own concurrency cap instead of sharing MAX_CONCURRENCY.
HASDATA_API_URL = "https://api.hasdata.com/scrape/google/ai-mode"
HASDATA_API_KEY = os.environ.get("HASDATA_API_KEY")
HASDATA_READ_TIMEOUT = 60.0
HASDATA_MAX_CONCURRENCY = 7
# Billing: flat per-call credit cost -> USD, independent of response size. Both
# depend on the HasData plan.
HASDATA_CREDITS_PER_CALL = int(os.environ.get("HASDATA_CREDITS_PER_CALL") or 10)
HASDATA_CREDIT_USD_PER_1K = float(os.environ.get("HASDATA_CREDIT_USD_PER_1K") or 0.246)
HASDATA_CALL_COST_USD = HASDATA_CREDITS_PER_CALL / 1000 * HASDATA_CREDIT_USD_PER_1K


class Provider(BaseModel):
    id: str
    type: Literal["openrouter"]
    model: str


class VariantSettings(BaseModel):
    """Tool flags a variant sets; everything else is shared across a query's variants."""

    use_akta: bool = False
    use_web_search: bool = False
    # Cap on OpenRouter `web` plugin (native engine) searches per model call.
    # Independent of MAX_TOOL_CALLS, which caps akta.pro function-tool calls only.
    max_web_searches: int = 3


class QueryVariant(VariantSettings):
    name: str  # unique per query, e.g. "web_search" | "akta"


class Query(VariantSettings):
    """One query run in one variant. queries.yaml lists variants per query;
    load_config flattens each (query, variant) pair into one of these."""

    id: str
    text: str
    structured_output: str | None = None  # name of a schema in src.schemas.SCHEMAS
    # HasData fact-pack questions (src.factpack), shared across a query's
    # variants. Each template's {{company_name}}/{{website}} are filled in per
    # company; list order is the order of the company's factpack entries.
    fact_questions: list[str] = []
    # Relevance validation judge inputs, shared across a query's variants.
    # Filenames resolved under VALIDATION_DIR.
    validation_prompt: str
    validation_schema: str
    variant: str


class Config(BaseModel):
    providers: list[Provider]
    queries: list[Query]


def load_config(query_id: str | None = None) -> Config:
    """Providers and query-variants from config/. With `query_id`, only that
    query's variants; raises if queries.yaml has no such query."""
    providers = yaml.safe_load(PROVIDERS_PATH.read_text(encoding="utf-8"))["providers"]
    raw_queries = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    queries = []
    for raw in raw_queries:
        variants = [QueryVariant(**v) for v in raw.pop("variants")]
        for v in variants:
            queries.append(Query(**raw, variant=v.name, **v.model_dump(exclude={"name"})))
    if query_id is not None:
        known = sorted({q.id for q in queries})
        queries = [q for q in queries if q.id == query_id]
        if not queries:
            raise ValueError(f"no query {query_id!r} in {QUERIES_PATH.name}; known: {', '.join(known)}")
    return Config(providers=providers, queries=queries)


def fill_template(template: str, **values: str | None) -> str:
    """Replace each {{key}} in template with its value ("" for None)."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value or "")
    return template


RUN_STAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
# create_run_dir names: <stamp>, or <stamp>-<n> when that stamp is taken.
RUN_DIR_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:-(\d+))?")


def latest_run_dir(base: Path = RESULTS_DIR) -> Path:
    """Most recently created run directory under `base`, by the creation time
    in its name. Folder mtime isn't used: it changes whenever something is added
    to or copied into a run folder, so an older run can look newest."""
    runs = []
    for p in base.iterdir():
        match = RUN_DIR_RE.fullmatch(p.name)
        if p.is_dir() and match:
            runs.append(((match.group(1), int(match.group(2) or 0)), p))
    if not runs:
        raise FileNotFoundError(f"no run directories under {base}")
    return max(runs)[1]


def create_run_dir(base: Path = RESULTS_DIR) -> Path:
    """Create and return a fresh timestamped directory for one benchmark run.
    Call once per invocation; every result of that run is written into it."""
    stamp = datetime.now().strftime(RUN_STAMP_FORMAT)
    run_dir = base / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir
