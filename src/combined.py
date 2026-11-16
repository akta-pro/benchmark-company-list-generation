"""Per-query candidate pool: concatenate the company lists every provider x
variant produced for a query, dedupe by website, and write one file per query
for src.factpack and src.validate to enrich later."""

from pathlib import Path
from urllib.parse import urlsplit

import tldextract

from src.config import Query, load_config
from src.logger import logger
from src.results import (
    HASDATA_RAW_SUBDIR,
    VALIDATION_RAW_SUBDIR,
    read_cost_ledger,
    read_json,
    write_json,
)

# results/<run_id>/combined/<query_id><COMBINED_SUFFIX>, one per query.
COMBINED_SUFFIX = "_all_companies.json"

# Use the bundled public-suffix snapshot; no network fetch, fully deterministic.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def registrable_domain(website: str | None) -> str:
    """eTLD+1 of a website URL, lowercased (e.g. 'https://www.Acme.com/x' ->
    'acme.com'). Empty string when there's no usable domain."""
    if not website:
        return ""
    ext = _extract(website.strip())
    return ext.top_domain_under_public_suffix.lower()


# Platforms that host many unrelated businesses (site builders, social and
# directory pages). Their registrable domain doesn't identify a company, so
# websites on them dedupe on host + path instead (joes.business.site,
# facebook.com/joeshvac).
SHARED_HOST_DOMAINS = frozenset({
    # site builders / hosting
    "business.site", "wixsite.com", "square.site", "godaddysites.com", "weebly.com",
    "wordpress.com", "blogspot.com", "squarespace.com", "webflow.io", "site123.me",
    "jimdosite.com", "carrd.co", "github.io", "netlify.app", "vercel.app", "myshopify.com",
    # social / link pages
    "facebook.com", "fb.com", "instagram.com", "linkedin.com", "x.com", "twitter.com",
    "youtube.com", "linktr.ee", "nextdoor.com",
    # directories / maps
    "google.com", "g.page", "yelp.com", "bbb.org", "angi.com", "homeadvisor.com",
    "thumbtack.com", "yellowpages.com", "mapquest.com", "crunchbase.com",
    # domain resellers that sell third-level names but are not public suffixes
    "uk.com", "us.com", "eu.com", "co.com", "com.de", "bank.in",
})


def dedupe_key(website: str | None) -> str:
    """Key companies are merged on: the host, minus "www.", plus path (+ query)
    for a site on a SHARED_HOST_DOMAINS platform. Empty string when the website
    can't identify a company: missing, or a bare platform URL such as
    https://www.facebook.com.

    The host is never reduced to its registrable domain: a subdomain belongs to
    whoever was given it, so hdfc.bank.in is not bank.in, idata.uk.com is not
    the uk.com registry, and pay.google.com is not Google. A company listed at
    both about.acme.com and acme.com therefore stays two rows."""
    domain = registrable_domain(website)
    if not domain:
        return ""
    url = website.strip()
    parts = urlsplit(url if "://" in url else f"https://{url}")
    host = (parts.hostname or "").removeprefix("www.").lower()
    if domain not in SHARED_HOST_DOMAINS:
        return host
    key = host + parts.path.rstrip("/").lower()
    if parts.query:
        key += f"?{parts.query}"
    return "" if key == domain else key


def build_request_metrics(record: dict) -> dict:
    """Provider/variant cost + latency entry for one per-request record.
    Included even on error, since a failed request can still have spent money
    (and time) on partial LLM/akta.pro calls before it failed."""
    metrics = record.get("metrics") or {}
    breakdown = metrics.get("cost_breakdown") or {}
    entry = {
        "request_id": record.get("request_id"),  # raw/<run_id>/fetch/<request_id>.json
        "total_cost": metrics.get("total_cost"),
        "llm_cost": breakdown.get("llm_cost"),
        "akta_cost": breakdown.get("akta_cost"),
        "latency_ms": metrics.get("latency_ms"),
    }
    if record.get("status") == "error":
        entry["errored"] = True
    return entry


def summarize_latency(entries: list[dict]) -> dict | None:
    """mean/min/max latency across a query's requests. None (not 0) when no
    request reported a latency at all."""
    values = [e["latency_ms"] for e in entries if e.get("latency_ms") is not None]
    if not values:
        return None
    return {
        "mean_ms": round(sum(values) / len(values)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_calls(calls: list[dict]) -> dict:
    """Cost/latency/count summary for one stage's {cost_usd, latency_ms,
    status} calls (see stage_calls)."""
    costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
    return {
        "total_cost": sum(costs) if costs else None,
        "latency_ms": summarize_latency(calls),
        "requests": len(calls),
        "errored": sum(1 for c in calls if c.get("status") == "error"),
    }


def stage_calls(run_id: str, query_id: str) -> dict[str, list[dict]]:
    """{stage: [{cost_usd, latency_ms, status, ...}, ...]} for every HasData and
    validation call of this query in the run's cost ledger
    (src.results.read_cost_ledger)."""
    calls: dict[str, list[dict]] = {HASDATA_RAW_SUBDIR: [], VALIDATION_RAW_SUBDIR: []}
    for entry in read_cost_ledger(run_id):
        if entry["query_id"] == query_id and entry["stage"] in calls:
            calls[entry["stage"]].append(entry)
    return calls


def refresh_stage_metrics(metrics: dict, run_id: str, query_id: str) -> None:
    """Recompute metrics.hasdata / metrics.validation from the run's cost ledger (stage_calls), then the top-level total_cost. A
    stage with no recorded calls yet gets no section."""
    for stage, calls in stage_calls(run_id, query_id).items():
        if calls:
            metrics[stage] = summarize_calls(calls)
    recompute_costs(metrics)


def recompute_costs(metrics: dict) -> None:
    """Refresh the cost roll-ups in a combined file's `metrics`:

    benchmark_cost  producing the lists - the per-request `breakdown`, summed
                    across every model and variant.
    evaluation_cost measuring them - the `hasdata` and `validation` stages,
                    spent on judging the pool, not on producing it.
    total_cost      both together: everything this run spent on this query.

    Each is None when nothing in that group reported a cost."""
    benchmark = [
        e["total_cost"]
        for e in (metrics.get("breakdown") or {}).values()
        if e.get("total_cost") is not None
    ]
    evaluation = [
        (metrics.get(key) or {})["total_cost"]
        for key in (HASDATA_RAW_SUBDIR, VALIDATION_RAW_SUBDIR)
        if (metrics.get(key) or {}).get("total_cost") is not None
    ]
    metrics["benchmark_cost"] = sum(benchmark) if benchmark else None
    metrics["evaluation_cost"] = sum(evaluation) if evaluation else None
    totals = [c for c in (metrics["benchmark_cost"], metrics["evaluation_cost"]) if c is not None]
    metrics["total_cost"] = sum(totals) if totals else None


def combined_file_path(run_dir: Path, query_id: str) -> Path:
    return run_dir / "combined" / f"{query_id}{COMBINED_SUFFIX}"


def find_combined_files(run_dir: Path) -> list[tuple[Query, Path]]:
    """(query config, combined file) for every combined file in the run. One
    Query per id: the config src.factpack/src.validate use is shared across a
    query's variants. Files with no matching query in queries.yaml are skipped."""
    queries = {q.id: q for q in load_config().queries}
    found = []
    for path in sorted((run_dir / "combined").glob(f"*{COMBINED_SUFFIX}")):
        query_id = path.name.removesuffix(COMBINED_SUFFIX)
        query = queries.get(query_id)
        if query is None:
            logger.warning("no query config for %s, skipping", query_id)
            continue
        found.append((query, path))
    return found


def load_combined(path: Path) -> dict:
    return read_json(path)


def save_combined(path: Path, data: dict, run_id: str, query_id: str) -> None:
    """Write a combined file, first recomputing its stage/cost metrics from the
    run's cost ledger so the saved costs are never stale."""
    refresh_stage_metrics(data.setdefault("metrics", {}), run_id, query_id)
    write_json(path, data)


def carry_over_factpacks(previous: list[dict], companies: list[dict]) -> int:
    """Copy each company's `factpack` from the combined file being rebuilt,
    matched on dedupe_key, so rebuilding after adding a provider keeps the fact
    packs already paid for. Verdicts are not carried: the pool changed, so
    src.validate reruns. Companies with no dedupe_key can't be matched and
    start without a fact pack. Returns how many fact packs were carried."""
    by_key = {
        key: c["factpack"]
        for c in previous
        if c.get("factpack") and (key := dedupe_key(c.get("website")))
    }
    carried = 0
    for company in companies:
        factpack = by_key.get(dedupe_key(company.get("website")))
        if factpack:
            company["factpack"] = factpack
            carried += 1
    return carried


def build_combined(run_dir: Path) -> list[Path]:
    """Write run_dir/combined/<query_id>_all_companies.json for each query.

    Companies are deduped by `dedupe_key`; the first occurrence's fields are
    kept and every (provider, variant, request_id) that listed it is recorded
    under `listed_by`. A company whose website yields no key is kept as its own
    row (never merged).

    Rebuilding over an existing combined file (e.g. after adding a provider to
    the run) keeps its fact packs (carry_over_factpacks); stage metrics come
    from the cost ledger (save_combined); verdicts are dropped. Returns the
    paths written."""
    by_query: dict[str, dict[str, dict]] = {}
    metrics_by_query: dict[str, dict[str, dict]] = {}
    for path in sorted(run_dir.glob("*__*__*.json")):
        record = read_json(path)
        query_id = record["query"]["id"]
        provider_id = record["provider"]["id"]
        variant = record["query"]["variant"]

        metrics_by_query.setdefault(query_id, {})[f"{provider_id}/{variant}"] = build_request_metrics(record)

        if record.get("status") == "error":
            continue
        source = {"provider": provider_id, "variant": variant, "request_id": record["request_id"]}
        pool = by_query.setdefault(query_id, {})
        for i, company in enumerate((record.get("output") or {}).get("companyList") or []):
            key = dedupe_key(company.get("website")) or f"{path.stem}#{i}"
            entry = pool.setdefault(key, {**company, "listed_by": []})
            if source not in entry["listed_by"]:
                entry["listed_by"].append(source)

    (run_dir / "combined").mkdir(exist_ok=True)
    written = []
    for query_id in metrics_by_query:
        companies = list(by_query.get(query_id, {}).values())
        path = combined_file_path(run_dir, query_id)
        previous = load_combined(path) if path.exists() else {}
        carried = carry_over_factpacks(previous.get("companies") or [], companies)

        breakdown = metrics_by_query[query_id]
        entries = list(breakdown.values())
        metrics = {"total_cost": None, "latency_ms": summarize_latency(entries), "breakdown": breakdown}

        if previous:
            logger.info("combined %s: rebuilt over existing file, %d fact packs carried over", query_id, carried)
        data = {"query_id": query_id, "count": len(companies), "metrics": metrics, "companies": companies}
        save_combined(path, data, run_dir.name, query_id)
        logger.info("combined %s: %d companies -> %s", query_id, len(companies), path)
        written.append(path)
    return written
