<div align="center">

# List Generation Benchmarking

### How AI models build company lists with [akta.pro](https://akta.pro) and with web search - by [akta.pro](https://akta.pro)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Queries: 18](https://img.shields.io/badge/queries-18-brightgreen.svg)](#queries)
[![Models: 3](https://img.shields.io/badge/models-3-orange.svg)](#models)

</div>

This repo gives AI models the same company-search tasks in two setups, one with the [akta.pro](https://akta.pro) MCP server and one with web search, then checks every returned company against gathered evidence.

A typical task:

> Find bootstrapped (founder-operated) HVAC service, maintenance & repair companies in the US with sub $10M valuation. Return a list of top 100 companies sorted based on revenue.

**How a run works:** every model gets every query once per setup through [OpenRouter](https://openrouter.ai), which makes 108 requests (3 models × 18 queries × 2 setups). A `web_search` request is one model call, and an `akta` request is a tool loop that continues until the model gives its final list. Each query's lists are merged into one pool, every company gets a fact pack of evidence, and an LLM judge decides from it whether the company matches the query.

> **Disclosure:** [akta.pro](https://akta.pro) is our own product, and this harness was built by the [akta.pro](https://akta.pro) team.

---

## Overview

Every model call, [akta.pro](https://akta.pro) tool call, HasData call and judge call is saved in full under `raw/<run_id>/`, so every step of a run can be inspected.

### Why this exists

Building company lists is a common ask of AI assistants. This repo runs the same list-building tasks with and without [akta.pro](https://akta.pro), the way a person would ask in a chat, and records every step.

### Variants

Every query runs once per model, per variant:

| Variant | Tools the model gets | How a request runs |
| --- | --- | --- |
| `web_search` | The model provider's native web search, through OpenRouter's `web` plugin | One model call, in which the model searches the web and returns its list |
| `akta` | Every tool listed by the [akta.pro](https://akta.pro) MCP server | A tool loop: the model calls [akta.pro](https://akta.pro) tools, reads the results and continues until it returns its list |

The `akta` tool list is read from the server when a fetch starts, so it can change over time. Refer to the [akta.pro MCP docs](https://docs.akta.pro/docs/developer-tools/mcp/setup) for more details on the tools and their usage. 

Both variants use the same [system prompt](config/system_prompt.txt) and query text, and return the same JSON structure: name, website, ownership type, HQ city, HQ country and product category.

### Models

Defined in [`config/providers.yaml`](config/providers.yaml). All calls go through [OpenRouter](https://openrouter.ai).

| Model | OpenRouter id |
| --- | --- |
| Claude Sonnet 5 | `anthropic/claude-sonnet-5` |
| GPT-5.6 Luna | `openai/gpt-5.6-luna` |
| GPT-5.6 Terra | `openai/gpt-5.6-terra` |


### Queries

Defined in [`config/queries.yaml`](config/queries.yaml), with the full query text and the fact questions for each. The `id` is used for the query's output files and run folders, and is purely used as an identifier.

To add your own query:

1. Copy an existing entry in [`config/queries.yaml`](config/queries.yaml).
2. Give it a new, unique `id`.
3. Change its `text` to your query, and its `fact_questions` to the questions that would confirm a company matches it. Keep `{{company_name}} ({{website}})` in each question, so it names the company being checked.
4. Leave `structured_output`, `validation_prompt`, `validation_schema` and `variants` as they are.
5. Run it with `--query <id>`, as in [Run a single query](#run-a-single-query).

| Query | Id |
| --- | --- |
| a16z-backed US AI companies, funded in last 2 years | `a16z_backed_us_ai` |
| US companies with a Tiger Global or Sequoia-led Series B+ | `tiger_sequoia_led_us` |
| European software, $20M+ Series B–D from Accel or Index | `europe_software_accel_index` |
| US unicorns founded in the last 3 years | `us_recent_unicorns` |
| PE-backed US/Canada professional services, $25–100M revenue | `pe_backed_b2b_services` |
| Corporate/PE-owned specialty chemicals, US & W. Europe, $50–500M revenue | `specialty_chemicals_corp_pe` |
| Nordic & Dutch energy storage hardware, no raise in 12 months | `nordic_dutch_energy_storage` |
| Series C+ PLG SaaS with usage-based pricing | `plg_usage_based_saas` |
| Bootstrapped US HVAC service cos, sub-$10M valuation | `bootstrapped_hvac` |
| Bootstrapped UK managed service providers, $5–10M revenue | `uk_bootstrapped_msps` |
| Bootstrapped US vet clinics, est. before 2000, $5–10M revenue | `us_bootstrapped_vet_clinics` |
| US AI-native companies selling to Microsoft | `us_ai_native_microsoft_customers` |
| Middle East real estate & proptech, $1M+ funding | `middle_east_proptech` |
| Series B+ Southeast Asian fintech with AI | `sea_fintech_seriesb` |
| Indian vertical marketplaces with network effects | `india_vertical_marketplaces` |
| Private European AI medical imaging software | `europe_ai_medical_imaging` |
| US data-monetization cos with a public API, funded in last 3 years | `us_data_monetization_api` |
| AI legal tech founded after 2010, global | `ai_legal_tech_global` |

---

## Pipeline

Three stages, each reading the previous stage's output:

| # | Stage | Module | What it does |
| --- | --- | --- | --- |
| 1 | **Fetch** | [`src/fetch.py`](src/fetch.py) | Runs every model on every query and variant, then merges each query's lists into one pool, deduplicated by website. |
| 2 | **Fact packs** | [`src/factpack.py`](src/factpack.py) | Gathers evidence on every company in the pool: answers to the query's fact questions from Google AI Mode (through [HasData](https://hasdata.com)). |
| 3 | **Judge** | [`src/validate.py`](src/validate.py) | An LLM judge decides whether each company matches the query, using only its fact pack. |

### How the judge decides

The judge is `google/gemini-3.5-flash`, and its instructions are in [`config/validation/q1.prompt.txt`](config/validation/q1.prompt.txt). We tested multiple versions of this prompt before the harness, and `q1` was finalised. That's where the name comes from. The other prompts were simply discarded before the runs.

The judge splits the query into criteria, marks each as satisfied, contradicted or missing, and applies this rule in order:

| Evidence | Verdict |
| --- | --- |
| Any criterion contradicted | `not_relevant` |
| Any criterion missing, none contradicted | `info_insufficient` |
| All criteria satisfied | `relevant` |

A company with no fact pack, or a judge call that fails, counts as `info_insufficient`.

---

## Getting started

### Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- API keys for OpenRouter, [akta.pro](https://akta.pro) and HasData
- An [akta.pro](https://akta.pro) plan whose company-list limit allows 100 results. The pay-as-you-go tier stops at 20.
- Enough monthly `generate_company_list` quota for the queries you run.
- An [akta.pro](https://akta.pro) plan that includes funding data. See [akta.pro pricing](https://akta.pro/pricing) for pricing and access details.

Our runs used an [akta.pro](https://akta.pro) Enterprise plan. [akta.pro](https://akta.pro) is a live service, so its data can change between runs.

### Before you run

- **Runs spend real money on your accounts.** One query can cost tens of dollars across generation, fact packs and judging, so start with `--query`. The code has no spending cap, and a single `akta` request can make billed tool calls up to the `MAX_TOOL_CALLS` cap.
- **Make sure every account has enough credit for the whole run.** The code doesn't check balances or quotas first. Running out partway gives you a partial run, as described in [When calls fail](#when-calls-fail).
- **Match concurrency to your plans.** The concurrency limits in [`src/config.py`](src/config.py) (`MAX_CONCURRENCY`, `HASDATA_MAX_CONCURRENCY`) must stay within what your OpenRouter, [akta.pro](https://akta.pro) and HasData plans allow. Going over them leads to rate-limited calls, which can fail after retries.
- **Results won't match ours exactly.** Models are non-deterministic, and [akta.pro](https://akta.pro), Google AI Mode and the web search results all change over time.
- **Each `src.fetch` without `--run-dir` starts a new run.** Stages 2 and 3 only process the latest run, so several `--query` runs are not combined unless you add them with `--run-dir`.
- **A missing API key stops the stage with an error.**

### 1. Install

```bash
uv sync
```

### 2. Configure `.env`

Copy [`.env.example`](.env.example) to `.env` in the repo root and fill it in:

```bash
OPENROUTER_API_KEY=...
AKTA_API_KEY=...
HASDATA_API_KEY=...
AKTA_CREDIT_USD=0.04
HASDATA_CREDITS_PER_CALL=10
HASDATA_CREDIT_USD_PER_1K=0.246
```

The last three only affect cost figures. Set them to match your own plans.

### 3. Run the stages

Run the three stages in order from the repo root:

```bash
uv run python -m src.fetch      # every model on every query, in a new run
uv run python -m src.factpack   # fact packs for the latest run
uv run python -m src.validate   # judge the latest run
```

#### Run a single query

Pass a query's `id` from [`config/queries.yaml`](config/queries.yaml) to `--query`. Every model still runs it in both variants, and stages 2 and 3 then process that run as usual:

```bash
uv run python -m src.fetch --query bootstrapped_hvac
uv run python -m src.factpack
uv run python -m src.validate
```

An unknown `id` stops the fetch and lists the valid ones.

#### Other options

These are for diagnostics and recovery, not a standard run. Use them to add to an existing run, rerun a stage on a specific run, or fill in fact-pack answers that failed:

```bash
uv run python -m src.fetch --run-dir results/<run_id>                # add to an existing run
uv run python -m src.fetch --run-dir results/<run_id> --query <id>   # add one query to an existing run
uv run python -m src.factpack results/<run_id>                       # fact packs for a specific run
uv run python -m src.factpack results/<run_id> --only-missing        # fill in failed answers and new companies
uv run python -m src.validate results/<run_id>                       # judge a specific run
```

`--run-dir` refuses to overwrite an existing result. It rebuilds the run's pools, keeping fact packs but dropping verdicts, so run the judge again afterwards.

### When calls fail

Stages finish and exit normally even when calls fail, so a finished stage doesn't mean a complete one. After each stage, check `metrics.<stage>.errored` in the pools for failed fact-pack and judge calls, and `status` in the per-request results for failed requests. Some typical failure cases are listed below.

| What happens | What you get |
| --- | --- |
| A model request fails after retries, is cut off or doesn't match the schema | `status` is `error`, and none of that request's companies enter the pool. Money already spent still counts in its cost. |
| The model's JSON has a syntax error | It is repaired where possible, and `json_repaired` is `true` |
| [akta.pro](https://akta.pro) MCP stays unavailable after retries | The whole request fails with `status` `error`, including any progress it had made |
| [akta.pro](https://akta.pro) rejects or refuses a tool call, including on quota | The model sees the error and carries on. The request can still finish as `completed` with a shorter list, and nothing in the result marks this. |
| The model reaches a tool limit | `status` is `max_tool_calls` or `repeated_tool_call`. Its final list is kept, or the request fails if it still asks for tools. |
| A HasData call fails after retries, including when credits run out | The answer is stored empty, and the judge still runs and is billed |
| A judge call fails, including when credits run out | The company becomes `info_insufficient`, and its rationale starts with `validation error:` |

Stopping a stage early has its own limits:

- **Fetch:** finished requests keep their results, but pools are only built when a fetch ends, so stages 2 and 3 have nothing to read. `--run-dir` then only works for a query with no results in that run.
- **Fact packs:** a pool is saved when its query finishes. Answers for an unfinished query stay in `raw/`, but they aren't reused and are asked again.
- **Judge:** there is no partial rerun. `src.validate` judges every company in the run again, at full cost.

---

## Output files

| Path | Contents |
| --- | --- |
| `results/<run_id>/<model>__<query>__<variant>.json` | One model's list for one query and variant, with status, cost, tokens and latency |
| `results/<run_id>/combined/<query>_all_companies.json` | The query's pool: every company, who listed it (`listed_by`), its fact pack and its verdict, plus costs by stage |
| `raw/<run_id>/<stage>/<request_id>.json` | The full record of every call, by stage: `fetch`, `hasdata`, `validation` |
| `raw/<run_id>/costs.jsonl` | Cost ledger for the fact pack and judge stages |

Pools record two costs separately: `benchmark_cost` for producing the lists, summed across every model and variant, and `evaluation_cost` for gathering fact packs and judging them. Each request's own cost is under `metrics.breakdown`. [akta.pro](https://akta.pro) usage is costed at $0.04 per credit by default, the Enterprise plan rate, set by `AKTA_CREDIT_USD` in `.env`. Other plans may have different prices. Learn more at [akta.pro pricing](https://akta.pro/pricing).

This repo ends at per-company verdicts. It has no scoring step, so turning verdicts into a score per model and variant is left to the reader, using `listed_by`. The curated top-20 lists, the website and existence check and the failure-reason classification are also not part of this repo.

---

## Metrics

This repo does not calculate these metrics. This section describes how the published figures were calculated from the verdicts.

Metrics are computed for each setup (one model and one variant) on each query, then combined across queries into overall figures.

| Metric | Definition |
| --- | --- |
| **Returned** | Distinct companies a setup listed for a query, de-duplicated by website. |
| **Correct** | Returned companies the LLM judge marked `relevant` for the query. |
| **Precision** | `correct ÷ returned`. What share of the returned companies actually match the query? |
| **Recall (Top 20)** | `curated top-20 companies returned ÷ 20`. Did the setup find the market's leaders? |

### Overall metrics

A setup's overall metrics are **pooled** across queries rather than averaged. Sum the numerators and the denominators across all queries, then divide. Do not average the per-query percentages.

```text
precision      = Σ_q correct(q)    / Σ_q returned(q)
recall_top20   = Σ_q top20_hits(q) / (20 × number_of_queries)    # denominator is 360 for 18 queries
```

### Curated top 20

Each query can have hundreds of valid companies, so total recall can't be measured. Recall is instead measured against a reference list of the 20 most relevant companies for each query.

To build it, take all the companies the LLM judge marked `relevant` for the query and use an LLM judge to select the 20 most relevant ones. For each setup and query, count how many of these 20 companies the setup returned. This count (`top20_hits`) is the numerator of Recall (Top 20).

## License

Released under the [MIT License](LICENSE).
