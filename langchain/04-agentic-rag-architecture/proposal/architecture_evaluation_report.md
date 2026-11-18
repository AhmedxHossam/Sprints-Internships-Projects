# End-to-End Agentic RAG Architecture Proposal

**Document Control**

- **Project:** Higher-Education Client Portal - V2 Architecture
- **Organization:** OmniGraph AI
- **Prepared By:** AI Engineering - Architecture Proposal
- **Intended Reviewer:** Mateo Silva, Lead AI Architect (review pending)
- **Date:** 2026-11-10
- **Status:** DRAFT - architecture proposal for engineering review
- **Inputs:** supplied architecture proposal template; [system query logs](assets/system_query_logs.csv).
- **Companion:** [Architecture diagram](architecture_diagram.pdf).

## Executive Summary

Adopt a two-path RAG architecture: retain a predictable, single-retrieval path for well-supported questions, and invoke a bounded ReAct agent when evidence is insufficient or the request requires multiple sources, calculations, or an external system. LangChain handles document ingestion and retrieval tools, ChromaDB stores versioned vectorized chunks, the OpenAI API supplies embeddings and tool-capable generation, and Langfuse records request-level and component-level behavior.

The supplied sample contains 20 problematic requests from 2023-11-12, with mean recorded latency **13.162 seconds**, median **12.850 seconds**, and maximum **25.800 seconds**. Six requests timed out. These observations motivate targeted retrieval and context improvements as well as agentic orchestration; an agent alone will not repair missing documents, incorrect versions, or unavailable integrations.

Proposed initial performance objectives are p95 completed-response latency below 4 seconds for the static path and below 12 seconds for the fallback path, subject to a shared 12-second request deadline. They are hypotheses for validation, not measured gains. Quality, authorization, and correct abstention take precedence over producing a fast unsupported answer.

**Decision requested:** approve the blueprint for a separately scheduled prototype and evaluation. No production migration, API execution, or service provisioning is included in this deliverable. All snippets below are configuration/pseudo-code; application helpers are explicitly identified and need implementation. Dependency versions must be locked and integration-tested before deployment.

## Evidence Baseline and Limitations

### Sample interpretation

The CSV has six fields: `query_id`, `timestamp`, `user_prompt`, `latency_ms`, `error_type`, and `resolution_status`. All timestamps are on 2023-11-12 UTC, from 08:14:22 to 17:55:33. Treat this as a historical failure-oriented sample, not representative production traffic or current institutional policy.

- **Recorded latency:** minimum 4.100 s; mean 13.162 s; median 12.850 s; maximum 25.800 s. Empirical nearest-rank p95 is 22.100 s, using rank `ceil(0.95 * 20) = 19`; this small sample is unsuitable for establishing a production tail-latency SLO.
- **Outcomes:** 5 `Failed` (25%), 6 `Timeout` (30%), 4 `User_Abandoned` (20%), 3 `Resolved_Partial` (15%), and 2 `Escalated_to_Support` (10%). No full-success status appears in this sample. This does **not** establish a 100% production failure rate.
- **Error categories:** `Vector_Search_Miss` 4; `Context_Truncation` 4; `Multi_Hop_Failure` 3; `Temporal_Filter_Miss` 3; `Incomplete_Retrieval` 2; `Logic_Aggregation_Failure` 2; `Max_Tokens_Exceeded` 1; `External_Tool_Missing` 1.
- **Unknowns:** component timings, retrieval scores, document corpus and permissions, tool availability, concurrency, model identity, tokens, billing, and reference answers are absent. Labels suggest failure modes; they do not prove root causes. The template's rising-cost concern is not independently established by this CSV.
- **Temporal caveat:** some questions name 2024 materials although logged in 2023. Validate whether these were future published versions, synthetic scenarios, or inconsistent records. Resolve “last month” and “next year” relative to the original request timestamp during replay, not the report date.

### Failure-to-design mapping

- **Multi-source comparison:** `REQ-10041` (14.500 s), `REQ-10052` (17.800 s), and `REQ-10059` (18.900 s) report `Multi_Hop_Failure`. Decompose evidence needs; retrieve each version/entity separately and combine with citations. Use deterministic credit deduplication for `REQ-10052`.
- **Version selection:** `REQ-10042` (8.200 s), `REQ-10051` (6.200 s), and `REQ-10060` (6.700 s) report `Temporal_Filter_Miss`. Preserve publication and effective dates, and apply explicit version/as-of filters before vector search.
- **Context pressure:** `REQ-10043` (18.340 s), `REQ-10047` (11.200 s), `REQ-10053` (12.300 s), and `REQ-10058` (13.400 s) report `Context_Truncation`; `REQ-10045` (22.100 s) reports `Max_Tokens_Exceeded`. Bound context with token-aware packing, deduplication, and selective neighboring-section fetches. Keep eligibility clauses and exceptions with their rules.
- **Discovery misses:** `REQ-10044` (4.100 s), `REQ-10049` (5.400 s), `REQ-10055` (4.800 s), and `REQ-10057` (7.900 s) report `Vector_Search_Miss`. Check corpus coverage, normalize entity aliases, and allow exact identifier/title lookup; do not assume larger chunks solve them.
- **Missing evidence coverage:** `REQ-10046` (9.550 s) and `REQ-10054` (15.600 s) report `Incomplete_Retrieval`. Check all requested offices/forms or grant categories before answering; widen retrieval only for uncovered subquestions.
- **Set operations and dates:** `REQ-10048` (19.450 s) and `REQ-10056` (21.000 s) report `Logic_Aggregation_Failure`. Read authorized structured faculty/schedule records and use validated eligibility/interval functions; approximate vector neighbors cannot establish an exhaustive list.
- **External dependency:** `REQ-10050` (25.800 s) reports `External_Tool_Missing`. Add a read-only library-reserves connector with timeout and availability state. If unavailable, return the sourced reading list and explicitly mark reserve availability unverified.

## Pipeline & Ingestion Strategy

### The 5 Core Phases of the RAG Pipeline

For this proposal, the five phases describe the **offline indexing pipeline**, from a raw document to searchable vectors. Online retrieval, augmentation, and generation follow indexing and are shown separately in the diagram; the entire service is still end-to-end RAG.

1. **Load and extract.** Accept allowlisted PDF syllabi, HTML policy pages, and approved structured exports. Use LangChain document loaders such as `PyPDFLoader` for PDFs and an HTML loader for sanitized policy snapshots. Output LangChain `Document` objects with text, source URI, and page/section identifiers. Scanned PDFs require OCR before proceeding; quarantine unreadable files instead of silently indexing empty text. Store raw snapshots and checksums for reproducibility.
2. **Normalize and enrich.** Remove repeated navigation/headers while retaining headings, table row labels, policy clauses, and exceptions. Normalize whitespace without destroying lists. Attach authoritative metadata: institution/tenant, access scope, document ID, title, department/course, version, publication timestamp, effective-from/effective-to timestamps, source URI, and extraction quality. Resolve dates from a trusted manifest, never an invented model value. Missing critical version/ACL information blocks publication. Output clean, metadata-rich `Document` objects, ideally one logical section per object.
3. **Split into traceable chunks.** Apply the recursive splitter below within each section. Inherit document metadata, add chunk ordinal, parent section ID, and start offset. Preserve table headers with their rows; split long tables at row boundaries in preprocessing. Retain the original section in an object store for bounded adjacent-context retrieval. Output overlapping text chunks, not vectors yet.
4. **Embed.** Batch chunks through `OpenAIEmbeddings(model="text-embedding-3-small")` as an initial candidate. The embedding maps each chunk into a numerical vector; online query text must use the same model and dimensions. Cache embeddings by normalized-text hash plus embedding model/configuration. Benchmark this model against alternatives on the portal's retrieval set before acceptance. A model/dimension change requires a new collection and reindex. See [OpenAI embedding guidance](https://developers.openai.com/api/docs/guides/embeddings).
5. **Store and publish the index.** Write vectors, text, metadata, and deterministic chunk IDs to ChromaDB. Validate counts, permissions, version filters, and sample retrieval before switching the active collection pointer. Publish an immutable ingestion manifest and corpus version. Use stable IDs for idempotent writes; remove obsolete chunk IDs when a version is replaced and process source deletions. Maintain historical policy versions for comparisons while excluding revoked access. Output a retrievable, versioned collection.

Ingestion runs asynchronously from user requests. Incremental reindexing processes changed documents only. Failed parsing/embedding batches enter a retry queue with backoff and a dead-letter record; they cannot become partially published documents. Use one authoritative ingestion writer and a staged collection/alias swap for consistent reads. For production, deploy an authenticated Chroma server or managed service with backups; local persistence below illustrates configuration, not an availability design.

### Chunking Strategy

```python
# Configuration stub: clean_sections is the phase-2 list[Document].
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,          # characters, NOT tokens
    chunk_overlap=240,        # target overlap: 15% of chunk_size
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
    is_separator_regex=False,
    add_start_index=True,
)
chunks = splitter.split_documents(clean_sections)
```

**Justification for Chunking Parameters:** 1,600 characters is an initial compromise between preserving a short policy clause or syllabus subsection and limiting irrelevant context. The 240-character target overlap helps retain a condition or exception at a boundary without repeatedly embedding whole pages. Actual overlap depends on split boundaries; neither continuity nor answer quality is guaranteed by this parameter alone. Paragraphs and lines are preferred before sentence/word boundaries. The configuration uses the [documented recursive-splitting interface](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

For many English passages, 1,600 characters is roughly 400 tokens, but this is only a sizing estimate; tables, equations, and other languages differ. Pack retrieved context using the deployed model's tokenizer, with an initial **4,500-token evidence limit** and explicit room for instructions, question, tool schemas, and output. Six ordinary chunks often fit comfortably; use fewer or selective neighboring sections when they do not. Deduplicate overlap before generation. Never truncate off a safety exception to meet a limit: fetch the bounded parent section or abstain.

The four truncation labels and `REQ-10045` motivate this bounded strategy, but no documents or original prompts-to-model were supplied to prove that these exact sizes fix them. Evaluate sizes 1,000/1,600/2,400 characters with 10%/15%/20% overlap; compare required-evidence recall, citation completeness, packed tokens, retrieval latency, and index size. Choose the smallest setting meeting coverage targets, rather than maximizing chunk size.

### Vector Ingestion & Tool Wrapping

```python
# Configuration/pseudo-code: phase-3 chunks already carry validated metadata.
from hashlib import sha256
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = Chroma(
    collection_name="higher_ed_v2_embed3small_c1600_o240",
    embedding_function=embeddings,
    persist_directory="./data/chroma",  # local prototype only
)

def chunk_id(doc):
    identity = "|".join(str(doc.metadata[k]) for k in (
        "tenant_id", "document_id", "version", "parent_section_id", "start_index"
    ))
    return sha256((identity + "|" + doc.page_content).encode()).hexdigest()

# Batching must respect provider limits; checkpoint writes in the manifest.
for batch in batched(chunks, 64):  # application helper
    ids = [chunk_id(doc) for doc in batch]
    for doc, item_id in zip(batch, ids):
        doc.metadata["chunk_id"] = item_id
    store.add_documents(documents=batch, ids=ids)  # embeds then stores

def make_search_tool(trusted_scope):
    # trusted_scope comes from authenticated request context, never tool args.
    @tool
    def search_university_documents(query: str, as_of_epoch: int) -> dict:
        """Find policy/syllabus evidence valid at an explicit UTC timestamp."""
        version_filter = {"$and": [
            {"effective_from_epoch": {"$lte": as_of_epoch}},
            {"effective_to_epoch": {"$gt": as_of_epoch}},
        ]}
        # AND tenant + permitted access scopes + version predicates.
        where = combine_authorized_filters(trusted_scope, version_filter)
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6, "fetch_k": 20, "lambda_mult": 0.7,
                "filter": where,
            },
        )
        docs = retriever.invoke(query)
        return pack_evidence(docs, max_tokens=4500)  # application helper
    return search_university_documents
```

`batched`, `combine_authorized_filters`, and `pack_evidence` are proposed application functions, not LangChain APIs. The packer returns bounded text plus `chunk_id`, source URI, page/section, version, effective dates, and truncation/coverage flags. All tools share a request-level evidence budget; 4,500 tokens is not multiplied by the number of tool calls. Chunk ordinals are assigned per parent section during preprocessing. Open-ended effective intervals use a documented numeric sentinel; dates use half-open intervals `[from, to)` consistently. Preserve publication timestamps separately so a historical replay cannot retrieve a version not yet published at that time.

The server injects and validates tenant/access filters for **every** retrieval and exact lookup. Recheck authorization on parent-section fetches and structured APIs. Do not treat a Chroma metadata filter as the sole security boundary; the authenticated application/service must enforce scope and prevent clients from issuing unrestricted database queries.

MMR diversifies the candidate set to reduce repeated fragments; it is not a relevance-confidence score. The adequacy gate uses authorized document/version coverage, validated entity matches, and a separately calibrated relevance threshold, not an arbitrary universal Chroma distance cutoff. If lexical identifiers such as `CS-504` are missed, the fallback uses an authorized exact title/course lookup over a side index or catalog. This is an additional tool, not a claim that dense retrieval supplies exhaustive or keyword search. Interfaces follow [LangChain's Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) and [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering).

## Agentic Fallback Analysis

### Static Chains vs. Agentic Workflow

A hardwired chain follows a fixed sequence: embed question, retrieve top chunks, construct prompt, generate. This has fewer model round trips and is suitable for a direct policy lookup when the requested entity/version is fully supported. Its weakness is that it repeats the same sequence even when evidence spans documents, requires an external service, or needs deterministic computation. `REQ-10041`, `REQ-10048`, and `REQ-10050` illustrate these different demands.

A ReAct agent alternates choosing a next action, calling an allowlisted tool, and inspecting its result. It can request the missing policy version, look up a library record, or run a validated credit-set calculation. This flexibility adds model calls, tool latency, and failure modes, so it is a bounded fallback rather than the default for every question. The controller runs in an explicit LangChain/LangGraph-style state machine; the agent uses OpenAI Responses function calling for decisions. It receives evidence and concise tool outcomes, never authority to alter permissions or execute arbitrary code.

### Routing and Low-Latency Controls

1. Authenticate, establish the request clock and explicit date/entity constraints, then run one shared initial retrieval. Lightweight rules detect comparisons, exhaustive lists, arithmetic, or a named external dependency; no separate LLM router is required by default.
2. If evidence supports every requested facet and the query needs no specialized tool, use the **static path**: one grounded generation, with tools disabled. The answer includes source/version citations.
3. Trigger **fallback** for missing required versions/entities, weak/empty retrieval, conflicting evidence, structured calculations, exhaustive enumeration, or an external dependency. Reuse initial evidence and embeddings. Do not first generate an answer already known to be unsupported. If final citation validation fails, a bounded repair may enter fallback only within the original budgets.
4. Restrict tools by intent: document/version retrieval and exact/adjacent lookup; read-only library reserves; authorized faculty/catalog/schedule APIs; deterministic credit, accrual, and interval functions. Supply short distinct descriptions and strict schemas. The server executes tool calls and validates arguments/results; a model-generated tool name is never executed without allowlist checks. See [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling).
5. Parallelize independent reads, such as two known policy versions, with at most two concurrent tools. Dependent calls wait for prior results, such as resolving a faculty ID before checking eligibility. Batch record lookups rather than calling once per row. Cache only within tenant/access scope, date/version, and corpus version; invalidate on access or source changes.

**Initial guardrails (configuration values to validate):**

```yaml
routing:
  static_target_p95_ms: 4000
  fallback_target_p95_ms: 12000
  request_deadline_ms: 12000       # from ingress, includes initial retrieval
  final_response_reserve_ms: 2000
agent:
  max_decision_rounds: 3
  max_tool_calls: 6
  max_parallel_tools: 2
  max_generation_calls: 4         # decisions + final + any static attempt
  provider_call_timeout_ms: 2000  # also capped by remaining request time
  tool_timeout_ms: 1000
  max_evidence_tokens: 4500       # shared across all observations
  max_aggregate_input_tokens: 18000
  max_aggregate_output_tokens: 1600
  max_final_output_tokens: 700
  max_identical_tool_calls: 1
  sdk_automatic_retries: 0
```

The wall-clock deadline overrides every count limit: not all maximums can necessarily be consumed. Before each decision/tool/final call, check remaining time, tokens, and the approved per-request spend cap; cap network timeouts to the remaining allowance and cancel in-flight work on expiry or user disconnect. Stop tool routing once only the final-response reserve remains. Use at most one transient retry only if explicitly budgeted by the controller; SDK retries remain disabled. A budget or dependency failure returns a deterministic, cited partial response/clarification with its missing facts, without an extra LLM call if time has expired.

At the pilot gate, finance/engineering must set the request spend cap from an approved model's current tariff and the token bounds above; deployment stays blocked until that cap is configured. Select and pin `OPENAI_RAG_MODEL` using latency/quality evaluation in the target region; the proposal deliberately does not assume an unbenchmarked model or invent a dollar saving. A 2-second call timeout may need adjustment after testing; increase it only while revisiting the end-to-end budget.

```python
# OpenAI request shape; definitions/results are supplied by the controller.
import os
from openai import OpenAI

client = OpenAI(max_retries=0)
response = client.responses.create(
    model=os.environ["OPENAI_RAG_MODEL"],
    input=bounded_messages,
    tools=validated_strict_tool_schemas,
    tool_choice="auto",
    parallel_tool_calls=True,
    max_output_tokens=min(300, remaining_output_budget),
    timeout=min(2.0, remaining_seconds_before_final_reserve),
)
# Each function schema uses strict=True, additionalProperties=False,
# and required fields (nullable types for optional values).
# The controller validates and runs response.output function_call items,
# then appends the response output plus matching function_call_output items.
# Preserve required provider continuation items; bound evidence separately.
# Validate budgets BEFORE the next request. Final synthesis uses tools=[].
```

The controller owns shared counters across both paths, including a rejected static generation. Refuse an unbudgeted repair, cancel duplicate tool calls, and circuit-break repeatedly failing dependencies. Retain provider-required continuation items when using a reasoning-capable model; do not attempt to extract hidden chain-of-thought. Trace action names, arguments after redaction, observed results, and short outcome codes only.

### The ReAct (Think-Act-Observe) Cycle in Practice

Use `REQ-10041`: “Compare the Spring 2023 and Fall 2024 academic integrity policies for graduate engineering students.” Its recorded 14.500-second `Multi_Hop_Failure` makes the missing-version check concrete.

1. **Initial retrieval and gate:** results cover only one policy version. The gate records `missing_required_version`, bypassing static generation. If the 2024 version's availability as of the historical request is uncertain, validate publication metadata first.
2. **Think / choose:** determine that each semester needs separately versioned evidence for the graduate-engineering audience. This describes the control decision, not a request to expose private reasoning.
3. **Act:** issue two independent authorized searches in parallel, filtered by the requested semester/version and publication availability. Use exact version IDs where available rather than guessing term start dates.
4. **Observe:** inspect the returned version IDs, eligibility scope, clauses, and citations. If the discipline/appeals exception is missing, fetch that specific section once within budget. Tool output is evidence, not instructions to the agent.
5. **Finish:** compare only supported changes and cite both versions per claim. Explicitly identify an unavailable future/draft version or unresolved conflict; ask for the intended version if needed. A complete answer requires both sources, not merely two high-scoring chunks from one source.

For `REQ-10050`, extract the reading list first, batch library-reserve checks second, and join by identifiers. For `REQ-10048`, use an authorized complete faculty dataset with pagination completeness checks and a deterministic service-date rule. Missing HR access must yield an access-limited response; the agent cannot infer eligibility from public biographies. These are distinct tools, not extra prompt instructions to a vector-only chain.

### Evidence and Answer Controls

- Return document title, effective version, page/section, and source URI for substantive claims; track contradictory policies rather than blending them.
- Treat retrieved text and tool responses as untrusted data. Separate them from system instructions, prohibit arbitrary URL fetching/code execution, and validate schemas and numeric units.
- Do not claim exhaustive lists from top-k retrieval. Confirm complete authorized structured coverage or label the result incomplete.
- For medical/lab safety, benefits, and financial policy questions such as `REQ-10053`, preserve required exceptions and authoritative references. If the exact applicable procedure is missing, direct the user to the responsible institutional office instead of fabricating it.

## Observability Strategy

### Trace Design and Configuration

Use Langfuse observations to separate the end-to-end clock from query embedding, Chroma search, coverage gate, context packing, each agent decision, each tool execution, and final synthesis. Offline ingestion gets a separate trace correlated by corpus/ingestion version; do not add its duration to interactive latency. The [Langfuse instrumentation interface](https://langfuse.com/docs/observability/sdk/instrumentation) supports nested observations; their start/end timestamps give component durations.

The current documentation describes Python SDK v5 with Langfuse v4 compatibility requirements. Verify the target deployment against the [SDK compatibility guidance](https://langfuse.com/docs/observability/sdk/overview), then lock the tested patch versions. This proposal uses explicit observations and the standard OpenAI client to avoid double-counting a call through both automatic and manual tracing.

```bash
# Configuration names only: inject real values through a secret manager.
OPENAI_API_KEY=secret-manager-reference
OPENAI_RAG_MODEL=approved-and-pinned-model-id
LANGFUSE_PUBLIC_KEY=secret-manager-reference
LANGFUSE_SECRET_KEY=secret-manager-reference
LANGFUSE_BASE_URL=https://cloud.langfuse.com
# Choose an approved region or self-hosted endpoint before deployment.
```

```python
# Instrumentation pseudo-code; imports follow the current public SDK.
import os
from time import perf_counter
from langfuse import Langfuse, propagate_attributes
from openai import OpenAI

lf = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    base_url=os.environ["LANGFUSE_BASE_URL"],
)
llm = OpenAI(max_retries=0)

def traced_generation(messages, timeout_s, output_limit):
    # Reuse for EVERY decision, synthesis, or repair call, not only the final.
    with lf.start_as_current_observation(
        name="openai.generation", as_type="generation",
        model=os.environ["OPENAI_RAG_MODEL"],
    ) as gen:
        result = llm.responses.create(
            model=os.environ["OPENAI_RAG_MODEL"], input=messages,
            max_output_tokens=output_limit, timeout=timeout_s,
        )  # decision-call variant also supplies the allowlisted tools
        if result.usage is not None:
            u = result.usage
            cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
            reasoning = getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0
            gen.update(
                usage_details={  # mutually exclusive buckets
                    "input": u.input_tokens - cached,
                    "input_cached_tokens": cached,
                    "output": u.output_tokens - reasoning,
                    "output_reasoning_tokens": reasoning,
                },
                metadata={"usage_available": True},
            )
        else:
            gen.update(metadata={"usage_available": False})  # never fake zero
        return result

def traced_request(req, ctx):
    started = perf_counter()
    with lf.start_as_current_observation(
        name="higher-ed.rag", as_type="span",
        metadata={"query_id": req.id, "corpus_version": ctx.corpus_version},
    ) as root:
        with propagate_attributes(
            user_id=ctx.pseudonymous_user_id,
            session_id=ctx.pseudonymous_session_id,
            tags=["higher-ed", "architecture-v2"],
        ):
            try:
                # Application controller emits nested observations for
                # embedding, search, gate, pack, decisions, tools, and final.
                outcome = run_bounded_controller(
                    req, ctx, langfuse=lf, generate=traced_generation,
                )
                root.update(metadata={
                    "route": outcome.route,
                    "fallback_reason": outcome.fallback_reason,
                    "error_type": outcome.error_type,
                    "resolution_status": outcome.status,
                    "latency_ms": round((perf_counter() - started) * 1000),
                })
                return outcome
            except Exception as exc:
                root.update(
                    level="ERROR", status_message=type(exc).__name__,
                    metadata={"resolution_status": "Failed"},
                )  # no raw exception text or student data
                raise

# Flush at graceful worker shutdown or after an offline evaluation batch.
# Do not call lf.flush() on the synchronous request path.
```

`run_bounded_controller` and the request/context/outcome objects are application contracts. Use the same generation wrapper for tool-enabled decisions with the additional request fields shown earlier. Create `as_type="embedding"` observations around document/query embedding calls and ingest provider-reported usage through an adapter or supported integration; LangChain's embedding abstraction alone must not be assumed to expose billing usage. Instrument tool failures and failed LLM calls even when provider usage is unavailable. Apply SDK/export masking and request redaction before capture, with retention and access controlled by the institution. No raw prompt, document text, or generated answer is persisted by the explicit stub.

**Cost accounting:** infer generation/embedding cost only when the exact model ID matches a verified Langfuse price definition and provider usage is present; otherwise show cost as unknown. The SDK's flat usage buckets must be disjoint, hence cached/reasoning tokens are subtracted from inclusive totals above. Configure the corresponding bucket prices, including reasoning output where applicable. For private tariffs, submit audited `cost_details` in USD instead of relying on defaults. Sum child costs once per trace; never add an already-aggregated trace total again. Separate online query-embedding/generation cost from offline ingestion cost and infrastructure charges. This follows [Langfuse usage and cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking); the supplied CSV supports no historical dollar estimate.

### Monitoring Objectives and Operational Response

- **Token usage and cost:** attribute requests to pseudonymous users/sessions, route, model, corpus version, and prompt version. Monitor input/output/cached/reasoning counts, cost per answered request, cost per completed request, spend by department where authorized, and usage/cost coverage. Count retries and failed calls when usage is returned. Investigate missing pricing or usage rather than interpreting missing cost as free service.
- **Latency bottlenecks:** chart p50/p95/p99 end-to-end completed-response latency by route and request category, plus time to first token separately when streaming. Compare embedding, Chroma, tool, and LLM spans. Parallel span durations overlap; analyze the critical path rather than summing them as if sequential. Slow Chroma points toward filters/index/server load; slow generation points toward model/output/context; repeated decisions point toward routing or evidence gaps. Report deadline cancellations separately from provider timeouts and failures.
- **Query-log continuity:** keep the supplied six CSV fields as the outcome vocabulary and add trace ID, route, fallback reason, stage, tool name, tool-call count, token counts, model/corpus/prompt versions, citation coverage, and usage-available flag. A generated answer is not automatically `Resolved`; define `Answered_Grounded`, `Partial`, `Abstained`, and `Dependency_Unavailable` alongside an explicit mapping to historical outcomes.
- **Targeted diagnosis:** for `Temporal_Filter_Miss`, inspect date resolution and returned versions before changing prompts. For `External_Tool_Missing`, inspect connector health/tool exposure. For `Context_Truncation` and `Max_Tokens_Exceeded`, inspect pre/post-pack token counts. For `Logic_Aggregation_Failure`, inspect record completeness and rule inputs. Refine tool descriptions when misrouting is observed; reindex when corpus coverage is the issue.
- **Initial alerts:** after at least 100 requests per route in a 15-minute window, alert if static p95 exceeds 4 s or fallback p95 exceeds 12 s. Alert when combined timeout/deadline rate exceeds 2%, or provider-usage coverage for completed model calls falls below 99% over that window. Alert on any authorization failure or repeated connector outage immediately. These thresholds are proposed pilot settings; tune for traffic and error budgets.
- **Telemetry reliability:** export asynchronously with bounded queues, watch exporter errors/drop counts, and keep a lightweight application latency/error counter if export fails. Telemetry failure cannot be allowed to block serving. Keep complete traces during the controlled pilot with sensitive content excluded; production sampling must preserve errors/slow requests and label sampling so dashboards are not mistaken for unsampled population metrics.

## Validation and Controlled Rollout

### Acceptance Plan

1. **Build evidence, not inferred answers.** Obtain authorized policy versions, syllabi, structured records, and library test access. Have subject-matter reviewers create expected source sets and answers for the 20 logged scenarios, including publication-date ambiguities. Add representative successful single-hop questions, permission tests, missing-source questions, and dependency failures. The CSV alone is not an answer-quality benchmark.
2. **Compare equal conditions.** Run the current static chain, the improved static retriever with the same corpus/chunking, and the proposed two-path system on the same frozen dataset, region, model policy, and concurrency. This isolates gains from better ingestion versus agent routing. Record warm/cold cache separately and include an unsampled production-like traffic set; do not infer the fallback rate from this problem-only sample.
3. **Proposed quality gates.** Required-source recall at six results at least 90% on the annotated retrieval set; at least 90% fully supported answers on answerable scenarios; every returned policy claim has a valid source/version citation; no unsupported complete answer on unanswerable cases; zero cross-scope data exposures. Measure completeness separately for enumeration, arithmetic accuracy for deterministic tools, and abstention quality. Sample sizes and confidence intervals accompany reported rates.
4. **Proposed performance gates.** Under agreed load, meet the 4 s static / 12 s fallback p95 objectives while tracking end-to-end success and separate timeout/deadline rates below 2%. Deadline-capped partial answers do not count as successful full answers. Model usage is captured for at least 99% of completed provider calls; every priced call matches an approved tariff. Set and enforce per-request spend caps and a pilot daily budget before enabling paid traffic.
5. **Pilot and rollback.** Begin with offline replay, then shadow retrieval with no user-visible changes, then a small authorized canary behind separate retrieval/agent feature flags. Monitor quality, route mix, latency, spend, and tool availability. Disable the agent flag on regression and retain the improved static path; restore the prior collection pointer on ingestion faults. Fail closed on permission issues. Engineering approval assigns owners and dates for a later migration roadmap.

### Remaining Engineering Decisions

- Approved model ID, tested package locks, hosting region, concurrency target, Chroma sizing/backup strategy, and Langfuse deployment compatibility.
- Authoritative document/version manifests, course/HR/library API contracts, and the institution's access/retention policy.
- Calibrated relevance/coverage thresholds, realistic call timeouts, monetary caps, and evaluation-set ownership.

These decisions do not block reviewing the architecture, but they block production readiness. No latency reduction, quality gain, or dollar saving is claimed as already achieved.

## Deliverables and Requirement Coverage

- **Five phases and LangChain ingestion:** Pipeline & Ingestion Strategy; top lane of the PDF.
- **Specific recursive chunking and justification:** Chunking Strategy; 1,600 characters / 240 overlap with a token-aware evidence cap.
- **ChromaDB storage and retrievable tool:** Vector Ingestion & Tool Wrapping; embeddings, persistence, IDs, metadata filters, and a scoped LangChain tool.
- **Static/ReAct comparison and low-latency routing:** Agentic Fallback Analysis; separate PDF lanes, bounded loop, external/structured tools, and stop outcomes.
- **Langfuse configuration and log-grounded monitoring:** Observability Strategy plus Evidence Baseline; timing, usage, cost, error attribution, and specific request IDs.
- **Submission package:** `proposal/` contains `architecture_evaluation_report.md`, `architecture_diagram.pdf`, and the supplied dataset at `assets/system_query_logs.csv`. Generation and validation utilities remain outside the submission folder.

## References

Primary supplied evidence is [assets/system_query_logs.csv](assets/system_query_logs.csv) (20 historical records); the structure follows the supplied architecture proposal template. External interfaces were checked against official documentation on 2026-11-10. Inline links identify the supporting pages; proposed values and design decisions are this report's recommendations.

- [LangChain recursive text splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Chroma metadata filtering](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Langfuse SDK overview and compatibility](https://langfuse.com/docs/observability/sdk/overview)
- [Langfuse observation instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation)
- [Langfuse Python SDK reference](https://python.reference.langfuse.com/langfuse)
- [Langfuse token and cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)

*Upon approval of this architectural blueprint, the engineering team will establish a rollout timeline and initiate the migration roadmap.*
