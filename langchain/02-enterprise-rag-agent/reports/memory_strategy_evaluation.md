# Executive RAG Agent: Memory Strategy & Architecture Evaluation

**Project:** Enterprise Executive Conversational Agent  
**Stakeholder:** Priya & Executive Board, NexusAgent Systems  
**Engineer:** Course participant  
**Date:** November 9, 2026

## 1. Executive Summary

NexusAgent combines a conversational agent with three independently described tools:
SQL for enterprise metrics, retrieval QA for HR and operational policies, and SERP API
for current external news. An executive can ask a revenue question, switch to remote-work
policy, and then refer back to the earlier result without resending the entire conversation.
The agent selects tools from their descriptions and synthesizes their observations.

A hybrid memory design balances recent conversational flow with retrieval of older,
relevant exchanges. `ConversationBufferWindowMemory(k=5)` supplies recent interactions,
while `VectorStoreRetrieverMemory` retrieves up to three historical exchanges for the
current question. `CombinedMemory` injects both into the prompt under different keys.
This reduces the need to repeatedly include an ever-growing transcript, and can recover
context that has left the recent window. Retrieval can miss relevant facts, so memory
reduces context loss but does not eliminate hallucination or guarantee factual correctness.

The local implementation uses FAISS as the vector-store stand-in and SQLite initialized
from [data/enterprise_schema.sql](../data/enterprise_schema.sql). PostgreSQL is supported through configuration. No hosted
Nexus custom vector database or real HR corpus was supplied; the sample HR policy is
therefore the complete available policy corpus, not a production integration.

## 2. Architectural Diagram

The Mermaid diagram is provided in [architecture/pipeline_diagram.md](../architecture/pipeline_diagram.md).
It shows natural-language input, both memory injections, the conversational ReAct router,
the SQL/wiki/news tools, tool observations, final responses, and memory updates.

`chat_history` and `semantic_history` are deliberately distinct keys. Reusing a common
`history` key would collide in `CombinedMemory`. Both are explicitly present in the
agent prompt; combining the objects without referencing their variables would not
actually expose both sources to the model.

## 3. Memory Strategy Comparison

### Recent conversation: ConversationBufferWindowMemory

**Token consumption:** With `k=5`, the prompt receives at most the last five human/AI
exchanges, normally ten messages. This bounds the number of included turns, not the
number of tokens: one unusually long exchange can still be expensive. The classic
implementation can retain older messages internally even when it excludes them from
the prompt, so the window is not a guarantee of constant application memory usage.

**Latency:** No embedding call or similarity search is needed to load recent history.
Loading the window is inexpensive, although additional prompt tokens increase model
processing time and cost. This suits pronouns, corrections, and immediate follow-ups.

**Context loss and recall:** Chronological context is predictable within the window.
An important decision from six or more exchanges ago is omitted even if it becomes
relevant again. Window memory alone cannot recover it. It also preserves earlier
mistakes verbatim, so history must not replace current source evidence.

### Relevant historical exchanges: VectorStoreRetrieverMemory

**Token consumption:** Each completed exchange is embedded and indexed. At inference,
only up to three retrieved exchanges are included, rather than the whole transcript.
With `return_docs=False`, they are joined into a single formatted string. Result count
does not cap token length; three large documents can still overflow a budget. Embedding
calls add cost in live mode when saving and retrieving exchanges.

**Latency:** Retrieval adds query embedding and vector-search work. Saving an exchange
also requires embedding it. A small local FAISS index is inexpensive to search; a hosted
store adds network latency, and search behavior changes with index size and configuration.
No latency measurements or performance guarantees are claimed for this assignment.

**Context loss and recall:** Relevant information can be recalled after leaving the
recent window, including semantically related phrasing with live Gemini embeddings.
However, a relevant item can rank below the retrieval cutoff, and unrelated or stale
items can rank highly. The offline hashing embedder tests mechanics rather than genuine
semantic understanding. Retrieval does not prove that a remembered answer was correct.

## 4. Context Window & Token Optimization

The implemented hybrid uses five recent exchanges plus up to three retrieved exchanges.
The recent window supplies continuity when vector retrieval is incomplete; long-term
retrieval supplies older context missing from the window. Both are loaded each turn,
rather than treating the buffer as an exception-only fallback.

Explicit input and output keys avoid ambiguities when the executor returns its answer
and tool traces. The semantic store must contain the current question and final answer,
without recursively embedding the injected histories or storing agent scratchpads.
History retrieval is query-dependent, and the prompt labels recalled material as
potentially stale. Current metrics and news should be refreshed from their tools.

The prototype bounds retrieval counts but does not implement a strict token-budget
manager. For production, chunk or summarize oversized exchanges, deduplicate overlap
between recent and retrieved history, cap source and tool-result tokens, and reserve
space for the answer. A useful budget is: recent history + retrieved memory + tool
evidence + system instructions + answer reserve must fit the selected model's context.
Summaries need provenance because compression can lose qualifications and numbers.

### 4.1 Numerical token estimates and recommendation

The estimates below are explicit planning assumptions, not measurements or claims
about Gemini's maximum context. Use an application envelope of **16,384 tokens**
only if it is within the selected deployment model's actual input/output limits.
Verify those limits with the provider before deployment. Count serialized messages
with the selected model's tokenizer/count-tokens API; word or character counts are
not a reliable token limit, especially across languages.

Assume one exchange (user question plus assistant answer) averages **600 tokens**.
For one agent synthesis request, allocate **1,400** tokens for system instructions
and tool definitions, **300** for the current question, **600** for action labels
and scratchpad framing, and **3,300** for evidence (**600 SQL + 1,800 wiki answer +
900 news**). Reserve **2,048** for output and **1,536** as a safety margin. These
non-memory allocations total **9,184 tokens**, leaving **7,200** for history.
Schema generation and the internal wiki QA are separate LLM calls with separate
budgets; their prompts are not added wholesale to this agent request. ReAct
iterations do accumulate observations inside the agent scratchpad, so the evidence
allocation must cap the cumulative results, not just the latest tool result.

- Full transcript after 100 exchanges: **100 x 600 = 60,000 history tokens**;
  **69,184 total reserved tokens**, exceeding the 16,384 envelope by **52,800**.
- Recent top-K, K=5: **5 x 600 = 3,000 history tokens**; **12,184 total**, leaving
  **4,200** inside the envelope. It reliably preserves immediate references but
  drops every exchange older than five turns, irrespective of business relevance.
- Semantic memory, k=3: **3 x 600 = 1,800 history tokens**; **10,984 total**, leaving
  **5,400**. This uses **40% less history** than recent top-5 at equal item sizes,
  or about **9.8% less total reserved context**. It adds embedding/search calls and
  can miss relevant context; these savings do not establish equal answer quality.
- Implemented hybrid, 5 recent plus 3 retrieved: at most **4,800 history tokens**;
  **13,984 total**, leaving **2,400**. Without deduplication, recent turns can also
  appear in semantic results. The hybrid improves recall coverage but consumes
  **60% more history tokens than recent-only** under these assumptions. It is
  leaner than unbounded history, not automatically leaner than either component.

Stress case: at **1,500 tokens per exchange**, recent-only uses **7,500 history /
16,684 total** (300 over budget); semantic-only uses **4,500 / 13,684** (2,700 spare);
the hybrid uses **12,000 / 21,184** (4,800 over). Therefore result counts alone cannot
solve the original context-overflow problem. Long exchanges need bounded excerpts
or provenance-preserving summaries, and source observations need size limits too.

Recommendation: keep the hybrid for conversational continuity and older decision
recall, but apply a token-aware admission policy in production. A tighter starting
target is **2 recent exchanges capped at 500 tokens each + 3 retrieved excerpts
capped at 400 each = 2,200 history tokens**, or **11,384 total reserved tokens** under
the same assumptions. Prefer current corrections, remove duplicate exchange IDs,
and discard low-value old excerpts before dropping current evidence or answer space.
If required evidence still does not fit, split the task or report insufficient
context instead of silently truncating. These are proposed production settings;
the supplied configuration stub retains K=5/k=3 and has no strict token manager.

### 4.2 International enterprise deployment trade-offs

**Languages and tokenization.** Arabic, CJK, mixed scripts, and translated policy
wording must be measured independently with the deployment tokenizer. As a stress
assumption, if only the history expands to **1.5x**, the current hybrid grows from
4,800 to **7,200 tokens** and exactly fills the planned envelope (**16,384 total**).
At **2x**, it needs **9,600 history / 18,784 total**, exceeding the envelope by
**2,400**. These are sensitivity examples, not universal language ratios; translated
questions and source evidence can also grow. Test bilingual follow-ups and names,
use a multilingual embedding model assessed on the actual languages, and retain
original-language source/version metadata rather than assuming translation preserves
every condition. A chronology window works without embedding-language quality;
semantic recall needs cross-language retrieval evaluation.

**Regional latency and availability.** Recent history adds no embedding or search
round trip. As illustrative, unmeasured planning values, query embedding might take
**150 ms** and local vector lookup **40 ms** (**190 ms** retrieval overhead). A
cross-region store round trip of **180 ms** raises that to **370 ms**, before LLM
generation; saving the completed exchange may add another **150 ms** embedding call
on the response path in this synchronous implementation. Co-locating embedding,
index, and application services reduces network time but adds regional operations
and replication work. Measure p50/p95 in each region and each language. A production
degradation path can retain recent history while clearly reporting that older
memory is unavailable; the stub does not implement this automatic fallback.

**Retention, isolation, and residency.** The buffer and vectors can contain sensitive
executive/HR information. Create per-user/session indexes or enforce authenticated
tenant/session/access filters before retrieval; a global top-k search followed by
display filtering is insufficient. Keep stored text and derived vectors in approved
regions, choose compatible embedding endpoints, and define deletion/retention for
both copies. Replicated durable memory enables follow-ups after restart or regional
failover but increases storage, synchronization, and deletion work. The in-process
FAISS prototype loses history on restart and provides no cross-region persistence.
Policies and residency choices require the enterprise's own approval; no jurisdiction-
specific legal compliance is claimed by this stub.

**Freshness and operational consistency.** Store source version, UTC event time,
user timezone, language, and session ID with production memories. Resolve dates
such as "last quarter" using the executive's timezone and fiscal calendar, and
label the currency and reporting period in SQL answers. Re-fetch current metrics
and market news, even when remembered text sounds relevant. Regional vector replicas
may lag corrections; chronology helps immediate corrections, while version filtering
and expiration are needed to prevent older semantic matches overriding newer policy.
The sample fixed schema has no event-time/currency metadata beyond the supplied
fields, so richer time/currency normalization requires an enterprise data contract.

**Cost and quality.** Per turn, semantic memory normally embeds the question once
and the saved exchange once; wiki retrieval adds a separate query embedding when
used. Recent-only pays for more prompt tokens but avoids these memory embedding
requests. Production vector storage/search and regional replication add cost.
Compare total regional cost and retrieval precision/recall, not prompt size alone;
evaluate recall beyond five turns, corrections, unrelated distractors, and cross-
language paraphrases before choosing smaller k. No cost or latency benchmark is
claimed for the offline hashing embedder.

Create a separate executor/index per user session. This prototype keeps FAISS history
in process: it survives across turns on one executor, but not across restarts. Production
long-term retention requires a persisted store, session metadata, retention policy, and
appropriate access filtering. Neither memory strategy alone guarantees strict accuracy.

## 5. Tool Integration

### SQL Database Chain

`EnterpriseSQLDB` wraps `SQLDatabaseChain.from_llm` and `SQLDatabase`, with the three
supplied tables exposed: `regional_sales`, `employee_roster`, and `client_contracts`.
The chain receives a standalone question and table information, generates SQL, and
executes it. `return_direct=True` returns database results directly to the agent, avoiding
an extra answer-generation call inside the SQL chain. A SQL output parser removes
Markdown fences, and an executor guard exposes the actual database result for SQL
answers instead of unverified model arithmetic. When SQL is used, the final response
contains every SQL observation and any wiki/news observations in source-labeled
sections, preventing the old last-SQL-result overwrite from losing other evidence.
Recognized current-metric questions
require SQL evidence even if the conversational agent attempts to skip the tool.

For example, total Q2 revenue maps to `SELECT SUM(q2_revenue) FROM regional_sales`,
whose seeded result is 5,240,000 USD. The schema supports actual Q1/Q2 figures, not
Q3/Q4 forecasts. SQLite opens in read-only mode with `query_only` enabled. Configured
PostgreSQL sessions default to read-only and each transaction is explicitly READ ONLY.
Startup rejects elevated logins, role memberships, schema/database creation rights,
temporary-table rights, ownership, and effective table/column write grants in user
schemas, then verifies SELECT access on the three approved tables. The dedicated-role
[DBA provisioning example](../architecture/postgres_readonly_role.sql) is never run
by the agent. PUBLIC grants and future schema changes must remain under DBA control.

The SQL parser uses SQLGlot to require one SELECT/set-operation AST, approved tables,
and a conservative syntax/function allowlist. It rejects stacked statements, writable
CTEs, SELECT INTO, row locks, commands and arbitrary functions, while allowing
read-only CTEs and semicolons inside string literals. Database privileges remain the
authorization boundary: parsing does not prove that every possible SELECT is free
of side effects, and trusted schema owners must control operators, views and functions.

### VectorStore QA

`HR_Ops_Wiki` wraps a `RetrievalQA` chain backed by a separate FAISS index. Its sample
document permits engineers to work remotely up to three days per week. Retrieved text
is inserted into a context-constrained prompt that asks for an explicit unknown response
when evidence is absent. This wiki index is distinct from the conversation-memory index.
A real custom vector database can replace the retriever while preserving the tool's
question-in/answer-out contract. Source grounding mitigates, but cannot guarantee absence
of, unsupported model claims.

### SERP API

`LiveMarketNews` uses `SerpAPIWrapper` with Google News search. The tool passes up to
three titles, dates, URLs, and snippets to the agent. Dates and links let the user assess
freshness and provenance; search snippets can still be incomplete or wrong. Failed or
empty search results must produce an insufficient-evidence response instead of invented
news. A separate SERPAPI_API_KEY is required for live news. If absent, the news tool reports unavailability; SQL, wiki, and memory remain usable. In offline demo mode, the tool explicitly
returns that live search was not performed.

### Compatibility and verification scope

The starter uses legacy imports. The implementation retains its required APIs through
`langchain-classic`, moves FAISS/utilities to `langchain-community`, model and embedding classes to
`langchain-google-genai`, and SQLDatabaseChain to `langchain-experimental`. Deprecation warnings
are expected because the assignment explicitly requests these legacy memory/agent types.
The scripted demo validates wiring, not autonomous model reasoning. On November 9,
2026 this revision additionally completed a live Gemini chat/embedding run: SQLite
revenue retrieval returned 5,240,000 through EnterpriseSQLDB, wiki QA returned the
three-day remote-work policy, and a follow-up recalled the revenue result. SERP was
invoked but yielded no usable news, so live-news grounding remains unverified as
working. Production PostgreSQL credentials and cross-language/long-history recall
quality still require deployment verification. See the README for regression checks.

References: [LangChain classic memory](https://reference.langchain.com/python/langchain-classic/chains/conversation/memory),
[SQLDatabaseChain source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/sql/base.py),
[LangChain embeddings](https://docs.langchain.com/oss/python/integrations/embeddings),
[PostgreSQL READ ONLY transactions](https://www.postgresql.org/docs/18/sql-set-transaction.html),
[PostgreSQL privilege inquiry functions](https://www.postgresql.org/docs/18/functions-info.html),
and [SQLGlot AST inspection](https://sqlglot.com/sqlglot.html).

## Gemini provider adaptation

At the learner's request, the original OpenAI provider was replaced with
ChatGoogleGenerativeAI (gemini-3.5-flash-lite) and GoogleGenerativeAIEmbeddings
(gemini-embedding-001). Both use GOOGLE_API_KEY from Google AI Studio and the
Gemini Developer API backend. Model IDs are configurable through .env. The
orchestration, memory contracts, SQL schema, and three-tool design are retained.

Free-tier access is subject to project/model quotas; it is not unlimited. Memory
retrieval/saving also consumes embedding quota in live mode. A Google key does not
replace the separate SERP key. Without SERP credentials, current-news requests
explicitly report unavailability rather than blocking the whole agent.

## Manual execution screenshot

The following original screenshot records the user's interactive live Gemini run:
SQL returns 5,240,000, wiki retrieval returns the three-day remote-work rule,
a follow-up recalls the earlier revenue, and missing parental-leave information
receives an explicit unknown response. The combined response preserves SQL, HR,
and news sections, with news clearly unavailable. This is evidence of the displayed
run; it does not prove successful live-news retrieval or long-history semantic recall.

User's live Gemini chat screenshot (original execution evidence retained in the private coursework workspace)
