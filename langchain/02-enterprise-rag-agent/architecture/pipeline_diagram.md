# NexusAgent Executive RAG Architecture

```mermaid
flowchart TD
    U[Executive] -->|Current question| CM[CombinedMemory]
    U -->|Current question| ME[Embedding model]
    ME -->|Query vector and nearest-neighbor search| HIST[(Session conversation FAISS index)]
    HIST -->|Up to 3 historical exchanges| VM[VectorStoreRetrieverMemory]
    BW[ConversationBufferWindowMemory: last 5 exchanges] -->|chat_history| CM
    VM -->|semantic_history| CM
    CM -->|Question and both histories| CTX[Agent LLM context window: instructions, tool descriptions, question, chat_history, semantic_history, agent_scratchpad]
    CTX --> LLM[Gemini conversational ReAct LLM]
    LLM -->|Tool name and standalone question| ROUTE[Tool dispatcher]

    ROUTE -->|Structured metrics| SQL[EnterpriseSQLDB: SQLDatabaseChain]
    SCHEMA[Allowlisted table schemas, no sample rows] -->|table_info| SP[SQL LLM context: dialect, schema, question, SELECT-only instruction]
    SQL -->|Standalone question| SP
    SP --> SLLM[SQL generation LLM]
    SLLM -->|Generated SQL| VALID[Parse one SELECT AST: approved syntax, functions, and tables only]
    VALID -->|Accepted SQL| GATE[Database access boundary]
    VALID -->|Rejected SQL| FAIL[Explicit failure; no mutation or invented evidence]
    DBA[DBA provisioning outside the agent] -->|Dedicated SELECT-only login| PGROLE[Startup privilege audit: reject elevated, owning, member, or write-capable login]
    PGROLE -->|Audit passes| GATE
    PGROLE -->|Audit fails| FAIL
    GATE -->|PostgreSQL mode: READ ONLY transaction, timeout| PG[(PostgreSQL enterprise metrics)]
    GATE -->|Local mode: mode=ro and query_only=ON| SQ[(SQLite fixture)]
    PG -->|Read schema metadata for approved tables| SCHEMA
    SQ -->|Read schema metadata for approved tables| SCHEMA
    SEED[data/enterprise_schema.sql: application bootstrap only] -->|Initialize fixture before opening agent read-only connection| SQ
    PG -->|Result rows| SQL
    SQ -->|Result rows| SQL
    SQL -->|Raw verified rows; return_direct| OBS[Tool observations]

    WIKI[Available HR policy text; production wiki connector] -->|Bounded chunks and source/access metadata in production| INGEST[Knowledge-base embedding model]
    INGEST -->|Document vectors and original text| KB[(Separate wiki FAISS index)]
    ROUTE -->|HR/Ops standalone question| QA[HR_Ops_Wiki: RetrievalQA / VectorStoreQA role]
    QA -->|Question| QE[Same knowledge-base embedding model]
    QE -->|Query vector; nearest-neighbor search| KB
    KB -->|Top-k text chunks; k=1 in fixture| QP[QA LLM context window: instructions, retrieved text as context, question]
    QA -->|Question| QP
    QP --> QLLM[Grounded QA LLM: chain_type=stuff]
    QLLM -->|Policy answer or unknown| QA
    QA -->|Grounded answer| OBS

    ROUTE -->|Current market query| NEWS[LiveMarketNews: SerpAPIWrapper]
    NEWS -->|External search request| SERP[SerpAPI / Google News]
    SERP -->|Search response| NEWS
    NEWS -->|Up to 3 titles, dates, URLs, snippets; or unavailable| OBS
    OBS -->|Normal ReAct loop: inject action and observation into agent_scratchpad| CTX
    LLM -->|Next tool action| ROUTE
    LLM -->|Final answer candidate| VERIFY[Evidence guard: enforce SQL for recognized current metrics; retain ALL tool evidence when SQL is used]
    VERIFY -->|Missing required SQL evidence| SQL
    OBS -->|Source results for final evidence guard| VERIFY
    VERIFY --> OUT[Final response or insufficient-evidence response]
    FAIL --> OUT
    OUT --> U
    OUT -->|Only successful final answer plus current question| SAVE[Memory save; exclude tool traces and injected histories]
    SAVE --> BW
    SAVE -->|Embed completed exchange| ME
    ME -->|New exchange vector and text| HIST
```

The three LLM context windows are distinct: the agent chooses tools and synthesizes
observations, the SQL chain receives only schema/question instructions, and RetrievalQA
receives retrieved wiki text through `{context}`. Wiki and news evidence return through
`agent_scratchpad`; remembered conversation uses `chat_history` and `semantic_history`.
The final evidence guard formats all source observations when SQL is used so a mixed
SQL/news/wiki question does not lose non-SQL evidence or replace database numbers.

The DBA role setup and fixture initialization run outside model-controlled tools. See
[postgres_readonly_role.sql](postgres_readonly_role.sql) for the dedicated-database
provisioning example. Startup verifies effective role privileges; each PostgreSQL
transaction also requests READ ONLY. SQL AST validation is an additional restriction,
not a replacement for database authorization. A rejected SQL statement raises an error;
the CLI reports that failure and does not save a successful conversation turn.

The history index and wiki index are separate. PostgreSQL and a hosted custom vector
database are production integration points; the local implementation uses seeded SQLite
and FAISS. The supplied wiki has one HR sentence, so k=1; production chunking, metadata
access filters, durable memory, and strict token-budget enforcement are deployment work.
Conversation FAISS is per executor, in process, and is not persisted across restarts.

In `--demo`, the same agent, memories, SQL chain, and wiki retriever execute with a
scripted model and deterministic local embeddings. The search tool explicitly reports
that live news is unavailable. No Gemini or SERP request is made in demo mode.
