To: Lead Architect, FinCorp RAG Project
Subject: LCEL architecture for concurrent retrieval, query routing, and validated reporting

Dear Lead Architect,

I propose an LCEL workflow that accepts a natural-language question, retrieves
financial and news context concurrently, and selects the appropriate reporting
chain using a model-based router. The attached workflow_architecture.py provides
a functional architecture stub. It uses the Gemini Developer API as the selected
model provider; the original reference document mentions GPT-4o-mini.

## Linear extended chains versus dynamic router chains

A linear extended chain has a fixed sequence, which is easy to follow and debug
when every step depends on the previous result. For FinCorp, financial database
retrieval and news retrieval are independent. Running these sequentially adds
both source latencies, and running both inference chains for every request adds
unnecessary model calls. Combining all evidence in one general-purpose prompt
also mixes financial and news instructions and makes the required output contract
less explicit. Linear sequencing remains useful inside each branch: its prompt
must precede inference, which must precede validation.

A router selects the financial or news inference chain for the actual question.
This isolates the branch-specific evidence and reporting contract and allows a
new branch to be added without rewriting a long fixed sequence. The tradeoffs
are an extra classification call, classification latency and cost, possible
misrouting, and more branches to test. Constraining the route to a validated
enumeration and rejecting ambiguous or mixed questions limits unsupported
transitions but cannot guarantee perfect model classification. The design
overlaps classification with retrieval to reduce serial waiting.

## LCEL execution and concurrent retrieval

First, a RunnableLambda validates the input before any retrieval or model call.
A classification chain composed of PromptTemplate, Gemini, and a
PydanticOutputParser derives a constrained RouteDecision from the question:
financial, news, or unsupported. Callers do not supply a route label. Ambiguous,
unrelated, and combined financial/news requests require clarification rather than
silently selecting an unsuitable reporting contract.

For retrieval, RunnableParallel runs the structured-financial loader and the
unstructured-news loader concurrently against the same question. A second
RunnableParallel runs this retrieval group alongside query classification.
RunnablePassthrough.assign preserves the original question while attaching these
results. This avoids serializing independent work: retrieval latency is governed
approximately by the slower source, plus orchestration overhead, rather than the
sum of both source latencies. Retrieving both sources on every request is an
intentional tradeoff for this stub; production deployments should weigh that
extra source usage against latency and caching needs.

A RunnableLambda router then returns the selected runnable chain, which LCEL
invokes with the current context. Each branch follows
PromptTemplate | Gemini | PydanticOutputParser and receives only its relevant
source. FinancialSummary enforces the company name, QX YYYY quarter, revenue,
and net income in millions. MarketNews enforces the company name, sentiment
between -1 and 1, and exactly three headline strings. Extra fields, invalid
quarters, non-finite numbers, and other schema violations are rejected.
Internal routing metadata is kept outside the downstream reporting schemas.

## Contract enforcement and hallucination/noise containment

Each parser's get_format_instructions() embeds the corresponding JSON schema
into its prompt. After inference, PydanticOutputParser parses the response and
validates field names, types, and constraints against the selected model.
ConfigDict(strict=True, extra="forbid") prevents permissive type coercion and
rejects unexpected fields. FinancialSummary additionally checks the quarter
format and finite numerical values. MarketNews rejects scores outside [-1, 1]
and lists containing anything other than exactly three string headlines.

For example, a Q5 quarter, a sentiment score of 1.5, a fourth headline, a string
in place of a float, or an extra commentary field causes validation failure.
That failure propagates as an exception; the affected invocation emits no report
object for downstream consumption. The CLI reports failure with a nonzero exit
code. This gate prevents structurally invalid output and extra fields from
propagating to the reporting tools. Prompt instructions also require JSON only.

To reduce unsupported claims, prompts restrict extraction to supplied evidence,
distinguish EPS from net income, and require headline copying. Insufficient
evidence must fail rather than produce invented reporting data. Only a parsed,
validated Pydantic object is returned to downstream tools. Pydantic guarantees
the accepted output's structure and constraints; it does not establish factual
truth or eliminate hallucinations. Production integration should add deterministic
source-value checks and provenance, especially for financial reporting.

A made-up revenue with the correct numeric type can still pass Pydantic.
Consequently, structural validity must not be described as a guarantee against
factual hallucination. This stub uses evidence-only prompts and validation;
deterministic factual checks and provenance are production extensions.

## Prompt placeholder mapping

The caller invokes the pipeline with {"query": "..."}. Input validation strips
whitespace and preserves that value as query. The classifier inserts it into
{query}; its {format_instructions} is fixed from the RouteDecision parser.
The parallel loader sends the same query string to both mock functions and
returns financial_data and news_data. After aggregation, a RunnableLambda
flattens these values into the dictionary passed to the selected chain.

The financial template substitutes {query} and {financial_data}; the news
template substitutes {query} and {news_data}. Each template's
{format_instructions} is supplied by partial_variables from its own parser.
The other branch's source and the internal intent value are not placeholders,
so they are not inserted into the branch prompt. Matching inline comments in
workflow_architecture.py document these exact runtime mappings.

## Stub scope and verification

The loaders currently provide clearly labelled synthetic fixtures so the stub
can demonstrate real routing and extraction without database credentials or a
news-service integration. Every model response comes from Gemini. The fixtures
can be replaced by database/report retrieval and a news retriever while retaining
the same LCEL interfaces. API and parsing failures propagate without a fallback
report; production adapters should add source timeouts, bounded retries, and
sanitized tracing before connecting the reporting dashboard.

For acceptance testing, run both bundled questions and a custom --query, verify
the displayed selected routes and schema outputs, and check exit code 0.
Unrelated or mixed requests should request clarification and exit unsuccessfully.
Capture the final terminal results as screenshot evidence.

## Running the attached stub

Use Python 3.12 or newer. From the directory containing workflow_architecture.py:

    python -m pip install -r requirements.txt

Copy .env.example to .env and set GOOGLE_API_KEY to your Google AI Studio key.
GEMINI_MODEL selects the Gemini model. Existing process environment variables
take precedence over .env. Then run:

    python workflow_architecture.py

Both bundled questions run through automatic routing. For a custom question:

    python workflow_architecture.py --query "What were FinCorp Inc.'s revenue and net income in Q3 2023?"

A successful run prints each selected route and validated JSON and exits with
code 0. In PowerShell, run $LASTEXITCODE immediately afterward to check it.
Unrelated or mixed questions produce a clarification message and exit code 1.
The supplied mock sources support FinCorp Inc., Q3 2023; they are not live data.
The sentiment score may vary. API access requires network connectivity, a valid
key, model access, and project quota. API use follows the project's billing settings.

The submission contains this proposal, workflow_architecture.py, requirements.txt,
.env.example, and evidence/task1_gemini_output.png, which shows a successful manual
run of both routes. Local API keys and .env are excluded.

Best regards,
FinCorp RAG Architecture Team
