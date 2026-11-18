# Trade-off Analysis: Local Schema Checks and LLM Evaluation

**Scope:** One approximately 1,000-word English financial report.  
**Date:** 11 November 2026.  
**Evidence status:** API latency and token counts below are planning assumptions, not observed staging traces. Local verification is summarized in the [README](../README.md#verification).

## Decision

Use a hybrid gate. Apply cheap deterministic checks first and spend a judge call only on a structurally valid draft. Use the judge for source-grounded factual review, prose formatting and professional tone. Schema-only approval is unsuitable for these financial reports because a well-typed, well-formatted report can still contain invented metrics.

Pydantic and an LLM judge have different responsibilities. Pydantic can also validate the judge's structured response; that local check does not replace the judge's semantic assessment. [Pydantic field constraints and descriptions](https://docs.pydantic.dev/latest/concepts/fields/) support this explicit data contract.

## Comparable planning assumptions

Use `gemini-3.5-flash-lite` for generation, evaluation and optimization, priced at **USD 0.30/MTok input and USD 2.50/MTok output**. These are standard paid-tier rates without caching, batch discounts or paid tools from [Google's pricing documentation](https://ai.google.dev/gemini-api/docs/pricing). Free-tier invoices may differ; paid-tier rates provide a comparable planning baseline. Configuration comes from this module's independent `Task Module_1/.env`.

The single-model selection is explicit. [Google's API migration documentation](https://ai.google.dev/gemini-api/docs/gemini-3#api-changes-and-parameter-updates) states that 3.5 Flash-Lite ignores temperature, so requesting zero does not enforce the assignment's sampling requirement. All roles use minimal thinking, which reduces reasoning effort rather than guaranteeing zero thinking tokens. Model availability and financial-evaluation quality still require verification.

Assume the English report occupies 1,400 tokens, original source data 1,000 tokens, and each evaluation prompt another 2,100 tokens for instructions, schema descriptions, serialization and provider overhead. One judge call therefore uses approximately **4,500 input tokens, 500 visible output tokens and an assumed 100 thinking tokens**, or **600 billable output tokens**. The judge output allowance is 1,000 tokens, not a promised completion length. The 100-token thinking allowance is an unmeasured scenario assumption, not a fixed Gemini budget. Measure actual usage, including thinking, for the chosen model and serialization. Other languages, dense tables and longer sources change these estimates.

## Incremental validation per report draft

| Dimension | Local Pydantic checks | LLM-as-a-judge plus output validation |
| --- | --- | --- |
| Added latency | Measured warm gate: p50 0.0108 ms, p95 0.0122 ms on a synthetic 1,000-word input; excludes imports and process start | Approximately 4,000-11,000 ms per judge call; local parsing adds a small separate cost |
| Added input tokens | 0 | Approximately 4,500 |
| Added output tokens | 0 | Approximately 500 visible + 100 thinking = 600 billable; configured allowance 1,000 |
| Added API cost | USD 0.0000 | USD 0.002850 at the expected token lengths |
| Reliable scope | Explicit types, presence, ranges and implemented validators | Contextual source comparison, narrative presentation and tone; quality requires calibration |
| Main blind spot | Unsupported financial claims that satisfy the schema | Can overlook contradictions or misgrade; schema-valid output is not proof of correct judgment |

Local CPU and hosting costs are not zero; they are excluded from the API cost column. The implemented local gate checks report text and required headings. Detecting an invalid date embedded in prose requires a parser/validator or the judge; Pydantic does not discover that rule automatically.

The local values come from 10,000 warm calls after 100 warmups on this Windows/Python 3.14.5 environment. The synthetic input tests scan length, not realistic financial complexity. They are not staging trace measurements and exclude judge-output validation, tracing, queueing and future validators. A provisional 1-10 ms service-span allowance is a planning budget, not the measured execution time; verify it in staging. See the [verification summary](../README.md#verification).

The judge latency range is an **analytical estimate**, not a published provider benchmark: assume 500-2,000 ms of request/prefill overhead, 60-150 visible output tokens/second and 0-500 ms of additional minimal-thinking overhead. Then `500 / throughput * 1,000 + overhead + thinking_overhead` is approximately 3,833-10,833 ms, rounded outward to 4,000-11,000 ms. This model simplifies input processing, queueing and network variability; actual tail latency can exceed the range. The attachment template's example figures are not treated as measurements. A small integration smoke test, if successful, is not a representative 1,000-word latency benchmark.

## API cost calculation

For input tokens `I`, output tokens `O`, and prices per million tokens `Pin` and `Pout`:

`cost_USD = (I * Pin + O * Pout) / 1,000,000`

One expected evaluation costs `(4,500 * 0.30 + 600 * 2.50) / 1,000,000 = USD 0.002850`. At 1,000 billable output tokens and the same input length, it costs USD 0.003850. Include schema overhead in input usage and thinking in output usage; do not count only visible text. If the provider reports thought tokens separately, avoid double-counting them when calculating billable output.

For total workflow estimates, assume the generator uses 1,800 input tokens and 1,400 visible + 100 thinking output tokens (**USD 0.004290**). Each optimizer uses 4,000 input and the same 1,500 billable output tokens (**USD 0.004950**). Optimizer input includes the source, current draft, instructions and feedback history; 4,000 is a scenario assumption that must be remeasured as history grows. All roles use the same model price. The graph has no separate extraction call because the generator reads the original source directly.

| Workflow outcome | Input / output tokens | Model calls | Expected API cost |
| --- | --- | --- | --- |
| Generate plus local checks only | 1,800 / 1,500 | 1 | USD 0.004290; inadequate acceptance policy for this task |
| Accept initial draft after judging | 6,300 / 2,100 | 2 | USD 0.007140 |
| Accept after one revision and re-evaluation | 14,800 / 4,200 | 4 | USD 0.014940 |
| Reach third evaluation after two revisions | 23,300 / 6,300 | 6 | USD 0.022740; accepted or human fallback according to score |

The last row is the maximum **call-count path under expected token lengths**, not a hard dollar bound. Longer inputs, maximum-length completions, billing changes or transport retries increase cost. This initial configuration has no automatic provider retries. Human review, observability subscription charges and infrastructure are excluded.

For reference, applying the same assumed visible-token throughput and minimal-thinking overhead to a 1,400-token generation/revision gives an approximate 10,000-26,000 ms per call. First-pass acceptance therefore models approximately **14,000-37,000 ms** end to end; the six-call path models **42,000-111,000 ms**, before other overhead. These are planning ranges, not latency SLOs or measured percentiles. They must be measured for each role on the selected model.

## When to use each mechanism

- **Use local checks alone for an individual check** when correctness is completely specified: required field presence, a score between 1 and 5, a parsed calendar date, allowed currency codes, or arithmetic reconciled to authoritative structured values. Only checks actually implemented provide protection.
- **Skip the judge on local failure.** Record the structural defect, assign a nonpassing control score and send the draft to the optimizer if a round remains. Repeated schema failures still consume evaluation rounds and reach human fallback.
- **Use the judge after local success** for every report in this financial workflow. It compares material claims with the original source and evaluates formatting in prose and professional tone. Do not skip it merely to reduce latency.
- **Use Pydantic after the judge** to reject malformed outputs, missing criteria, out-of-range scores and nonactionable feedback. An invalid response cannot become an acceptance decision.
- **Use human review** at the round cap, or withhold the request when provider failures prevent a valid decision. The hosting service must record failed requests for review rather than treating an exception as approval.

No accuracy percentage is asserted without labeled data. Compare methods using material-error detection recall, false-acceptance rate, false-rejection rate and agreement with independent financial reviewers. Include intentionally schema-valid hallucinations so the benchmark does not confuse structural compliance with factual accuracy.

## Observability evidence needed to validate the estimates

Instrument one root trace per report and child runs for `generator`, local validation, judge invocation/output parsing and `optimizer`. Log report ID, evaluation round, model snapshot, rubric version, source version, input/output tokens, cache usage, request duration and route outcome. Keep these labels on relevant child runs as well as the root.

[LangSmith dashboards](https://docs.langchain.com/langsmith/dashboards) expose trace/LLM latency, token counts, costs and feedback scores. [LangSmith cost tracking](https://docs.langchain.com/langsmith/cost-tracking) explains per-call and aggregate token/cost accounting. Langfuse can provide equivalent tracing if that is the team's existing platform.

Collect at least 100 representative staging reports, with failures and retries retained. Compare p50 and p95/p99 child-run latency with the planning ranges; measure the local span using a monotonic timer. Report root trace latency separately from the sum of model spans, and distinguish cold schema construction from warm validation. Reconcile observed token counts with the pricing formula and confirm all spans have completed before analyzing totals.

Publish first-pass acceptance, revisions per report, fallback rate and material-error rates alongside latency and cost. Lower latency is not a valid improvement if false acceptance rises. Until these traces exist, the ranges above remain estimates; no LangSmith or Langfuse measurements have been fabricated for this submission.
