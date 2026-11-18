# Verification notes

Portfolio export checked on **18 November 2026** on Windows using the existing Python 3.14 project environments. Source was tested from the new portfolio folders, independently of the original course folder names.

## Executed checks

- **NexusAgent:** 10 offline regression tests passed. Covers SQL restrictions, database write denial, retained multi-source evidence, and memory behavior.
- **Financial ReAct Agent:** built-in self-check passed for five-turn memory, parser repair, valid-output handling, threshold equality, NOT_ASSESSED output, and restricted arithmetic.
- **Lumina Rescue:** 33 offline regression tests passed, and strict mypy reported no issues in three source files. Its demo completed after two attempts with `Termination: sufficient` using explicitly simulated model/search evidence.
- **AegisHealth:** 51 tests passed, including MCP authorization and protocol responses, threshold routing, approval gates, record changes, and persistent recovery.
- **CapitalTrust:** 21 tests passed against the final consolidated source, including rejection/revision, wire validation, checkpoint recovery across Python processes, bound human decisions, and audit checks. Test imports and the CLI path were adapted from the retained development suite to the portfolio's final three-module implementation. Configuration now loads the project's own `.env`.
- **Nile Insights:** `--mock` passed the same-thread memory, independent-thread, and historical-checkpoint checks.
- **LCEL Routing and Financial Report Evaluator:** CLI help/import checks passed. Their live model generation was not rerun as part of publication.

The four regression suites total **115 passing tests**. CLI self-checks and demo assertions are additional checks and are not included in that count. Expected deprecation warnings from classic LangChain APIs and some dependencies remain visible.

## Publication hygiene

- Nine curated, standalone project archives are built from the source with deterministic ZIP metadata.
- Private `.env` files, installed environments, duplicate submission archives, generated databases, local execution logs, and workstation screenshots are excluded.
- Notebook cell outputs and execution counts were cleared. Explicit sample reports and public operational traces remain where useful.
- The export was checked against seven distinct private credential/configuration values found in the original local environment files; no match was found. Values were not printed or copied.
- `python tools/check_repository.py` checks Python syntax, JSON/notebook structure, local Markdown links, common credential patterns, archive integrity, and byte-for-byte archive/source consistency.

## Limits of this verification

These checks use the existing installed local environments; they do not establish that every dependency can be freshly installed on every operating system or Python version. Each project retains its own dependency manifest because versions differ across modules. The repository-wide GitHub Actions workflow runs dependency-free publication checks rather than the complete project test suites.

No live provider call was made for publication. Current Gemini/OpenAI/Tavily/SERP/Langfuse account access, quota, and model availability must be verified with your own credentials. Live external news and production PostgreSQL were not integration-tested here. Historical screenshots and prior live-run claims in course design documents describe those earlier executions and are not new publication checks.

Test success demonstrates the covered software behavior. It does not validate medical accuracy, financial recommendations, regulatory authenticity, model truthfulness, or production readiness.
