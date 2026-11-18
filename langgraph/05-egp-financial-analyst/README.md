# CapitalTrust: EGP Financial Analyst

Build an evidence-grounded analyst/reviewer loop with RAG, MCP foreign-exchange fixtures, historical market data, an audit trail, durable checkpoints, and mandatory human review.

**Stack:** LangGraph · Chroma · MCP · SQLite · Gemini · Langfuse

**Project type:** Capstone CLI and notebook; demo mode works without model or observability credentials.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langgraph--05-egp-financial-analyst.zip) · [Portfolio](../../README.md)

## Explore the project

[Execution notebook](notebooks/system_execution.ipynb) · [State and wire models](src/state_schema.py) · [Workflow tests](tests/test_workflow.py)

## Setup

Use Python 3.12 or newer; the local verification environment uses Python 3.14. Create a separate environment for this project because module dependencies differ. Run these commands from the folder containing this README:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell blocks activation, run `.\.venv\Scripts\python.exe` in place of `python`.

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Enter your own credentials only for the integrations you use. Existing process environment variables take precedence. Offline checks do not require model API keys. Live calls need internet, provider model access, and available quota; model IDs can be changed in `.env`.

## Run

```powershell
python src/workflow_graph.py start --thread-id portfolio-001 --mode demo --audit-mode local --embedding-mode local --inject-demo-defect --as-of 2026-11-16
python src/workflow_graph.py inspect --thread-id portfolio-001 --mode demo --audit-mode local --embedding-mode local
```

## Expected behavior

The injected unsupported claim is rejected, the analyst revises it, and the approved draft pauses for a human decision. The inspect command exposes the exact draft hash required for resume.

## Verify

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Resume after review

Read `draft_report` and the approved draft hash from `inspect`, then use the same thread and runtime directory:

```powershell
python src/workflow_graph.py resume --thread-id portfolio-001 --mode demo --audit-mode local --embedding-mode local --action publish --reviewer "Your Name" --draft-sha256 "HASH_FROM_INSPECT"
```

Use `--action reject` to reject. A reviewer name alone does not authenticate a person. Keep the runtime directory and database to recover after restarting Python. Use a fresh thread ID when starting a separate run.

For live inference, set `APP_MODE=live`, `EMBEDDING_MODE=gemini`, and your own `GOOGLE_API_KEY`. Langfuse additionally requires both keys and `AUDIT_MODE=langfuse`. The notebook uses the selected environment and starts a fresh run; install `jupyterlab` and open `notebooks/system_execution.ipynb` from the project root or `notebooks/`. Notebook cell outputs were cleared before publication.

## Scope and limitations

FX remains synthetic in both demo and live modes. EGX records end on 2024-03-07. The CBE-style document is supplied training material, not independently authenticated current regulation. publish records authorization only; it does not send reports or place trades.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
