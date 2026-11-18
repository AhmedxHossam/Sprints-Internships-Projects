# AegisHealth: Medical Triage & Human Review

Demonstrate confidence-based escalation, separate MCP record access, durable checkpoints, validated agent handoffs, and clinician approval gates using synthetic records.

**Stack:** LangGraph · Async SQLite · MCP Streamable HTTP · JSON Schema

**Project type:** Executable architecture blueprint with offline tests and a local authenticated MCP service.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langgraph--04-medical-triage-workflow.zip) · [Portfolio](../../README.md)

## Explore the project

[Architecture handover](architecture_handover.md) · [A2A contract](schemas/a2a_schema.json) · [Regression tests](tests/)

## Setup

Use Python 3.12 or newer; the local verification environment uses Python 3.14. Create a separate environment for this project because module dependencies differ. Run these commands from the folder containing this README:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell blocks activation, run `.\.venv\Scripts\python.exe` in place of `python`.

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Enter your own credentials only for the integrations you use. Existing process environment variables take precedence. Offline checks do not require model API keys. Live calls need internet, provider model access, and available quota; model IDs can be changed in `.env`.

Before running the MCP service, generate **two distinct local secrets** with `python -c "import secrets; print(secrets.token_urlsafe(32))"` (run it twice). Put one in `MCP_RECORDS_TOKEN` and the other in `DOCTOR_REVIEW_TOKEN` in your private `.env`; each must contain at least 32 characters. The tests supply their own test credentials.

## Run

```powershell
# Terminal 1: start the local record service
python -B -m src.mcp_server
# Terminal 2: start a synthetic case
python -B -m src.triage_workflow start --thread-id encounter-01 --confidence 0.80
python -B -m src.triage_workflow status --thread-id encounter-01
```

## Expected behavior

A confidence of 0.80 pauses for escalation review; 0.85 routes to the specialist. Every specialist draft still needs final approval. Reusing the same thread and SQLite file supports recovery.

## Verify

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Human review

Use the current checkpoint ID from `status`:

```powershell
python -B -m src.triage_workflow review --thread-id encounter-01 --checkpoint-id CHECKPOINT_FROM_STATUS --reviewer doctor-local
```

The CLI prompts for the hidden doctor token and a decision. Low-confidence cases require escalation approval and then a second review using the new checkpoint ID. Runtime state stays inside this project's `.local/runtime/`. The handover documents the review contract and limitations; its submission-era folder descriptions predate this portfolio layout.

## Scope and limitations

All supplied patient records are synthetic. This is not a clinically validated service; the specialist validates a handoff rather than diagnosing. Shared local tokens and a trusted single writer are development assumptions.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
