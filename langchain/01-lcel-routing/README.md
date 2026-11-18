# LCEL Routing & Structured Reporting

Route financial and news questions to separate validated reporting chains while retrieving both sources concurrently.

**Stack:** LCEL · RunnableParallel · Gemini · Pydantic

**Project type:** Runnable architecture prototype; synthetic retrieval fixtures and real model inference.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langchain--01-lcel-routing.zip) · [Portfolio](../../README.md)

## Explore the project

[Design rationale](design_proposal.md) · [Implementation](workflow_architecture.py)

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
python workflow_architecture.py
python workflow_architecture.py --query "Summarize the FinCorp market news."
```

## Expected behavior

A selected route and a validated JSON financial summary or news report. Unsupported or mixed requests exit with a clarification error.

## Verify

```powershell
python workflow_architecture.py --help
```

## Scope and limitations

The loaders return fictional FinCorp fixtures. Schema validation checks structure and bounds, not factual accuracy. Running the workflow requires your own Gemini access and quota.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
