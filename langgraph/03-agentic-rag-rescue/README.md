# Lumina Rescue: Bounded Agentic RAG

Retrieve synthetic local medical research and optional web snippets, critique whether evidence answers the question, and refine retrieval within a deterministic retry budget.

**Stack:** LangGraph · LangChain tool calling · DuckDuckGo · Gemini / OpenAI

**Project type:** Runnable prototype with offline scenarios, regression tests, and type-check configuration.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langgraph--03-agentic-rag-rescue.zip) · [Portfolio](../../README.md)

## Explore the project

[Architecture](lumina_rescue/architecture_proposal.md) · [Incident report](lumina_rescue/docs/cto_incident_report.md) · [Routing](lumina_rescue/loop_logic.py)

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
python lumina_rescue/state_graph_config.py --mode demo --trace
# Optional live model and web retrieval:
python lumina_rescue/state_graph_config.py --mode live --provider gemini --question "Find research on CBT and depression."
```

## Expected behavior

An attributed answer with termination status sufficient, retry_limit, or fatal_error. There is one initial attempt and at most three retries; retry_limit can return exit 0, so inspect the status.

## Verify

```powershell
python lumina_rescue/loop_logic.py
python -m unittest discover -s tests -v
python -m mypy --strict
```

## Scope and limitations

Medical records are synthetic, local retrieval is lexical, and web evidence consists of unverified snippets. State lasts for one run. This is a software research prototype without clinical validation.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
