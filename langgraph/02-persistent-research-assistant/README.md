# Nile Insights Research Assistant

Research questions with a Tavily search tool and demonstrate retained conversation state, thread isolation, and checkpoint history in LangGraph.

**Stack:** LangGraph · ToolNode · MemorySaver · Tavily · Gemini

**Project type:** Runnable CLI with a key-free simulation and optional live model/search integration.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langgraph--02-persistent-research-assistant.zip) · [Portfolio](../../README.md)

## Explore the project

[Research-assistant proposal](docs/nile_insights_proposal.md) · [Original brief](docs/nile_insights_brief.md) · [Implementation](src/research_agent.py)

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
python src/research_agent.py --mock
# Live interactive session, after configuring .env:
python src/research_agent.py --interactive
```

## Expected behavior

The offline run verifies that earlier messages survive a second invocation, a new thread starts independently, and an earlier checkpoint can be read. Interactive mode supports /thread, /state and /quit.

## Verify

```powershell
python src/research_agent.py --mock
```

## Scope and limitations

MemorySaver persists state only within the running process. Restart durability requires a database-backed checkpointer. Thread IDs partition state but do not authenticate users. --mock-search combines a real model with fictional search fixtures.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
