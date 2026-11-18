# NexusAgent: Enterprise RAG Assistant

Answer business questions with read-only SQL, policy retrieval, optional market news, and a combination of recent and retrieved conversation memory.

**Stack:** LangChain Classic · SQLAlchemy · SQLGlot · FAISS · Gemini

**Project type:** Runnable prototype with an offline integration demo and regression tests.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langchain--02-enterprise-rag-agent.zip) · [Portfolio](../../README.md)

## Explore the project

[Pipeline diagram](architecture/pipeline_diagram.md) · [Memory evaluation](reports/memory_strategy_evaluation.md) · [SQL tests](tests/test_agent_config.py)

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
python src/agent_config_stub.py --demo
# Live interactive session, after configuring .env:
python src/agent_config_stub.py --chat
```

## Expected behavior

The offline demo retrieves USD 5,240,000 in seeded Q2 revenue, a three-day remote-work policy, an unavailable-news result, and a remembered follow-up.

## Verify

```powershell
python -m unittest discover -s tests -v
```

## Scope and limitations

SQLite and synthetic enterprise records work by default. SERP news and PostgreSQL are optional integrations requiring separate configuration. Offline embeddings are a lexical baseline; memory is in process and does not survive a restart. Classic LangChain deprecation warnings are expected.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
