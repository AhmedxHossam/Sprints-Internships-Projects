# Financial Report Evaluator–Optimizer

Generate a financial report, evaluate it against source notes and a structured rubric, and revise it within a fixed budget before accepting it or requesting human review.

**Stack:** LangGraph · Gemini · Structured evaluation · Pydantic

**Project type:** Executable workflow with architecture and trade-off documents.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langgraph--01-financial-report-evaluator.zip) · [Portfolio](../../README.md)

## Explore the project

[Architecture](docs/architecture_proposal.md) · [Feedback-loop specification](docs/feedback_loop_spec.md) · [Trade-offs](docs/tradeoff_analysis.md)

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
python src/graph_schema.py
# Or read your own UTF-8 source notes:
python src/graph_schema.py --source-file notes.txt
```

## Expected behavior

Paste source notes and enter /done on a separate line. Each request saves its source notes, report and evaluation under run_output/. Exit 0 means accepted; 2 means pending human review; 1 means an error.

## Verify

```powershell
python src/graph_schema.py --help
```

## Scope and limitations

Acceptance requires all three rubric scores to reach 4/5. At most three evaluation rounds are allowed. Judge scores do not guarantee accuracy. src/current_sequential_chain.py is the original failing reference; use graph_schema.py as the entry point.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
