# Financial ReAct Agent & Compliance Reports

Combine policy retrieval, restricted arithmetic, five-turn conversation memory, and validated JSON reports to assess a fictional leverage scenario.

**Stack:** ReAct · FAISS · Pydantic · OutputFixingParser · Gemini

**Project type:** Runnable CLI and self-contained notebook; offline self-check available.

[Download this project as ZIP](https://github.com/AhmedxHossam/Sprints-Internships-Projects/raw/refs/heads/main/downloads/langchain--03-financial-react-agent.zip) · [Portfolio](../../README.md)

## Explore the project

[Notebook](deliverables/notebooks/enterprise_agent.ipynb) · [Sample report](deliverables/outputs/final_report.json) · [Example execution trace](deliverables/outputs/evaluation_traces.md)

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
python financial_agent.py --self-check
# Live interactive session, after configuring .env:
python financial_agent.py --chat
# Evaluate the supplied questions:
python financial_agent.py
```

## Expected behavior

The sample ratios are 45/28 and 52/28. The fictional policy marks ratios at or above 1.5 as NON_COMPLIANT. Runs write final_report.json, evaluation_traces.md and a local JSONL trace.

## Verify

```powershell
python financial_agent.py --self-check
```

## Notebook

Install Jupyter with `python -m pip install jupyterlab ipykernel`, launch `python -m jupyterlab`, and open `deliverables/notebooks/enterprise_agent.ipynb`. Select this project's environment. Notebook cell outputs were cleared for publication; the sample report and Markdown trace remain under `deliverables/outputs/`. Live reruns overwrite those sample outputs.

## Scope and limitations

The calculator accepts arithmetic through an AST allowlist. It is not an unrestricted Python interpreter. Reports use supplied training policies; no financial transaction or remediation is executed. Classic LangChain deprecation warnings are expected.

Original training briefs, synthetic data, and design documents are retained where they help explain the implementation. This portfolio is educational work completed during Sprints training; it is not an official Sprints product.
