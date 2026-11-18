from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from uuid import uuid4

import jsonschema
from langgraph.types import Command
import pytest
from pydantic import ValidationError

from workflow_graph import AgentBackend, grounding_issues, numeric_and_date_tokens
from workflow_graph import AuditLog
from workflow_graph import ROOT, Settings
from workflow_graph import load_market_context
from workflow_graph import compact_messages
from workflow_graph import RAGService
from state_schema import A2AMessage, FinancialBrief, FXQuote, RetrievalAssessment, ReviewVerdict
from workflow_graph import initial_state, open_workflow, thread_config


@pytest.fixture
def settings(tmp_path):
    return Settings(runtime_dir=tmp_path / "runtime", inject_demo_defect=True)


@pytest.mark.asyncio
async def test_reject_revise_approve_then_wait_and_validate_wire_schema(settings):
    async with open_workflow(settings) as (graph, services):
        config = thread_config("self-correction")
        output = await graph.ainvoke(initial_state("Prepare a training brief", "2026-11-16"), config)
        snapshot = await graph.aget_state(config)
        assert output["status"] == "pre_publish"
        assert snapshot.next == ("pre_publish",)
        assert output.get("__interrupt__")
        assert [review["is_approved"] for review in output["review_history"]] == [False, True]
        assert "UNSUPPORTED-DEMO-CITATION" not in output["draft_report"]
        assert "human_decision" not in snapshot.values
        schema = json.loads((ROOT / "schemas" / "a2a_message_schema.json").read_text())
        for message in output["a2a_messages"]:
            A2AMessage.model_validate(message)
            jsonschema.validate(message, schema)
        assert services["audit"].verify()
        events = services["audit"].events("self-correction")
        retrievals = output["retrieval_audits"]
        assert len([e for e in events if e["kind"] == "retrieval_start"]) == len(retrievals)
        assert len([e for e in events if e["kind"] == "retrieval_end"]) == len(retrievals)
        assert all(record["trace_url"] is None for record in retrievals)
        assert all(record["source_ids"] for record in retrievals)
        assert len(output["scratchpad_memory"]) <= 12
        assert output["fx_quote"]["is_mock"] is True
        # A valid quote must not make an unrelated invented number acceptable.
        corrupted = FinancialBrief.model_validate(output["brief"])
        corrupted.facts[0].text = "The deposit rate is 99.99%."
        assert any("numerical" in issue for issue in grounding_issues(corrupted, output["evidence"]))
        # Approval remains evidence-based even after many iterations.
        corrupted.facts[0].source_id = "INVENTED"
        assert not (await services["backend"].review(corrupted, output["evidence"])).is_approved


@pytest.mark.asyncio
@pytest.mark.parametrize("action,expected", [("publish", "published"), ("reject", "rejected")])
async def test_resume_after_new_graph_instance(settings, action, expected):
    config = thread_config("reopen-" + action)
    async with open_workflow(settings) as (graph, _):
        await graph.ainvoke(initial_state("Prepare brief", "2026-11-16"), config)
        before = await graph.aget_state(config)
        digest = before.values["approved_draft_sha256"]
        retrieval_count = len(before.values["retrieval_audits"])
    # Connection, checkpointer, graph and MCP server are all reconstructed.
    async with open_workflow(settings) as (graph, services):
        result = await graph.ainvoke(Command(resume={"action": action,
            "reviewer": "Integration test director", "draft_sha256": digest}), config)
        assert result["status"] == expected
        assert not (await graph.aget_state(config)).next
        assert len(result["retrieval_audits"]) == retrieval_count
        assert result["publication"]["delivered"] is False
        assert result["publication"]["authorized"] is (action == "publish")
        assert services["audit"].verify()


@pytest.mark.asyncio
async def test_invalid_command_and_wrong_hash_remain_interrupted(settings):
    async with open_workflow(settings) as (graph, _):
        config = thread_config("invalid-human-input")
        await graph.ainvoke(initial_state("Prepare brief", "2026-11-16"), config)
        for decision in ["publish", {"action": "delete", "reviewer": "test", "draft_sha256": "0" * 64},
                         {"action": "publish", "reviewer": "test", "draft_sha256": "0" * 64}]:
            result = await graph.ainvoke(Command(resume=decision), config)
            assert result["status"] == "pre_publish"
            assert result.get("__interrupt__")
            assert "publication" not in result
        state = (await graph.aget_state(config)).values
        result = await graph.ainvoke(Command(resume={"action": "reject", "reviewer": "test",
            "draft_sha256": state["approved_draft_sha256"]}), config)
        assert result["status"] == "rejected"


@pytest.mark.asyncio
async def test_exhausted_review_budget_never_reaches_publication(settings):
    settings = replace(settings, max_iterations=3, force_rejection=True)
    async with open_workflow(settings) as (graph, _):
        result = await graph.ainvoke(initial_state("Prepare brief", "2026-11-16"), thread_config("exhausted"))
        assert result["status"] == "review_failed"
        assert result["iteration"] == 3
        assert not result["approval_status"]
        assert "__interrupt__" not in result
        assert "publication" not in result
        drafts = [m for m in result["a2a_messages"] if m["sender"] == "analyst"]
        assert "UNSUPPORTED-DEMO-CITATION" in drafts[0]["draft_report"]
        assert all("UNSUPPORTED-DEMO-CITATION" not in m["draft_report"] for m in drafts[1:])


@pytest.mark.asyncio
async def test_tool_failure_stops_before_approval(settings, monkeypatch):
    async with open_workflow(settings) as (graph, services):
        async def fail(*args, **kwargs):
            raise RuntimeError("Simulated unavailable MCP source")
        monkeypatch.setattr(type(services["fx_tool"]), "ainvoke", fail)
        with pytest.raises(RuntimeError, match="unavailable MCP"):
            await graph.ainvoke(initial_state("Prepare brief", "2026-11-16"), thread_config("tool-failure"))
        state = (await graph.aget_state(thread_config("tool-failure"))).values
        assert not state["approval_status"]
        assert "publication" not in state


@pytest.mark.asyncio
async def test_insufficient_rag_context_fails_after_bounded_followup(settings, monkeypatch):
    async with open_workflow(settings) as (graph, services):
        async def insufficient(*args, **kwargs):
            return RetrievalAssessment(sufficient=False, follow_up_query="Missing evidence test")
        monkeypatch.setattr(services["backend"], "assess_retrieval", insufficient)
        config = thread_config("insufficient-rag")
        with pytest.raises(RuntimeError, match="remains insufficient"):
            await graph.ainvoke(initial_state("Prepare brief", "2026-11-16"), config)
        events = services["audit"].events("insufficient-rag")
        assert len([event for event in events if event["kind"] == "retrieval_end"]) == 3
        assert not any(event["kind"] == "mcp_tool_call" for event in events)
        assert "publication" not in (await graph.aget_state(config)).values


def test_pydantic_rejects_coercion_and_unknown_fields():
    with pytest.raises(ValidationError):
        ReviewVerdict(is_approved="false", feedback_notes="No")
    with pytest.raises(ValidationError):
        ReviewVerdict(is_approved=True, feedback_notes="OK", extra_field="unexpected")
    with pytest.raises(ValidationError):
        A2AMessage.model_validate({"sender": "analyst"})
    with pytest.raises(ValidationError):
        FXQuote(base="USD", target="EGP", rate=float("nan"), inverse_rate=0.1,
            unit="EGP per 1 USD", source="TRAINING_MOCK_NOT_CBE", is_mock=True,
            last_updated="2026-11-16T00:00:00")


def test_egx_asof_prices_and_no_future_rows():
    context, evidence = load_market_context(ROOT / "data" / "egx_daily_summaries.csv", "2024-03-06")
    assert context["latest_session"] == "2024-03-06"
    assert all(row["Date"] <= "2024-03-06" for row in context["rows"])
    comi = next(row for row in context["rows"] if row["Ticker"] == "COMI" and row["Date"] == "2024-03-06")
    assert comi["open_to_close_pct"] == 11.41
    assert all("open-to-close" in item["content"] for item in evidence)
    with pytest.raises(ValueError, match="No EGX sessions"):
        load_market_context(ROOT / "data" / "egx_daily_summaries.csv", "2020-01-01")


def test_invalid_market_prices_are_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Ticker,Open,Close,Volume,Sector\n2024-03-06,COMI,0,80,1,Banking\n")
    with pytest.raises(ValueError, match="Invalid prices"):
        load_market_context(path, "2024-03-06")


@pytest.mark.asyncio
async def test_sliding_window_and_summary_are_bounded(settings):
    backend = AgentBackend(settings)
    messages = [{"role": "user", "content": f"Reviewer feedback {i}: " + "evidence " * 200} for i in range(30)]
    kept, summary = await compact_messages(messages, "Previous summary", settings, backend)
    assert len(kept) <= settings.window_messages
    assert sum(len(m["content"]) for m in kept) <= settings.context_chars
    assert 0 < len(summary) <= settings.summary_chars
    assert "feedback 29" in kept[-1]["content"]


def test_chroma_ingestion_is_idempotent(settings):
    audit = AuditLog(settings.runtime_dir / "audit.sqlite")
    first = RAGService(settings, audit)
    second = RAGService(settings, audit)
    assert first.collection_name == second.collection_name
    assert len(second.store.get(include=[])["ids"]) == first.chunk_count


def test_audit_detects_changed_records(tmp_path):
    log = AuditLog(tmp_path / "audit.sqlite")
    log.append("test", "example", {"source": "original"})
    assert log.verify()
    with sqlite3.connect(log.path) as db:
        db.execute("UPDATE events SET payload=? WHERE id=1", ('{"source":"changed"}',))
    assert not log.verify()


def test_missing_credentials_fail_explicitly(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        replace(settings, mode="live", provider="openai")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        replace(settings, mode="live", provider="gemini")
    with pytest.raises(ValueError, match="Langfuse"):
        replace(settings, audit_mode="langfuse")


def test_true_process_restart_persists_interrupt_and_resumes(tmp_path):
    runtime = tmp_path / "process-restart"
    thread = "process-" + uuid4().hex
    common = [sys.executable, str(ROOT / "src" / "workflow_graph.py")]
    options = ["--thread-id", thread, "--runtime-dir", str(runtime),
               "--mode", "demo", "--audit-mode", "local", "--embedding-mode", "local"]
    env = os.environ | {"PYTHONUTF8": "1", "ANONYMIZED_TELEMETRY": "False"}
    def launch(command, extra):
        process = subprocess.run(common + [command] + options + extra,
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout.strip().splitlines()[-1])
    started = launch("start", ["--inject-demo-defect", "--as-of", "2026-11-16"])
    assert started["state"]["status"] == "pre_publish"
    digest = started["state"]["approved_draft_sha256"]
    resumed = launch("resume", ["--action", "publish", "--reviewer", "Simulated test director",
                                 "--draft-sha256", digest])
    assert started["process_id"] != resumed["process_id"]
    assert resumed["state"]["status"] == "published"
    assert resumed["state"]["publication"]["delivered"] is False
    assert resumed["state"]["iteration"] == started["state"]["iteration"]
    assert resumed["audit_chain_valid"]


def test_gemini_env_discovery_and_explicit_overrides(tmp_path, monkeypatch):
    import workflow_graph as config_module
    project = tmp_path / "sprint"
    deliverables = project / "starter" / "deliverables"
    deliverables.mkdir(parents=True)
    (project / ".env").write_text("GOOGLE_API_KEY=wrong-sibling-key\nGEMINI_MODEL=wrong-sibling-model\n")
    (deliverables / ".env").write_text("GOOGLE_API_KEY=unit-test-placeholder\nGEMINI_MODEL=test-project-model\n")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", project)
    monkeypatch.setattr(config_module, "ROOT", deliverables)
    for key in ("APP_MODE", "LLM_PROVIDER", "EMBEDDING_MODE", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    live = Settings.from_env()
    assert live.provider == "gemini" and live.mode == "live"
    assert live.model == "test-project-model"
    assert live.embedding_mode == "gemini"
    assert live.embedding_model == "gemini-embedding-001"
    offline = Settings.from_env(mode="demo")
    assert offline.mode == "demo" and offline.embedding_mode == "local"
    monkeypatch.setenv("GEMINI_MODEL", "test-environment-model")
    assert Settings.from_env().model == "test-environment-model"


def test_gemini_api_key_fallback(settings, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder")
    configured = replace(settings, mode="live", provider="gemini")
    assert configured.mode == "live"


def test_date_spellings_are_equivalent_but_different_values_are_not():
    numbers, dates = numeric_and_date_tokens("On March 7, 2024, change -3.01%; volume 4,100,000.")
    iso_numbers, iso_dates = numeric_and_date_tokens("2024-03-07 change -3.01%; volume 4100000.")
    assert dates == iso_dates == {"2024-03-07"}
    assert numbers == iso_numbers
    assert numeric_and_date_tokens("change +3.01%")[0] != numeric_and_date_tokens("change -3.01%")[0]
    assert numeric_and_date_tokens("March 8, 2024")[1] != iso_dates


def test_date_metadata_can_ground_date_but_not_a_new_policy_value():
    from state_schema import FactClaim
    evidence = [{"source_id": "policy", "content": "Deposit rate: 19.25%.",
                 "source_date": "2023-11-12", "kind": "cbe_training_document"}]
    brief = FinancialBrief(title="Test", facts=[FactClaim(
        text="The training document dated November 12, 2023 states a deposit rate of 19.25%.",
        source_id="policy", evidence_quote="Deposit rate: 19.25%.")],
        analysis=[], caveats=["Historical mock unverified human review required."])
    issues = grounding_issues(brief, evidence)
    assert not any("numerical" in issue or "Unsupported date" in issue for issue in issues)
    brief.facts[0].text = "The rate is 99.99% on November 13, 2023."
    issues = grounding_issues(brief, evidence)
    assert any("numerical" in issue for issue in issues)
    assert any("Unsupported date" in issue for issue in issues)


@pytest.mark.asyncio
async def test_langfuse_readback_requires_matching_query_and_all_sources():
    from types import SimpleNamespace
    from workflow_graph import verify_retrieval_traces
    record = {"trace_url": "https://cloud.langfuse.com/project/test/traces/abc123",
              "retrieval_id": "retrieval-test", "query": "Policy rate", "source_ids": ["CBE-001"]}
    observation = SimpleNamespace(name="cbe_context_retrieval", input='{"query":"Policy rate"}',
                                  output='[{"metadata":{"source_id":"CBE-001"}}]')
    api = SimpleNamespace(observations=SimpleNamespace(
        get_many=lambda **kwargs: SimpleNamespace(data=[observation])))
    client = SimpleNamespace(api=api)
    verified = await verify_retrieval_traces(client, [record], attempts=1, pause_seconds=0)
    assert verified[0]["read_back_verified"]
    observation.output = '[]'
    with pytest.raises(RuntimeError, match="verification failed"):
        await verify_retrieval_traces(client, [record], attempts=1, pause_seconds=0)
    observation.output = '[{"source_id":"CBE-001"}]'
    observation.input = '"A different query"'
    with pytest.raises(RuntimeError, match="verification failed"):
        await verify_retrieval_traces(client, [record], attempts=1, pause_seconds=0)


def test_promissory_language_does_not_reject_negated_disclaimers():
    from workflow_graph import has_promissory_advice
    assert not has_promissory_advice("This dataset should not be used as a basis for buy/sell orders or guaranteed returns.")
    assert not has_promissory_advice("This report does not promise guaranteed returns.")
    assert has_promissory_advice("This strategy provides guaranteed returns.")
    assert has_promissory_advice("No advice is provided. Guaranteed profits are available.")
    assert has_promissory_advice("There are no guaranteed returns, but buy now.")
