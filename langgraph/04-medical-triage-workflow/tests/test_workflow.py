import json
import logging

import pytest
from langgraph.types import Command

from src.triage_workflow import DemoAssessor, RecordsFailure, validate_handoff
from src.mcp_server import JsonRecordsRepository
from src.triage_workflow import ReviewAuthority, open_workflow, route_after_triage

TOKEN = "doctor-test-credential-" + "x" * 32
PATIENT = {"patient_id": "P-1001", "symptoms": ["Synthetic symptom"]}


class Records:
    async def fetch(self, patient_id):
        raw = await JsonRecordsRepository().fetch(patient_id)
        return {"patient_id": patient_id, "age": raw["age"],
                "chronic_conditions": raw["chronic_conditions"], "recent_visits": []}


def authority():
    return ReviewAuthority(TOKEN, frozenset({"doctor-test"}))


def options(score=0.8, records=None, assessor=None):
    return {"records": records or Records(), "assessor": assessor or DemoAssessor(score),
            "authority": authority()}


async def review(service, thread="case", decision="approved", priority="Medium", **kwargs):
    snapshot = await service.snapshot(thread)
    return await service.review(
        thread, checkpoint_id=snapshot.config["configurable"]["checkpoint_id"],
        reviewer="doctor-test", token=TOKEN,
        decision={"decision": decision, "triage_priority": priority, "note": "Synthetic review"},
        **kwargs,
    )


@pytest.mark.parametrize("score,expected", [(0.0, "doctor_review"), (0.849999, "doctor_review"),
                                            (0.85, "specialist_agent"), (1.0, "specialist_agent")])
def test_exact_routing(score, expected):
    assert route_after_triage({"confidence_score": score}) == expected


@pytest.mark.parametrize("score", [None, True, "0.95", -1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_fails_closed(score):
    assert route_after_triage({"confidence_score": score}) == "doctor_review"


async def test_low_confidence_pauses_and_requires_two_reviews(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options()) as service:
        state = await service.start("case", PATIENT)
        snapshot = await service.snapshot("case")
        assert snapshot.next == ("doctor_review",)
        assert not any(task.interrupts for task in snapshot.tasks)  # static breakpoint
        assert "specialist_handoff" not in state
        assert [m.name for m in state["messages"]] == ["triage_agent"]
        state = await review(service, priority="High")
        assert state["specialist_handoff"]["triage_priority"] == "High"
        assert state["status"] == "awaiting_final_review"
        assert "released_payload" not in state
        assert (await service.snapshot("case")).next == ("final_review",)
        state = await review(service, priority="High")
        assert state["status"] == "approved"
        assert state["released_payload"] == state["specialist_handoff"]
        assert [m.name for m in state["messages"]] == [
            "triage_agent", "doctor_review", "specialist_agent", "final_review"
        ]
        assert (await service.snapshot("case")).next == ()
        assert TOKEN not in json.dumps(state, default=str)


async def test_high_confidence_still_needs_final_human_signoff(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options(0.85)) as service:
        state = await service.start("case", PATIENT)
        assert state["status"] == "awaiting_final_review"
        assert "doctor_review" not in state and "released_payload" not in state
        state = await review(service)
        assert state["status"] == "approved"


@pytest.mark.parametrize("score", [0.8, 0.95])
async def test_rejection_never_releases(tmp_path, score):
    async with open_workflow(tmp_path / "case.sqlite", **options(score)) as service:
        await service.start("case", PATIENT)
        state = await review(service, decision="rejected")
        assert state["status"] == "rejected"
        assert not state.get("released_payload")
        if score < 0.85:
            assert "specialist_handoff" not in state


async def test_persistent_checkpoint_can_resume_after_reopening(tmp_path):
    db = tmp_path / "persistent.sqlite"
    async with open_workflow(db, **options()) as service:
        await service.start("case", PATIENT)
        old = await service.snapshot("case")
    async with open_workflow(db, **options()) as restarted:
        snapshot = await restarted.snapshot("case")
        assert snapshot.next == ("doctor_review",)
        assert snapshot.config == old.config
        assert (await review(restarted))["status"] == "awaiting_final_review"
    async with open_workflow(db, **options()) as restarted:
        assert (await review(restarted))["status"] == "approved"


async def test_bare_resume_does_not_impersonate_a_doctor(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options()) as service:
        await service.start("case", PATIENT)
        state = await service.graph.ainvoke(None, service.config("case"))
        assert "specialist_handoff" not in state
        snapshot = await service.snapshot("case")
        assert any(task.interrupts for task in snapshot.tasks)
        assert (await review(service))["status"] == "awaiting_final_review"


async def test_forged_dynamic_review_cannot_release(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options(0.95)) as service:
        await service.start("case", PATIENT)
        with pytest.raises(PermissionError):
            await service.graph.ainvoke(Command(resume={"decision": "approved"}), service.config("case"))
        assert not (await service.snapshot("case")).values.get("released_payload")


async def test_wrong_identity_stale_review_and_reused_thread_are_rejected(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options()) as service:
        await service.start("case", PATIENT)
        snapshot = await service.snapshot("case")
        base = {"checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                "reviewer": "doctor-test", "token": TOKEN,
                "decision": {"decision": "approved", "triage_priority": "Medium", "note": "review"}}
        with pytest.raises(PermissionError):
            await service.review("case", **{**base, "token": "wrong"})
        with pytest.raises(PermissionError):
            await service.review("case", **{**base, "reviewer": "unknown"})
        with pytest.raises(ValueError, match="Stale"):
            await service.review("case", **{**base, "checkpoint_id": "old"})
        with pytest.raises(ValueError, match="already exists"):
            await service.start("case", PATIENT)
        await service.review("case", **base)
        with pytest.raises(ValueError, match="Stale"):
            await service.review("case", **base)


async def test_signature_is_bound_to_case_contents(tmp_path):
    auth = authority()
    async with open_workflow(tmp_path / "case.sqlite", **{**options(), "authority": auth}) as service:
        await service.start("case", PATIENT)
        snapshot = await service.snapshot("case")
        signed = auth.sign(state=snapshot.values, phase="escalation", checkpoint_id="checkpoint", thread_id="case",
                           reviewer="doctor-test", token=TOKEN,
                           decision={"decision": "approved", "triage_priority": "High", "note": "review"})
        assert auth.verify(signed, snapshot.values, "escalation", "case")
        assert not auth.verify(signed, snapshot.values, "escalation", "another-case")
        changed = {**snapshot.values, "symptoms": ["Changed symptom"]}
        assert not auth.verify(signed, changed, "escalation", "case")
        assert not auth.verify({**signed, "decision": "rejected"}, snapshot.values, "escalation", "case")


class FailedRecords:
    async def fetch(self, patient_id):
        raise RecordsFailure("UNAVAILABLE")


class InvalidAssessor:
    async def assess(self, symptoms, records):
        return {"triage_priority": "Medium", "confidence_score": float("nan")}


@pytest.mark.parametrize("dependencies", [{"records": FailedRecords()}, {"assessor": InvalidAssessor()}])
async def test_dependency_failure_blocks_specialist_even_after_approval(tmp_path, dependencies):
    async with open_workflow(tmp_path / "case.sqlite", **options(**dependencies)) as service:
        state = await service.start("case", PATIENT)
        assert state["confidence_score"] == 0 and not state["assessment_valid"]
        assert (await service.snapshot("case")).next == ("doctor_review",)
        state = await review(service)
        assert state["status"] == "blocked"
        assert "specialist_handoff" not in state


async def test_transition_logs_exclude_patient_data_and_credentials(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="triage.workflow")
    async with open_workflow(tmp_path / "case.sqlite", **options()) as service:
        await service.start("case", PATIENT)
        await review(service)
        await review(service)
    text = caplog.text
    for marker in ("from=triage_agent to=doctor_review", "from=doctor_review to=specialist_agent",
                   "from=specialist_agent to=final_review", "from=final_review to=END"):
        assert marker in text
    for secret in (PATIENT["patient_id"], PATIENT["symptoms"][0], TOKEN):
        assert secret not in text


async def test_changed_records_cannot_be_approved(tmp_path):
    class ChangingRecords(Records):
        changed = False

        async def fetch(self, patient_id):
            record = await super().fetch(patient_id)
            if self.changed:
                record["age"] += 1
            return record

    records = ChangingRecords()
    async with open_workflow(tmp_path / "case.sqlite", **options(records=records)) as service:
        await service.start("case", PATIENT)
        records.changed = True
        with pytest.raises(ValueError, match="records changed"):
            await review(service)
        assert "specialist_handoff" not in (await service.snapshot("case")).values
        assert (await review(service, decision="rejected"))["status"] == "rejected"


async def test_final_priority_cannot_change_without_reassessment(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options(0.95)) as service:
        await service.start("case", PATIENT)
        with pytest.raises(ValueError, match="silently change"):
            await review(service, priority="High")
        assert not (await service.snapshot("case")).values.get("released_payload")


async def test_two_threads_keep_their_own_messages(tmp_path):
    async with open_workflow(tmp_path / "case.sqlite", **options()) as service:
        await service.start("first", PATIENT)
        await service.start("second", {**PATIENT, "symptoms": ["Another synthetic symptom"]})
        await review(service, thread="first")
        first = (await service.snapshot("first")).values
        second = (await service.snapshot("second")).values
        assert len(first["messages"]) == 3 and len(second["messages"]) == 1
        assert first["symptoms"] != second["symptoms"]


async def test_external_payload_tracing_is_not_silently_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ValueError, match="External payload tracing"):
        async with open_workflow(tmp_path / "case.sqlite", **options()):
            pass


@pytest.mark.parametrize("payload", [
    {}, {"patient_id": "P-1001", "symptoms": [], "triage_priority": "Low"},
    {**PATIENT, "triage_priority": "Urgent"}, {**PATIENT, "triage_priority": "low"},
    {**PATIENT, "triage_priority": "Low", "diagnosis": "invented"},
    {**PATIENT, "triage_priority": "Low", "symptoms": [42]},
])
def test_a2a_rejects_invalid_payloads(payload):
    with pytest.raises(ValueError):
        validate_handoff(payload)


def test_optional_escalation_reason():
    validate_handoff({**PATIENT, "triage_priority": "Medium"})
    validate_handoff({**PATIENT, "triage_priority": "Medium", "escalation_reason": "Low confidence"})
