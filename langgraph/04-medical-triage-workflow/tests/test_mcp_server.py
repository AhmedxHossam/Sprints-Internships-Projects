import json

import pytest
from starlette.testclient import TestClient
from jsonschema import Draft202012Validator

from src.triage_workflow import MCPRecordsClient
from src.mcp_server import JsonRecordsRepository, create_app, TOOL_INPUT_SCHEMA, TOOL_RESULT_SCHEMA

TOKEN = "records-test-credential-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN,
           "Accept": "application/json, text/event-stream"}


@pytest.fixture
def client():
    app = create_app(token=TOKEN, allowed_patient_ids=frozenset({"P-1001", "P-9999"}))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


def call(client, arguments, tool="get_medical_records"):
    response = client.post("/mcp/", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    })
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0" and data["id"] == 42
    result = data["result"]
    Draft202012Validator(TOOL_RESULT_SCHEMA).validate(result["structuredContent"])
    return result


def test_mcp_initialization_discovery_and_fetch(client):
    response = client.post("/mcp/", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    })
    assert "tools" in response.json()["result"]["capabilities"]
    response = client.post("/mcp/", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/list",
    })
    tool = response.json()["result"]["tools"][0]
    assert tool["name"] == "get_medical_records"
    assert tool["inputSchema"] == TOOL_INPUT_SCHEMA
    result = call(client, {"patient_id": "P-1001"})
    assert not result["isError"]
    record = result["structuredContent"]
    assert record["patient_id"] == "P-1001"
    assert "first_name" not in record and "last_name" not in record
    assert "attending_physician" not in record["recent_visits"][0]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("arguments", [{}, {"patient_id": None}, {"patient_id": 1001},
    {"patient_id": ""}, {"patient_id": "../secret"}, {"patient_id": "P-1001", "extra": True}])
def test_missing_or_invalid_arguments_are_structured_errors(client, arguments):
    result = call(client, arguments)
    assert result["isError"]
    assert result["structuredContent"]["error"]["code"] == -32602
    assert result["structuredContent"]["error"]["data"]["kind"] == "INVALID_PARAMS"


@pytest.mark.parametrize("patient_id,code,kind", [
    ("P-1002", -32003, "FORBIDDEN"), ("P-9999", -32004, "NOT_FOUND")])
def test_patient_acl_and_missing_record(client, patient_id, code, kind):
    result = call(client, {"patient_id": patient_id})
    assert result["isError"]
    assert result["structuredContent"]["error"]["code"] == code
    assert result["structuredContent"]["error"]["data"]["kind"] == kind


def test_authentication_origin_and_host_checks(client):
    assert client.post("/mcp/", json={}).status_code == 401
    assert client.post("/mcp/", headers={**HEADERS, "Authorization": "Bearer wrong"}, json={}).status_code == 401
    assert client.post("/mcp/", headers={**HEADERS, "Origin": "https://evil.invalid"}, json={}).status_code == 403
    assert client.post("/mcp/", headers={**HEADERS, "Host": "evil.invalid"}, json={}).status_code == 421


def test_unknown_tool(client):
    result = call(client, {}, tool="unknown")
    assert result["isError"] and result["structuredContent"]["error"]["code"] == -32601


def test_malformed_json_returns_protocol_error(client):
    response = client.post("/mcp/", headers={**HEADERS, "Content-Type": "application/json"}, content="{")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


class BrokenRepository:
    async def fetch(self, patient_id):
        raise RuntimeError("PRIVATE database connection secret")


def test_backend_error_is_sanitized():
    app = create_app(token=TOKEN, allowed_patient_ids=frozenset({"P-1001"}), repository=BrokenRepository())
    with TestClient(app, base_url="http://127.0.0.1") as client:
        result = call(client, {"patient_id": "P-1001"})
    assert result["isError"]
    assert result["structuredContent"]["error"]["data"]["kind"] == "UNAVAILABLE"
    assert "PRIVATE" not in json.dumps(result)


def test_rate_limit_and_insecure_configuration():
    app = create_app(token=TOKEN, allowed_patient_ids=frozenset({"P-1001"}), requests_per_minute=1)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        call(client, {"patient_id": "P-1001"})
        assert client.post("/mcp/", headers=HEADERS, json={}).status_code == 429
    with pytest.raises(ValueError):
        create_app(token="", allowed_patient_ids=frozenset({"P-1001"}))
    with pytest.raises(ValueError):
        create_app(token=TOKEN, allowed_patient_ids=frozenset())


@pytest.mark.parametrize("url", ["http://remote.invalid/mcp/", "https://user:secret@remote.invalid/mcp/",
                                  "https://remote.invalid/mcp/?token=secret"])
def test_remote_client_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        MCPRecordsClient(url, TOKEN)
