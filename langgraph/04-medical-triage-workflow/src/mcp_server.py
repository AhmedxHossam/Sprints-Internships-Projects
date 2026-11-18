"""Authenticated async MCP record service. Run: python -m src.mcp_server.

Tool request/result schemas and typed record validation are declared in this
file. The service remains independent of LangGraph and reads mock_patients.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mcp.types as types
import uvicorn
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = ROOT / ".local"
ENV_FILE = Path(os.environ.get("TRIAGE_ENV_FILE", str(
    ROOT / ".env" if (ROOT / ".env").exists() else LOCAL_DIR / ".env"
)))
RUNTIME_DIR = Path(os.environ.get("TRIAGE_RUNTIME_DIR", str(LOCAL_DIR / "runtime")))

class Visit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    department: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=1000)
    diagnosis: str = Field(min_length=1, max_length=1000)
    treatment_plan: str = Field(min_length=1, max_length=2000)


class MedicalRecords(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    patient_id: str = Field(pattern=r"^P-[0-9]{4}$")
    age: int = Field(ge=0, le=130)
    chronic_conditions: list[str] = Field(max_length=100)
    recent_visits: list[Visit] = Field(max_length=100)


TOOL_INPUT_SCHEMA = {'$schema': 'https://json-schema.org/draft/2020-12/schema',
 'title': 'get_medical_records arguments',
 'type': 'object',
 'additionalProperties': False,
 'properties': {'patient_id': {'type': 'string', 'pattern': '^P-[0-9]{4}$'}},
 'required': ['patient_id']}

TOOL_RESULT_SCHEMA = {'$schema': 'https://json-schema.org/draft/2020-12/schema',
 'title': 'Medical records tool result',
 '$defs': {'Visit': {'additionalProperties': False,
                     'properties': {'date': {'pattern': '^\\d{4}-\\d{2}-\\d{2}$',
                                             'title': 'Date',
                                             'type': 'string'},
                                    'department': {'maxLength': 200,
                                                   'minLength': 1,
                                                   'title': 'Department',
                                                   'type': 'string'},
                                    'reason': {'maxLength': 1000,
                                               'minLength': 1,
                                               'title': 'Reason',
                                               'type': 'string'},
                                    'diagnosis': {'maxLength': 1000,
                                                  'minLength': 1,
                                                  'title': 'Diagnosis',
                                                  'type': 'string'},
                                    'treatment_plan': {'maxLength': 2000,
                                                       'minLength': 1,
                                                       'title': 'Treatment Plan',
                                                       'type': 'string'}},
                     'required': ['date', 'department', 'reason', 'diagnosis', 'treatment_plan'],
                     'title': 'Visit',
                     'type': 'object'},
           'MedicalRecords': {'additionalProperties': False,
                              'properties': {'patient_id': {'pattern': '^P-[0-9]{4}$',
                                                            'title': 'Patient Id',
                                                            'type': 'string'},
                                             'age': {'maximum': 130,
                                                     'minimum': 0,
                                                     'title': 'Age',
                                                     'type': 'integer'},
                                             'chronic_conditions': {'items': {'type': 'string'},
                                                                    'maxItems': 100,
                                                                    'title': 'Chronic Conditions',
                                                                    'type': 'array'},
                                             'recent_visits': {'items': {'$ref': '#/$defs/Visit'},
                                                               'maxItems': 100,
                                                               'title': 'Recent Visits',
                                                               'type': 'array'}},
                              'required': ['patient_id',
                                           'age',
                                           'chronic_conditions',
                                           'recent_visits'],
                              'title': 'MedicalRecords',
                              'type': 'object'}},
 'oneOf': [{'$ref': '#/$defs/MedicalRecords'},
           {'type': 'object',
            'additionalProperties': False,
            'required': ['error'],
            'properties': {'error': {'type': 'object',
                                     'additionalProperties': False,
                                     'required': ['code', 'message', 'data'],
                                     'properties': {'code': {'type': 'integer'},
                                                    'message': {'type': 'string'},
                                                    'data': {'type': 'object',
                                                             'additionalProperties': False,
                                                             'required': ['kind'],
                                                             'properties': {'kind': {'type': 'string'}}}}}}}]}

TOOL_VALIDATOR = Draft202012Validator(TOOL_INPUT_SCHEMA)

log = logging.getLogger("triage.mcp")


PATIENTS_FILE = Path(__file__).with_name("mock_patients.json")


@dataclass(frozen=True)
class Principal:
    name: str
    patient_ids: frozenset[str]


CURRENT_PRINCIPAL: ContextVar[Principal | None] = ContextVar("records_principal", default=None)


class RecordsRepository(Protocol):
    async def fetch(self, patient_id: str) -> dict | None: ...


class JsonRecordsRepository:
    """Replace this adapter with an async clinical data service in production."""

    def __init__(self, path: Path = PATIENTS_FILE):
        self.path = path

    async def fetch(self, patient_id: str) -> dict | None:
        def read() -> dict | None:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            matches = [row for row in rows if row["patient_id"] == patient_id]
            if len(matches) > 1:
                raise ValueError("Duplicate record identity")
            return matches[0] if matches else None

        return await asyncio.to_thread(read)


def tool_error(code: int, message: str, kind: str) -> types.CallToolResult:
    payload = {"error": {"code": code, "message": message, "data": {"kind": kind}}}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload, isError=True,
    )


async def get_medical_records(
    patient_id: str, *, repository: RecordsRepository, principal: Principal
) -> types.CallToolResult:
    """Fetch authorized historical records; never infer missing records."""
    if not TOOL_VALIDATOR.is_valid({"patient_id": patient_id}):
        return tool_error(-32602, "A valid patient_id is required", "INVALID_PARAMS")
    if patient_id not in principal.patient_ids:
        log.info("event=records_access_denied")
        return tool_error(-32003, "Record access denied", "FORBIDDEN")
    try:
        async with asyncio.timeout(5):
            row = await repository.fetch(patient_id)
        if row is None:
            return tool_error(-32004, "Medical record not found", "NOT_FOUND")
        # Exclude patient/physician names and unrelated fields from the output.
        record = MedicalRecords.model_validate({
            "patient_id": row["patient_id"], "age": row["age"],
            "chronic_conditions": row["chronic_conditions"],
            "recent_visits": [
                {key: visit[key] for key in ("date", "department", "reason", "diagnosis", "treatment_plan")}
                for visit in row["recent_visits"]
            ],
        })
        if record.patient_id != patient_id:
            raise ValueError("Record identity mismatch")
        payload = record.model_dump()
        log.info("event=records_fetched")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload, isError=False,
        )
    except Exception:
        log.warning("event=records_unavailable")
        return tool_error(-32001, "Medical records temporarily unavailable", "UNAVAILABLE")


def create_mcp_server(repository: RecordsRepository) -> Server:
    server = Server("aegishealth-medical-records", version="1.0.0")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(
            name="get_medical_records",
            description="Read authorized historical records for one patient.",
            inputSchema=TOOL_INPUT_SCHEMA,
            outputSchema=TOOL_RESULT_SCHEMA,
            annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
        )]

    # Explicit validation yields stable structured errors without echoing PHI.
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
        if name != "get_medical_records":
            return tool_error(-32601, "Unknown tool", "UNKNOWN_TOOL")
        if not TOOL_VALIDATOR.is_valid(arguments):
            return tool_error(-32602, "A valid patient_id is required", "INVALID_PARAMS")
        principal = CURRENT_PRINCIPAL.get()
        if principal is None:
            return tool_error(-32003, "Authenticated principal required", "FORBIDDEN")
        return await get_medical_records(arguments["patient_id"], repository=repository, principal=principal)

    return server


class ServiceAuthMiddleware:
    """One local service principal; replace with organization identity in production."""

    def __init__(self, app, *, token: str, principal: Principal, requests_per_minute: int = 120):
        self.app, self.token, self.principal = app, token, principal
        self.limit = requests_per_minute
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        supplied = Request(scope).headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {self.token}".encode()):
            log.warning("event=authentication_denied")
            response = JSONResponse({"error": "Unauthorized"}, 401,
                                    headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
            return await response(scope, receive, send)
        async with self.lock:
            now = time.monotonic()
            while self.requests and self.requests[0] <= now - 60:
                self.requests.popleft()
            limited = len(self.requests) >= self.limit
            if not limited:
                self.requests.append(now)
        if limited:
            return await JSONResponse({"error": "Rate limit exceeded"}, 429,
                                      headers={"Retry-After": "60"})(scope, receive, send)
        marker = CURRENT_PRINCIPAL.set(self.principal)

        async def private_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"cache-control", b"no-store")]
            await send(message)

        try:
            await self.app(scope, receive, private_send)
        finally:
            CURRENT_PRINCIPAL.reset(marker)


def create_app(*, token: str, allowed_patient_ids: frozenset[str],
               repository: RecordsRepository | None = None, requests_per_minute: int = 120):
    if len(token) < 32 or not allowed_patient_ids:
        raise ValueError("A strong service token and explicit patient ACL are required")
    if any(not TOOL_VALIDATOR.is_valid({"patient_id": value}) for value in allowed_patient_ids):
        raise ValueError("Invalid patient ACL")
    server = create_mcp_server(repository or JsonRecordsRepository())
    manager = StreamableHTTPSessionManager(
        app=server, stateless=True, json_response=True,
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1", "localhost", "[::1]", "127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1", "http://localhost", "http://127.0.0.1:*", "http://localhost:*"],
        ),
    )

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Mount("/mcp", app=manager.handle_request)], lifespan=lifespan)
    return ServiceAuthMiddleware(
        app, token=token, principal=Principal("triage-service", allowed_patient_ids),
        requests_per_minute=requests_per_minute,
    )


def main() -> None:
    load_dotenv(ENV_FILE, override=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("mcp").setLevel(logging.WARNING)
    app = create_app(
        token=os.environ.get("MCP_RECORDS_TOKEN", ""),
        allowed_patient_ids=frozenset(filter(None, os.environ.get("MCP_ALLOWED_PATIENT_IDS", "").split(","))),
    )
    uvicorn.run(app, host="127.0.0.1", port=8765, access_log=False)


if __name__ == "__main__":
    main()
