"""MCP stdio FX server and LangChain tool adapter. Run this file to serve MCP."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import ExitStack
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state_schema import FXQuote, FXRequest


@asynccontextmanager
async def connect_fx_tool(log_path: Path | None = None):
    # Do not forward model/tracer credentials to the tool subprocess.
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve())], env=env,
    )
    # Jupyter's stderr has no OS fileno on Windows; the SDK needs a real file.
    log_path = log_path or Path(__file__).resolve().parents[1] / "runtime" / "mcp_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as files:
        errlog = files.enter_context(log_path.open("a", encoding="utf-8"))
        async with _tool_session(params, errlog) as tool:
            yield tool


@asynccontextmanager
async def _tool_session(params, errlog):
    async with stdio_client(params, errlog=errlog) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            discovered = await session.list_tools()
            spec = next((item for item in discovered.tools if item.name == "fetch_egp_rate"), None)
            if spec is None:
                raise RuntimeError("MCP server did not advertise fetch_egp_rate")

            async def call(base_currency: str = "USD") -> dict:
                request = FXRequest(base_currency=base_currency)
                result = await asyncio.wait_for(
                    session.call_tool(spec.name, request.model_dump()), timeout=20,
                )
                if result.isError:
                    raise RuntimeError("MCP FX tool returned an error")
                payload = result.structuredContent
                if payload is None:
                    payload = json.loads(next(c.text for c in result.content if c.type == "text"))
                return FXQuote.model_validate(payload).model_dump(mode="json")

            yield StructuredTool.from_function(
                coroutine=call, name=spec.name,
                description=spec.description or "Synthetic USD to EGP training quote",
                args_schema=FXRequest,
            )

# MCP server entry point
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from state_schema import FXQuote, FXRequest

mcp = FastMCP("CapitalTrust_Training_FX")


@mcp.tool()
def fetch_egp_rate(base_currency: str = "USD") -> dict:
    """Return a TRAINING MOCK, never a live/CBE quote. Unit: EGP per 1 USD.

    Only USD is supported. last_updated is generation time of the synthetic
    observation, not an observation from a financial market.
    """
    FXRequest(base_currency=base_currency)
    rate = 47.85
    return FXQuote(
        base="USD", target="EGP", rate=rate, inverse_rate=1 / rate,
        unit="EGP per 1 USD", source="TRAINING_MOCK_NOT_CBE",
        last_updated=datetime.now(timezone.utc).isoformat(), is_mock=True,
    ).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run(transport="stdio")
