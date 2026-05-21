"""Stdio MCP server exposing the annotation knowledge base tool.

Launched by Claude CLI via --mcp-config. The server holds a read-only
SQLite connection to the project DB and exposes a single tool,
check_past_experience.

Invocation:
    python -m annotation_pipeline_skill.mcp.kb_server \\
        --project-root <annotation-pipeline workspace> \\
        --project-id <project id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from annotation_pipeline_skill.mcp.check_past_experience import check_past_experience
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


logger = logging.getLogger("annotation_kb_mcp")


def build_server(*, project_root: Path, project_id: str) -> Server:
    server: Server = Server("annotation-kb")
    store = SqliteStore.open(project_root)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="check_past_experience",
                description=(
                    "Query the project's annotation history for a candidate "
                    "entity/span. Returns the current convention (if any), "
                    "the distribution of past type proposals, up to 3 "
                    "diverse sentence-level examples per type, and a "
                    "wordfreq Zipf score. Use this BEFORE deciding the "
                    "type of an ambiguous or unfamiliar span — past "
                    "decisions and concrete row examples beat statistical "
                    "summaries for in-context generalization."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entry": {
                            "type": "string",
                            "description": "The candidate span text (case-insensitive lookup).",
                        },
                    },
                    "required": ["entry"],
                },
            )
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name != "check_past_experience":
            raise ValueError(f"unknown tool: {name}")
        entry = arguments.get("entry", "")
        try:
            result = check_past_experience(store, project_id=project_id, entry=entry)
        except ValueError as exc:
            payload = {"error": str(exc)}
        else:
            payload = result
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="annotation-kb-mcp-server")
    parser.add_argument("--project-root", required=True, type=Path,
                        help="Path to the annotation-pipeline workspace root (contains db.sqlite).")
    parser.add_argument("--project-id", required=True,
                        help="Project ID to scope queries against.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    server = build_server(project_root=args.project_root, project_id=args.project_id)

    async def _run() -> None:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
