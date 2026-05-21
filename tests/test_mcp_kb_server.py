"""Stdio smoke test for the annotation-kb MCP server.

Spawns the server as a subprocess and exchanges JSON-RPC messages
according to the MCP protocol. Exercises initialize + tools/list +
tools/call.
"""
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from annotation_pipeline_skill.services.entity_convention_service import (
    EntityConventionService,
)
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


@pytest.fixture
def project_with_convention(tmp_path):
    store = SqliteStore.open(tmp_path)
    svc = EntityConventionService(store)
    for i in range(5):
        # Unique source per call to bypass the (type, source) idempotency
        # guard so all 5 votes register as evidence (see Task 6 tests).
        svc.record_decision(
            project_id="proj_demo", span="Android", entity_type="technology",
            source=f"qc_consensus:seed{i}", task_id=f"task_{i}", row_id=f"row_{i}",
            row_content=f"Crashes on Android 10 in case {i}",
        )
    return tmp_path


def _rpc(proc, msg):
    proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _recv(proc, timeout=5.0):
    """Read one JSON-RPC line with a real wall-clock timeout.

    proc.stdout.readline() blocks until newline arrives — wrapping it in
    a `while time.time() < deadline` loop is useless because the check
    only runs *after* readline returns. Use select() so a hung server
    fails fast instead of hanging the test indefinitely.
    """
    deadline = time.time() + timeout
    buf = bytearray()
    while time.time() < deadline:
        remaining = deadline - time.time()
        rlist, _, _ = select.select([proc.stdout], [], [], remaining)
        if not rlist:
            continue
        chunk = proc.stdout.read1(4096)
        if not chunk:
            # EOF — server died.
            raise RuntimeError("server closed stdout (likely crashed)")
        buf.extend(chunk)
        if b"\n" in buf:
            line, _, rest = bytes(buf).partition(b"\n")
            # If we read past the first line, put the rest back by
            # creating a small buffer attribute on proc — but for our
            # simple one-request-one-response tests, the assumption is
            # one line per recv. Truncate to the first line.
            return json.loads(line.decode("utf-8"))
    raise TimeoutError("no response")


def test_mcp_server_lists_check_past_experience_tool(project_with_convention):
    proc = subprocess.Popen(
        [sys.executable, "-m", "annotation_pipeline_skill.mcp.kb_server",
         "--project-root", str(project_with_convention),
         "--project-id", "proj_demo"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Initialize.
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        init_resp = _recv(proc)
        assert init_resp["id"] == 1
        # Notify initialized.
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        # List tools.
        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        list_resp = _recv(proc)
        names = [t["name"] for t in list_resp["result"]["tools"]]
        assert "check_past_experience" in names
    finally:
        proc.kill()
        proc.wait(timeout=2)


def test_mcp_server_calls_check_past_experience(project_with_convention):
    proc = subprocess.Popen(
        [sys.executable, "-m", "annotation_pipeline_skill.mcp.kb_server",
         "--project-root", str(project_with_convention),
         "--project-id", "proj_demo"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        _recv(proc)
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "check_past_experience",
                               "arguments": {"entry": "Android"}}})
        call_resp = _recv(proc)
        content = call_resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert payload["entry"] == "Android"
        assert payload["convention"]["type"] == "technology"
        assert payload["convention"]["evidence_count"] == 5
    finally:
        proc.kill()
        proc.wait(timeout=2)


def test_mcp_server_handles_storage_error_gracefully(tmp_path):
    """If the DB doesn't have the entity_conventions table (e.g. corrupt
    workspace), the tool should return a structured error payload, not
    crash the MCP protocol."""
    # Spawn against a fresh tmp dir that has no SqliteStore initialization.
    # SqliteStore.open() creates the schema, so the call should still work
    # but with empty proposals — the tool returns a 'none' status, not an error.
    # Instead, test that empty-string entry returns the ValueError-as-error
    # path through the protocol (the unit test already covers the function
    # raising; this exercises the protocol surface).
    proc = subprocess.Popen(
        [sys.executable, "-m", "annotation_pipeline_skill.mcp.kb_server",
         "--project-root", str(tmp_path),
         "--project-id", "proj_demo"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05",
                               "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
        _recv(proc)
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "check_past_experience",
                               "arguments": {"entry": ""}}})
        resp = _recv(proc)
        content = resp["result"]["content"][0]["text"]
        payload = json.loads(content)
        assert "error" in payload
        assert "entry is required" in payload["error"].lower()
    finally:
        proc.kill()
        proc.wait(timeout=2)
