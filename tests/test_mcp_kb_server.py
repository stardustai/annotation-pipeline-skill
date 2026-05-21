"""Stdio smoke test for the annotation-kb MCP server.

Spawns the server as a subprocess and exchanges JSON-RPC messages
according to the MCP protocol. Exercises initialize + tools/list +
tools/call.
"""
import json
import os
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
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
