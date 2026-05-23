"""In-process tools the Anthropic SDK client exposes to subagent LLMs.

These were previously wrapped as MCP stdio servers (``annotation_pipeline_skill
.mcp.*``) because the legacy ``runtime: claude_cli`` path needed a stdio MCP
config to spawn them as subprocesses. The SDK runtime calls the pure
functions directly — no subprocess, no JSON-RPC marshalling — so the MCP
wrapper is gone and the tools live here.

Tool schemas + dispatch wiring: ``annotation_pipeline_skill.llm.tool_registry``.
"""
