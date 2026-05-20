"""Migrate llm_profiles.yaml from old layered schema to flat runtime schema.

Old fields removed: provider, provider_flavor, cli_kind, cli_binary, reasoning_capable
New field added: runtime (claude_cli | codex_cli)

Mapping:
  local_cli + cli_kind: claude   -> runtime: claude_cli
  local_cli + cli_kind: codex    -> runtime: codex_cli
  openai_compatible + any flavor -> runtime: claude_cli  (WARN: base_url may need updating)
  openai_responses               -> runtime: claude_cli  (WARN: base_url may need updating)

Usage:
  python scripts/migrate_llm_profiles.py --input path/to/llm_profiles.yaml
  python scripts/migrate_llm_profiles.py --input path/to/llm_profiles.yaml --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


_KNOWN_ANTHROPIC_SUFFIXES = ("/anthropic",)


def _needs_base_url_warning(base_url: str | None) -> bool:
    if not base_url:
        return True
    return not any(base_url.endswith(s) for s in _KNOWN_ANTHROPIC_SUFFIXES)


def migrate_profile(name: str, raw: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    out = dict(raw)

    provider = out.pop("provider", None)
    cli_kind = out.pop("cli_kind", None)
    out.pop("provider_flavor", None)
    out.pop("cli_binary", None)
    out.pop("reasoning_capable", None)

    if provider == "local_cli":
        runtime = "claude_cli" if cli_kind == "claude" else "codex_cli"
    elif provider in {"openai_compatible", "openai_responses"}:
        runtime = "claude_cli"
        base_url = out.get("base_url")
        if _needs_base_url_warning(base_url):
            warnings.append(
                f"  [{name}] base_url={base_url!r} may need updating to an Anthropic endpoint "
                f"(e.g. https://api.deepseek.com -> https://api.deepseek.com/anthropic)"
            )
    else:
        warnings.append(f"  [{name}] unknown provider={provider!r} — defaulting to claude_cli")
        runtime = "claude_cli"

    out["runtime"] = runtime
    # Ensure required fields present (warn if missing rather than crash)
    for field in ("model", "base_url", "api_key_env"):
        if not out.get(field):
            warnings.append(f"  [{name}] missing required field: {field}")

    # Reorder keys for readability
    ordered = {}
    for key in ("runtime", "model", "base_url", "api_key_env"):
        if key in out:
            ordered[key] = out.pop(key)
    ordered.update(out)
    return ordered, warnings


def migrate(input_path: Path) -> tuple[dict, list[str]]:
    payload = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    all_warnings: list[str] = []
    new_profiles: dict[str, dict] = {}
    for name, raw in (payload.get("profiles") or {}).items():
        migrated, warns = migrate_profile(str(name), dict(raw or {}))
        new_profiles[str(name)] = migrated
        all_warnings.extend(warns)
    payload["profiles"] = new_profiles
    return payload, all_warnings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to llm_profiles.yaml")
    ap.add_argument("--output", help="Output path (default: print to stdout)")
    ap.add_argument("--apply", action="store_true", help="Write output to --input path")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    payload, warnings = migrate(input_path)

    if warnings:
        print("WARNINGS — manual review required:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    out_yaml = yaml.dump(payload, allow_unicode=True, default_flow_style=False, sort_keys=False)

    if args.apply:
        output_path = Path(args.output) if args.output else input_path
        output_path.write_text(out_yaml, encoding="utf-8")
        print(f"Written to {output_path}", file=sys.stderr)
    elif args.output:
        Path(args.output).write_text(out_yaml, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(out_yaml)

    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
