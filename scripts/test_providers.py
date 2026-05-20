"""Quick smoke-test every profile in an llm_profiles.yaml.

Usage:
  python scripts/test_providers.py --yaml projects/v3_initial_deployment/.annotation-pipeline/llm_profiles.yaml
  python scripts/test_providers.py --yaml projects/llm_profiles.yaml --profiles deepseek_flash,glm_46
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation_pipeline_skill.llm.client import LLMGenerateRequest
from annotation_pipeline_skill.llm.local_cli import LocalCLIClient
from annotation_pipeline_skill.llm.profiles import load_llm_registry

PROMPT = "Reply with exactly one word: ok"


async def test_profile(profile, *, timeout: int = 60) -> tuple[bool, str]:
    client = LocalCLIClient(profile)
    request = LLMGenerateRequest(prompt=PROMPT)
    try:
        result = await asyncio.wait_for(client.generate(request), timeout=timeout)
        text = result.final_text.strip()
        return True, repr(text[:80])
    except asyncio.TimeoutError:
        return False, f"timeout after {timeout}s"
    except Exception as exc:
        return False, str(exc)[:200]


async def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", required=True, help="Path to llm_profiles.yaml")
    ap.add_argument("--profiles", help="Comma-separated profile names to test (default: all)")
    ap.add_argument("--timeout", type=int, default=60, help="Per-profile timeout seconds")
    args = ap.parse_args(argv)

    registry = load_llm_registry(args.yaml)
    names = (
        [n.strip() for n in args.profiles.split(",")]
        if args.profiles
        else list(registry.profiles)
    )

    results: list[tuple[str, bool, str]] = []
    for name in names:
        profile = registry.profiles.get(name)
        if profile is None:
            results.append((name, False, "profile not found"))
            continue
        print(f"  testing {name:30s} [{profile.runtime:10s}] {profile.model} ...", end="", flush=True)
        t0 = time.monotonic()
        ok, msg = await test_profile(profile, timeout=args.timeout)
        elapsed = time.monotonic() - t0
        status = "OK " if ok else "FAIL"
        print(f"  {status}  ({elapsed:.1f}s)  {msg}")
        results.append((name, ok, msg))

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    print(f"\n{passed}/{len(results)} passed", end="")
    if failed:
        print(f"  —  FAILED: {', '.join(n for n, ok, _ in results if not ok)}")
    else:
        print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
