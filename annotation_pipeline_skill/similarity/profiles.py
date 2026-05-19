"""SimilarityProfile + yaml loader — config for embedding providers.

Mirrors annotation_pipeline_skill/llm/profiles.py. A workspace-global
``similarity_profiles.yaml`` or a project-local one defines named
profiles (jina_http, random baseline, future sentence-transformers
local, etc.). Scripts pass ``--profile <name>`` to pick one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_PROVIDERS = {"jina_http", "random", "minhash"}


@dataclass(frozen=True)
class SimilarityProfile:
    name: str
    provider: str  # one of ALLOWED_PROVIDERS
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    batch_size: int = 32
    max_tokens: int | None = None
    timeout_seconds: float = 60.0
    dim: int | None = None  # for the random-baseline provider
    # MinHash-only parameters (ignored by other providers).
    shingle_size: int = 5
    num_perm: int = 128

    def resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


def load_similarity_profiles(path: Path | str) -> dict[str, SimilarityProfile]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles_raw = raw.get("profiles") or {}
    if not isinstance(profiles_raw, dict):
        raise ValueError("similarity_profiles.yaml must define a top-level 'profiles' map")
    out: dict[str, SimilarityProfile] = {}
    for name, body in profiles_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"profile {name!r} must be a mapping")
        provider = body.get("provider")
        if provider not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"profile {name!r}: unknown provider {provider!r}; "
                f"allowed: {sorted(ALLOWED_PROVIDERS)}"
            )
        out[name] = SimilarityProfile(
            name=name,
            provider=provider,
            model=str(body.get("model") or ""),
            base_url=body.get("base_url"),
            api_key=body.get("api_key"),
            api_key_env=body.get("api_key_env"),
            # batch_size: 0 is a meaningful value ("don't batch on the
            # client side; send all inputs in one request"). The naive
            # ``or 32`` fallback ate the 0 and clobbered it back to 32 —
            # explicit None-check restores 0 as a legitimate setting.
            batch_size=int(
                body["batch_size"] if body.get("batch_size") is not None else 32
            ),
            max_tokens=body.get("max_tokens"),
            timeout_seconds=float(body.get("timeout_seconds") or 60.0),
            dim=body.get("dim"),
            shingle_size=int(
                body["shingle_size"] if body.get("shingle_size") is not None else 5
            ),
            num_perm=int(
                body["num_perm"] if body.get("num_perm") is not None else 128
            ),
        )
    return out
