"""Changing a provider target in llm_profiles.yaml must take effect on the
NEXT task's client build — without waiting for the background config-watcher
and without restarting the scheduler. Regression: the watcher proved unreliable
in production (annotation stayed on the old provider for 7.5h after the edit),
so target/profile resolution re-reads the yaml on mtime change at resolve time."""
import os

from annotation_pipeline_skill.core.runtime import RuntimeConfig
from annotation_pipeline_skill.llm.profiles import load_llm_registry
from annotation_pipeline_skill.runtime.local_scheduler import LocalRuntimeScheduler
from annotation_pipeline_skill.store.sqlite_store import SqliteStore


def _yaml(annotation_profile: str) -> str:
    return f"""profiles:
  p_old:
    runtime: openai_sdk
    model: m-old
    base_url: http://x
  p_new:
    runtime: openai_sdk
    model: m-new
    base_url: http://x
targets:
  annotation: {annotation_profile}
  arbiter: p_old
  qc: p_old
"""


def test_provider_change_takes_effect_on_next_resolve(tmp_path):
    yaml_path = tmp_path / "llm_profiles.yaml"
    yaml_path.write_text(_yaml("p_old"), encoding="utf-8")
    store = SqliteStore.open(tmp_path / "proj" / ".annotation-pipeline")

    built: list[str] = []

    def builder(profile):
        built.append(profile.name)
        return profile.name

    sched = LocalRuntimeScheduler(
        store=store,
        client_factory=lambda t: None,
        config=RuntimeConfig(),
        registry=load_llm_registry(yaml_path),
        client_builder=builder,
        profiles_yaml_path=yaml_path,
    )

    # initial resolve → old provider
    assert sched.client_factory("annotation") == "p_old"

    # operator edits the yaml: annotation now points at p_new
    yaml_path.write_text(_yaml("p_new"), encoding="utf-8")
    st = yaml_path.stat()
    os.utime(yaml_path, (st.st_atime + 50, st.st_mtime + 50))  # force a newer mtime

    # next resolve must pick up the new provider immediately — no watcher tick
    assert sched.client_factory("annotation") == "p_new"
