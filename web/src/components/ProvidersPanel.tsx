import { useEffect, useMemo, useState } from "react";
import { fetchProviderConfig, saveProviderConfig, savePipelineConfig, testProvider } from "../api";
import { createProviderProfile, profileStatusLabel, profileTitle, providerConfigPayload } from "../providers";
import type { ProviderConfigSnapshot, Runtime, ProviderProfileConfig, PipelineView } from "../types";
import { WorkflowDiagram } from "./WorkflowDiagram";

// Preferred display order for the well-known stage targets; ANY other target
// present in llm_profiles.yaml (annotation_2, annotation_sonnet, custom ones)
// is rendered after these — the grid is NOT limited to a fixed set.
const STAGE_TARGET_ORDER = [
  "annotation", "annotation_2", "annotation_sonnet", "qc",
  "arbiter", "arbiter_secondary", "fallback",
];

export function ProvidersPanel({ storeKey = null }: { storeKey?: string | null }) {
  // Providers are workspace-global: a single llm_profiles.yaml shared across
  // every project in the workspace. We always pass storeKey=null so the API
  // resolves to the workspace-level file (with project-local fallback).
  const [snapshot, setSnapshot] = useState<ProviderConfigSnapshot | null>(null);
  // The pipeline (workflow.yaml) IS per-project, so it's fetched separately with
  // the selected store key while profile editing stays workspace-global.
  const [pipeline, setPipeline] = useState<PipelineView | null>(null);
  const [pipelineForm, setPipelineForm] = useState<{
    targets: string[]; keep_threshold: number; arbiter_target: string; on_disagree: string; run_qc: boolean;
  } | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null);
  const [newRuntime, setNewRuntime] = useState<Runtime>("claude_cli");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; latency_ms: number; error?: string } | null>(null);
  const [newTargetName, setNewTargetName] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchProviderConfig(null)
      .then((nextSnapshot) => {
        if (!active) return;
        setSnapshot(nextSnapshot);
        setSelectedProfile(nextSnapshot.profiles[0]?.name ?? null);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setMessage(reason instanceof Error ? reason.message : "Unable to load providers");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  // Per-project pipeline view (multi-annotation vs classic), refetched when the
  // selected project changes. Kept separate from the global profiles snapshot.
  useEffect(() => {
    let active = true;
    fetchProviderConfig(storeKey)
      .then((snap) => {
        if (active) setPipeline(snap.pipeline ?? null);
      })
      .catch(() => {
        if (active) setPipeline(null);
      });
    return () => {
      active = false;
    };
  }, [storeKey]);

  const availableTargets = useMemo(
    () => (snapshot ? Object.keys(snapshot.targets) : []),
    [snapshot],
  );

  // All stage targets present in llm_profiles.yaml, well-known ones first then
  // any extras (annotation_2, annotation_sonnet, custom). Not a fixed list.
  const orderedStageTargets = useMemo(() => {
    const keys = snapshot ? Object.keys(snapshot.targets) : [];
    const known = STAGE_TARGET_ORDER.filter((k) => keys.includes(k));
    const extra = keys.filter((k) => !STAGE_TARGET_ORDER.includes(k)).sort();
    return [...known, ...extra];
  }, [snapshot]);

  function addStageTarget() {
    const name = newTargetName.trim();
    if (!name || !snapshot || snapshot.targets[name] !== undefined) return;
    setSnapshot({ ...snapshot, targets: { ...snapshot.targets, [name]: "" } });
    setNewTargetName("");
  }

  // Keep the always-visible config form in sync with the live pipeline — on
  // load, on project switch, and after a save (which updates `pipeline`).
  useEffect(() => {
    if (!pipeline) {
      setPipelineForm(null);
      return;
    }
    setPipelineForm({
      targets: pipeline.annotators.map((a) => a.target),
      keep_threshold: pipeline.keep_threshold,
      arbiter_target: pipeline.arbiter.target,
      on_disagree: pipeline.on_disagree,
      run_qc: pipeline.qc.enabled, // multi: qc.enabled === !accept_directly
    });
  }, [pipeline]);

  function resetPipelineForm() {
    if (!pipeline) return;
    setPipelineForm({
      targets: pipeline.annotators.map((a) => a.target),
      keep_threshold: pipeline.keep_threshold,
      arbiter_target: pipeline.arbiter.target,
      on_disagree: pipeline.on_disagree,
      run_qc: pipeline.qc.enabled,
    });
  }

  function patchPipelineForm(patch: Partial<NonNullable<typeof pipelineForm>>) {
    setPipelineForm((prev) => (prev ? { ...prev, ...patch } : prev));
  }

  const selected = useMemo(
    () => snapshot?.profiles.find((profile) => profile.name === selectedProfile) ?? null,
    [snapshot, selectedProfile],
  );

  function updateSelected(updates: Partial<ProviderProfileConfig>) {
    if (!snapshot || !selected) return;
    const nextProfiles = snapshot.profiles.map((profile) =>
      profile.name === selected.name ? normalizeProfile({ ...profile, ...updates }) : profile,
    );
    const nextTargets = Object.fromEntries(
      Object.entries(snapshot.targets).map(([stage, profileName]) => [
        stage,
        profileName === selected.name && updates.name ? updates.name : profileName,
      ]),
    );
    setSnapshot({ ...snapshot, profiles: nextProfiles, targets: nextTargets });
    if (updates.name) setSelectedProfile(updates.name);
  }

  function addProfile() {
    if (!snapshot) return;
    const profile = createProviderProfile(newRuntime, snapshot.profiles.length + 1);
    setSnapshot({ ...snapshot, profiles: [...snapshot.profiles, profile] });
    setSelectedProfile(profile.name);
  }

  function deleteProfile() {
    if (!snapshot || !selected) return;
    const nextProfiles = snapshot.profiles.filter((profile) => profile.name !== selected.name);
    const replacement = nextProfiles[0]?.name ?? "";
    const nextTargets = Object.fromEntries(
      Object.entries(snapshot.targets).map(([stage, profileName]) => [stage, profileName === selected.name ? replacement : profileName]),
    );
    setSnapshot({ ...snapshot, profiles: nextProfiles, targets: nextTargets });
    setSelectedProfile(nextProfiles[0]?.name ?? null);
  }

  function updateTarget(stage: string, profileName: string) {
    if (!snapshot) return;
    setSnapshot({ ...snapshot, targets: { ...snapshot.targets, [stage]: profileName } });
  }

  async function runTest() {
    if (!selected) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testProvider(selected.name, null);
      setTestResult(result);
    } catch (reason: unknown) {
      setTestResult({ ok: false, latency_ms: 0, error: reason instanceof Error ? reason.message : "Unknown error" });
    } finally {
      setTesting(false);
    }
  }

  async function validateProviders() {
    setMessage(null);
    const nextSnapshot = await fetchProviderConfig(null);
    setSnapshot(nextSnapshot);
    setSelectedProfile((current) => current ?? nextSnapshot.profiles[0]?.name ?? null);
    setMessage("Provider validation refreshed");
  }

  const allProfileNames = useMemo(() => (snapshot ? snapshot.profiles.map((p) => p.name) : []), [snapshot]);

  // Options for any model picker: named targets (role → model) + every profile
  // directly, so an annotator/arbiter can be a target OR a provider.
  // `disabledValues` greys out names already chosen elsewhere (used by the
  // annotator rows so two annotators can't pick the same model — that would
  // make consensus trivial). The current row's own value is never disabled.
  function renderModelOptions(disabledValues?: Set<string>) {
    const off = (name: string) => (disabledValues?.has(name) ? true : undefined);
    return (
      <>
        <optgroup label="Targets (role → model)">
          {availableTargets.map((name) => (
            <option key={`t-${name}`} value={name} disabled={off(name)}>{name}{snapshot?.targets[name] ? ` (${snapshot.targets[name]})` : ""}</option>
          ))}
        </optgroup>
        <optgroup label="Models (pick a provider directly)">
          {allProfileNames.map((name) => (
            <option key={`p-${name}`} value={name} disabled={off(name)}>{name}</option>
          ))}
        </optgroup>
      </>
    );
  }

  // First profile/target not already used as an annotator, or null if every
  // one is taken. Never returns a name already in pipelineForm.targets, so
  // seeding/adding can't create a duplicate annotator.
  function firstUnusedAnnotator(): string | null {
    const used = new Set(pipelineForm?.targets ?? []);
    return allProfileNames.find((n) => !used.has(n)) ?? availableTargets.find((t) => !used.has(t)) ?? null;
  }

  // Toggle multi-annotation: ON seeds a distinct 2nd annotator + disables QC;
  // OFF collapses to a single annotator + re-enables QC.
  function setMultiMode(on: boolean) {
    if (!pipelineForm) return;
    if (on) {
      if (pipelineForm.targets.length >= 2) return;
      const second = firstUnusedAnnotator();
      if (!second) {
        setMessage("Add a second model profile before enabling multi-annotation.");
        return;
      }
      patchPipelineForm({ targets: [pipelineForm.targets[0], second], keep_threshold: 2, run_qc: false });
    } else {
      patchPipelineForm({ targets: pipelineForm.targets.slice(0, 1), keep_threshold: 1, run_qc: true });
    }
  }

  // One Save persists BOTH the provider map / profiles (llm_profiles.yaml) and
  // the workflow annotation config (workflow.yaml). Providers first so any new
  // target mapping exists before the pipeline save validates against it.
  async function saveAll() {
    if (!snapshot) return;
    // Guard the partial save: every annotator/arbiter the pipeline references
    // must exist as a profile or target in the snapshot we're about to persist,
    // so we never write the global llm_profiles.yaml and then fail the
    // workflow.yaml write (leaving the two files diverged).
    if (pipelineForm) {
      const known = new Set<string>([
        ...snapshot.profiles.map((p) => p.name),
        ...Object.keys(snapshot.targets),
      ]);
      const missing = [...pipelineForm.targets, pipelineForm.arbiter_target].filter((n) => !known.has(n));
      if (missing.length) {
        setMessage(`Cannot save: annotator/arbiter points at a missing profile/target: ${[...new Set(missing)].join(", ")}`);
        return;
      }
      // Reject duplicate annotators — two identical annotators always agree, so
      // consensus is meaningless and the arbiter never engages.
      if (pipelineForm.targets.length !== new Set(pipelineForm.targets).size) {
        setMessage("Cannot save: each annotator must be a distinct model. Remove or change the duplicate.");
        return;
      }
    }
    setSaving(true);
    setMessage(null);
    try {
      const savedProviders = await saveProviderConfig(providerConfigPayload(snapshot), null);
      setSnapshot(savedProviders);
      setSelectedProfile((current) => current ?? savedProviders.profiles[0]?.name ?? null);
    } catch (reason) {
      setMessage(reason instanceof Error ? `Provider save failed: ${reason.message}` : "Provider save failed");
      setSaving(false);
      return;
    }
    if (pipelineForm) {
      try {
        const replicas = pipelineForm.targets.length;
        const res = await savePipelineConfig(
          {
            replicas,
            targets: pipelineForm.targets,
            keep_threshold: Math.min(Math.max(1, pipelineForm.keep_threshold), Math.max(1, replicas)),
            on_disagree: pipelineForm.on_disagree,
            arbiter_target: pipelineForm.arbiter_target,
            accept_directly: replicas > 1 ? !pipelineForm.run_qc : undefined,
          },
          storeKey,
        );
        if (res.pipeline) setPipeline(res.pipeline);
        setMessage("Saved — restart the project runtime to apply pipeline changes.");
      } catch (reason) {
        setMessage(reason instanceof Error
          ? `Providers saved, but pipeline rejected: ${reason.message} (workflow.yaml unchanged)`
          : "Providers saved, but pipeline save failed");
      } finally {
        setSaving(false);
      }
    } else {
      setMessage("Provider configuration saved.");
      setSaving(false);
    }
  }

  async function resetAll() {
    const next = await fetchProviderConfig(null);
    setSnapshot(next);
    resetPipelineForm();
    setMessage(null);
  }

  if (loading) return <section className="work-panel">Loading providers</section>;
  if (!snapshot) return <section className="work-panel">{message ?? "No provider configuration loaded"}</section>;

  const isMulti = (pipelineForm?.targets.length ?? 0) >= 2;

  // Diagram + mode prose reflect the UNSAVED pipelineForm edits (toggle,
  // annotators, arbiter, keep_threshold) so the read-only diagram never
  // contradicts the editor below before Save.
  const resolveProfile = (t: string): string | null =>
    // `||` (not `??`): an unassigned target maps to "" — fall through to the
    // direct-profile check so the diagram shows the target name, not a blank.
    snapshot.targets[t] || (allProfileNames.includes(t) ? t : null);
  const previewPipeline: PipelineView | null =
    pipeline && pipelineForm
      ? {
          mode: isMulti ? "multi-annotation" : "single",
          replicas: pipelineForm.targets.length,
          annotators: pipelineForm.targets.map((t) => ({ target: t, profile: resolveProfile(t) })),
          keep_threshold: pipelineForm.keep_threshold,
          on_disagree: pipelineForm.on_disagree,
          arbiter: { target: pipelineForm.arbiter_target, profile: resolveProfile(pipelineForm.arbiter_target) },
          accept_directly: isMulti ? !pipelineForm.run_qc : false,
          qc: { enabled: isMulti ? pipelineForm.run_qc : true, target: "qc", profile: snapshot.targets["qc"] ?? null },
        }
      : pipeline;

  return (
    <section className="providers-panel" aria-label="Provider Configuration">
      <div className="runtime-header">
        <div>
          <h2>Providers</h2>
          <p>Configure subagent profiles, stage targets, local CLI binaries, API base URLs, and key environment names.</p>
        </div>
        <div className="provider-actions">
          <button className="view-tab" type="button" onClick={validateProviders}>
            Validate
          </button>
          <button className="primary-button" type="button" disabled={saving} onClick={saveAll}>
            {saving ? "Saving" : "Save"}
          </button>
        </div>
      </div>

      {message ? <div className="notice compact">{message}</div> : null}

      {/* Pipeline diagram — read-only visual of the project's annotation
          workflow (multi-annotation vs classic), derived from workflow.yaml.
          One workflow per project, so there's no pipeline selector. */}
      {pipeline ? (
        <div className="provider-pipeline" style={{ marginBottom: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.25rem" }}>
            <h3 style={{ margin: 0 }}>Pipeline</h3>
            {pipelineForm ? (
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)", cursor: "pointer" }}>
                Multi-annotation
                <input type="checkbox" role="switch" aria-checked={isMulti} aria-label="Multi-annotation mode" checked={isMulti} onChange={(e) => setMultiMode(e.target.checked)} />
              </label>
            ) : null}
          </div>
          <p style={{ marginTop: 0, marginBottom: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
            {(previewPipeline ?? pipeline).mode === "multi-annotation"
              ? `Multi-annotation: ${(previewPipeline ?? pipeline).replicas} annotators → consensus → arbiter → accept (QC disabled — the arbiter is the gate).`
              : "Classic: annotation → QC → arbiter → accept."}
          </p>
          <WorkflowDiagram pipeline={previewPipeline ?? pipeline} />
          {/* The editable pipeline config lives in the mode-aware Stage Targets below. */}
        </div>
      ) : null}

      {/* Stage Targets — mode-aware. Multi-annotation: annotators + consensus
          knobs live here, with qc / arbiter_secondary greyed (unused). Single:
          the classic role → profile grid. Each routes via client_factory. */}
      <div className="provider-targets" style={{ marginBottom: "1rem" }}>
        <h3 style={{ marginTop: 0 }}>Stage Targets</h3>
        {isMulti && pipelineForm ? (
          <>
            <p style={{ marginTop: "-0.25rem", marginBottom: "0.6rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
              Each annotator runs in parallel; spans agreed by <code>keep_threshold</code> are kept and the
              rest go to the arbiter. Pick any provider directly. <code>qc</code> and{" "}
              <code>arbiter_secondary</code> are unused in multi-annotation.
            </p>
            <label style={{ display: "block", fontWeight: 500, marginBottom: "0.35rem" }}>Annotators ({pipelineForm.targets.length})</label>
            {pipelineForm.targets.map((t, i) => (
              <div key={i} style={{ display: "flex", gap: "0.5rem", marginBottom: "0.35rem", alignItems: "center" }}>
                <span style={{ width: "5.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>annotator {i + 1}</span>
                <select
                  style={{ flex: 1 }}
                  aria-label={`annotator ${i + 1}`}
                  value={t}
                  onChange={(e) => {
                    const value = e.target.value;
                    // Ignore a pick already used by another annotator row — two
                    // identical annotators make consensus trivial. (The dup
                    // option is also disabled below, so this is a belt-and-braces
                    // guard against value coercion.)
                    if (pipelineForm.targets.some((other, j) => j !== i && other === value)) return;
                    const next = [...pipelineForm.targets];
                    next[i] = value;
                    patchPipelineForm({ targets: next });
                  }}
                >
                  {renderModelOptions(new Set(pipelineForm.targets.filter((_, j) => j !== i)))}
                </select>
                {pipelineForm.targets.length > 2 ? (
                  <button
                    className="view-tab"
                    type="button"
                    onClick={() => {
                      const next = pipelineForm.targets.filter((_, j) => j !== i);
                      // Re-clamp keep_threshold: it can never exceed the annotator count.
                      patchPipelineForm({ targets: next, keep_threshold: Math.min(pipelineForm.keep_threshold, next.length) });
                    }}
                  >
                    Remove
                  </button>
                ) : null}
              </div>
            ))}
            <button
              className="view-tab"
              type="button"
              style={{ marginBottom: "0.75rem" }}
              disabled={firstUnusedAnnotator() === null}
              onClick={() => {
                // Only ever append a model not already an annotator — no
                // duplicate-creating fallback. Button is disabled when none left.
                const unused = firstUnusedAnnotator();
                if (unused) patchPipelineForm({ targets: [...pipelineForm.targets, unused] });
              }}
            >
              + Add annotator
            </button>
            <div className="target-grid">
              <label>
                <span>keep_threshold (1–{pipelineForm.targets.length})</span>
                <input
                  type="number" min={1} max={pipelineForm.targets.length} value={pipelineForm.keep_threshold}
                  onChange={(e) => {
                    // Clamp to [1, annotator count] so the consensus threshold can
                    // never be unsatisfiable (> replicas) or zero.
                    const raw = Number(e.target.value);
                    const clamped = Number.isFinite(raw) ? Math.min(Math.max(1, Math.round(raw)), pipelineForm.targets.length) : 1;
                    patchPipelineForm({ keep_threshold: clamped });
                  }}
                />
              </label>
              <label>
                <span>on disagreement</span>
                <select value={pipelineForm.on_disagree} onChange={(e) => patchPipelineForm({ on_disagree: e.target.value })}>
                  <option value="arbiter">arbiter (resolve + fill)</option>
                  <option value="drop">drop (discard)</option>
                </select>
              </label>
              <label>
                <span>arbiter</span>
                <select value={pipelineForm.arbiter_target} onChange={(e) => patchPipelineForm({ arbiter_target: e.target.value })}>
                  {renderModelOptions()}
                </select>
              </label>
              <label>
                <span>fallback</span>
                <select value={snapshot.targets["fallback"] ?? ""} onChange={(e) => updateTarget("fallback", e.target.value)}>
                  <option value="">Unassigned</option>
                  {snapshot.profiles.map((p) => (<option key={p.name} value={p.name}>{p.name}</option>))}
                </select>
              </label>
              <label style={{ opacity: 0.45 }}>
                <span>qc · unused</span>
                <select disabled><option>{snapshot.targets["qc"] ?? "Unassigned"}</option></select>
              </label>
              <label style={{ opacity: 0.45 }}>
                <span>arbiter_secondary · unused</span>
                <select disabled><option>{snapshot.targets["arbiter_secondary"] ?? "Unassigned"}</option></select>
              </label>
              <NumberField
                label="Max Concurrent Tasks"
                value={snapshot.limits.max_concurrent_tasks}
                onChange={(value) => setSnapshot({ ...snapshot, limits: { max_concurrent_tasks: value } })}
              />
            </div>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
              <button className="primary-button" type="button" disabled={saving} onClick={saveAll}>{saving ? "Saving" : "Save"}</button>
              <button className="view-tab" type="button" disabled={saving} onClick={resetAll}>Reset</button>
            </div>
          </>
        ) : (
          <>
            <p style={{ marginTop: "-0.25rem", marginBottom: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
              Classic single-annotator flow (annotation → QC → arbiter). Each target routes to one profile
              via <code>client_factory(target_name)</code> — point any at any provider, or add a target below.
            </p>
            <div className="target-grid">
              {orderedStageTargets.map((stage) => (
                <label key={stage}>
                  <span>{stage}</span>
                  <select value={snapshot.targets[stage] ?? ""} onChange={(event) => updateTarget(stage, event.target.value)}>
                    <option value="">Unassigned</option>
                    {snapshot.profiles.map((profile) => (
                      <option key={profile.name} value={profile.name}>{profile.name}</option>
                    ))}
                  </select>
                </label>
              ))}
              <NumberField
                label="Max Concurrent Tasks"
                value={snapshot.limits.max_concurrent_tasks}
                onChange={(value) => setSnapshot({ ...snapshot, limits: { max_concurrent_tasks: value } })}
              />
            </div>
            <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.5rem" }}>
              <input
                type="text"
                placeholder="new target name (e.g. annotation_3)"
                value={newTargetName}
                onChange={(event) => setNewTargetName(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") addStageTarget(); }}
                style={{ flex: "0 1 280px" }}
              />
              <button className="view-tab" type="button" onClick={addStageTarget} disabled={!newTargetName.trim()}>
                + Add target
              </button>
            </div>
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.75rem" }}>
              <button className="primary-button" type="button" disabled={saving} onClick={saveAll}>{saving ? "Saving" : "Save"}</button>
              <button className="view-tab" type="button" disabled={saving} onClick={resetAll}>Reset</button>
            </div>
          </>
        )}
      </div>

      <div className="providers-layout">
        <aside className="provider-list">
          <div className="provider-add-row">
            <select value={newRuntime} onChange={(event) => setNewRuntime(event.target.value as Runtime)}>
              <option value="claude_cli">claude_cli</option>
              <option value="codex_cli">codex_cli</option>
              <option value="anthropic_sdk">anthropic_sdk</option>
              <option value="openai_sdk">openai_sdk</option>
            </select>
            <button className="view-tab" type="button" onClick={addProfile}>
              Add
            </button>
          </div>
          {snapshot.profiles.map((profile) => (
            <button
              className={profile.name === selectedProfile ? "provider-list-item selected" : "provider-list-item"}
              key={profile.name}
              type="button"
              onClick={() => { setSelectedProfile(profile.name); setTestResult(null); }}
            >
              <span>{profileTitle(profile)}</span>
              <small className={`provider-status ${profileStatusLabel(snapshot, profile.name)}`}>
                {profileStatusLabel(snapshot, profile.name)}
              </small>
            </button>
          ))}
        </aside>

        <div className="provider-editor">
          {selected ? (
            <>
              <div className="provider-section-header">
                <h3>Profile</h3>
                <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                  {testResult && (
                    <span className={`provider-test-result ${testResult.ok ? "ok" : "fail"}`}>
                      {testResult.ok
                        ? `✓ ${testResult.latency_ms}ms`
                        : `✗ ${testResult.error ?? "failed"}`}
                    </span>
                  )}
                  <button className="view-tab" type="button" onClick={runTest} disabled={testing}>
                    {testing ? "Testing…" : "Test"}
                  </button>
                  <button className="view-tab danger" type="button" onClick={deleteProfile} disabled={snapshot.profiles.length <= 1}>
                    Delete
                  </button>
                </div>
              </div>
              <div className="provider-form-grid">
                <TextField label="Name" value={selected.name} onChange={(value) => updateSelected({ name: value })} />
                <SelectField
                  label="Runtime"
                  value={selected.runtime}
                  options={["claude_cli", "codex_cli", "anthropic_sdk", "openai_sdk"]}
                  onChange={(value) => updateSelected({ runtime: value as Runtime })}
                />
                <TextField label="Model" value={selected.model} onChange={(value) => updateSelected({ model: value })} />
                <TextField label="Base URL" value={selected.base_url ?? ""} onChange={(value) => updateSelected({ base_url: value })} />
                <TextField label="API Key Env" value={typeof selected.api_key_env === "string" ? selected.api_key_env : (selected.api_key_env ?? []).join(", ")} onChange={(value) => updateSelected({ api_key_env: value })} />
                <PasswordField
                  label="API Key (inline)"
                  value={selected.api_key ?? ""}
                  placeholder={selected.api_key_set ? "set" : "not set"}
                  hint={selected.api_key_set ? "Leave blank to keep current key" : undefined}
                  onChange={(value) => updateSelected({ api_key: value })}
                />
                <TextField label="Reasoning Effort" value={selected.reasoning_effort ?? ""} onChange={(value) => updateSelected({ reasoning_effort: value || null })} />
                <TextField label="Permission Mode" value={selected.permission_mode ?? ""} onChange={(value) => updateSelected({ permission_mode: value || null })} />
                <NumberField label="Timeout Seconds" value={selected.timeout_seconds} onChange={(value) => updateSelected({ timeout_seconds: value })} />
                <NumberField label="Max Retries" value={selected.max_retries} onChange={(value) => updateSelected({ max_retries: value })} />
                <NumberField label="Concurrency Limit" value={selected.concurrency_limit} onChange={(value) => updateSelected({ concurrency_limit: value })} />
                <NumberField label="No Progress Timeout" value={selected.no_progress_timeout_seconds} onChange={(value) => updateSelected({ no_progress_timeout_seconds: value })} />
                <BoolField
                  label="Disable Continuity"
                  hint="Off (default) replays the conversation via claude --resume so vLLM's prefix cache stays warm. On = every turn opens a fresh session (no cache reuse). Only set on if continuity is causing context overflow that delta-prompt can't contain."
                  value={selected.disable_continuity}
                  onChange={(value) => updateSelected({ disable_continuity: value })}
                />
              </div>

              <div className="provider-diagnostics">
                <h3>Doctor</h3>
                {(snapshot.diagnostics[selected.name]?.checks ?? []).map((check) => (
                  <div className={`provider-check ${check.status}`} key={check.id}>
                    <span>{check.id}</span>
                    <strong>{check.status}</strong>
                    <p>{check.message}</p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div>No provider selected.</div>
          )}
        </div>
      </div>

    </section>
  );
}

function normalizeProfile(profile: ProviderProfileConfig): ProviderProfileConfig {
  return { ...profile };
}

function TextField(props: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label>
      <span>{props.label}</span>
      <input value={props.value} onChange={(event) => props.onChange(event.target.value)} />
    </label>
  );
}

function PasswordField(props: {
  label: string;
  value: string;
  placeholder?: string;
  hint?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label>
      <span>{props.label}</span>
      <input
        type="password"
        value={props.value}
        placeholder={props.placeholder}
        onChange={(event) => props.onChange(event.target.value)}
      />
      {props.hint ? <small className="provider-field-hint">{props.hint}</small> : null}
    </label>
  );
}

function NumberField(props: { label: string; value: number | null; onChange: (value: number | null) => void }) {
  return (
    <label>
      <span>{props.label}</span>
      <input
        type="number"
        min="0"
        value={props.value ?? ""}
        onChange={(event) => props.onChange(event.target.value ? Number(event.target.value) : null)}
      />
    </label>
  );
}

function BoolField(props: {
  label: string;
  value: boolean | null;
  hint?: string;
  onChange: (value: boolean | null) => void;
}) {
  // Tri-state: null = "inherit/unset", true/false = explicit. Stored as null
  // when the operator clears it so the YAML stays sparse (no `disable_continuity: false`
  // noise on every profile).
  const display = props.value === null || props.value === undefined ? "" : props.value ? "true" : "false";
  return (
    <label>
      <span>{props.label}</span>
      <select
        value={display}
        onChange={(event) => {
          const next = event.target.value;
          props.onChange(next === "" ? null : next === "true");
        }}
      >
        <option value="">(unset)</option>
        <option value="false">false</option>
        <option value="true">true</option>
      </select>
      {props.hint ? <small className="provider-field-hint">{props.hint}</small> : null}
    </label>
  );
}

function SelectField(props: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label>
      <span>{props.label}</span>
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
