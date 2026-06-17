import { useEffect, useMemo, useState } from "react";
import { fetchProviderConfig, saveProviderConfig, savePipelineConfig, testProvider } from "../api";
import { createProviderProfile, profileStatusLabel, profileTitle, providerConfigPayload } from "../providers";
import type { ProviderConfigSnapshot, Runtime, ProviderProfileConfig, PipelineView } from "../types";
import { WorkflowDiagram } from "./WorkflowDiagram";

// Stages the runtime resolves via `client_factory(target_name)`. Order
// matters for layout (top row = the most-edited two; second row =
// arbitration; third = fallback). Note: `arbiter_secondary` is the
// prior-divergence second arbiter and should usually be a different
// LLM family from `arbiter` to keep the cross-LLM check honest.
const stageTargets = [
  "annotation", "qc",
  "arbiter", "arbiter_secondary",
  "fallback",
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
  const [pipelineSaving, setPipelineSaving] = useState(false);
  const [pipelineMsg, setPipelineMsg] = useState<string | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null);
  const [newRuntime, setNewRuntime] = useState<Runtime>("claude_cli");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; latency_ms: number; error?: string } | null>(null);

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
    setPipelineMsg(null);
  }

  function patchPipelineForm(patch: Partial<NonNullable<typeof pipelineForm>>) {
    setPipelineForm((prev) => (prev ? { ...prev, ...patch } : prev));
  }

  async function savePipeline() {
    if (!pipelineForm) return;
    const replicas = pipelineForm.targets.length;
    setPipelineSaving(true);
    setPipelineMsg(null);
    try {
      const res = await savePipelineConfig(
        {
          replicas,
          targets: pipelineForm.targets,
          keep_threshold: Math.min(Math.max(1, pipelineForm.keep_threshold), replicas),
          on_disagree: pipelineForm.on_disagree,
          arbiter_target: pipelineForm.arbiter_target,
          accept_directly: replicas > 1 ? !pipelineForm.run_qc : undefined,
        },
        storeKey,
      );
      if (res.pipeline) setPipeline(res.pipeline);
      setPipelineMsg("Saved to workflow.yaml — restart the project runtime to apply.");
    } catch (reason) {
      setPipelineMsg(reason instanceof Error ? reason.message : "Pipeline save failed");
    } finally {
      setPipelineSaving(false);
    }
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

  async function saveProviders() {
    if (!snapshot) return;
    setSaving(true);
    setMessage(null);
    try {
      const saved = await saveProviderConfig(providerConfigPayload(snapshot), null);
      setSnapshot(saved);
      setSelectedProfile((current) => current ?? saved.profiles[0]?.name ?? null);
      setMessage("Provider configuration saved");
    } catch (reason: unknown) {
      setMessage(reason instanceof Error ? reason.message : "Unable to save providers");
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <section className="work-panel">Loading providers</section>;
  if (!snapshot) return <section className="work-panel">{message ?? "No provider configuration loaded"}</section>;

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
          <button className="primary-button" type="button" disabled={saving} onClick={saveProviders}>
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
          <h3 style={{ marginTop: 0, marginBottom: "0.25rem" }}>Pipeline</h3>
          <p style={{ marginTop: 0, marginBottom: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
            {pipeline.mode === "multi-annotation"
              ? `Multi-annotation: ${pipeline.replicas} annotators → consensus → arbiter → accept (QC disabled — the arbiter is the gate).`
              : "Classic: annotation → QC → arbiter → accept."}
          </p>
          <WorkflowDiagram pipeline={pipeline} />
          {pipelineMsg ? <div className="notice compact" style={{ marginTop: "0.5rem" }}>{pipelineMsg}</div> : null}
          {pipelineForm ? (
            <div className="pipeline-editor" style={{ marginTop: "0.75rem", padding: "0.75rem", border: "1px solid var(--border, #2a2a2a)", borderRadius: 8 }}>
              <h4 style={{ marginTop: 0, marginBottom: "0.25rem" }}>Configure multi-annotation</h4>
              <p style={{ marginTop: 0, marginBottom: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
                One row = one annotator (replicas). Two or more annotators → multi-annotation (consensus + arbiter, QC off).
                Targets resolve to models via Stage Targets / llm_profiles.yaml.
              </p>
              <label style={{ display: "block", fontWeight: 500, marginBottom: "0.25rem" }}>Annotators ({pipelineForm.targets.length})</label>
              {pipelineForm.targets.map((t, i) => (
                <div key={i} style={{ display: "flex", gap: "0.5rem", marginBottom: "0.35rem", alignItems: "center" }}>
                  <select
                    value={t}
                    onChange={(e) => {
                      const next = [...pipelineForm.targets];
                      next[i] = e.target.value;
                      patchPipelineForm({ targets: next });
                    }}
                  >
                    {availableTargets.map((name) => (
                      <option key={name} value={name}>{name}{snapshot?.targets[name] ? ` (${snapshot.targets[name]})` : ""}</option>
                    ))}
                  </select>
                  {pipelineForm.targets.length > 1 ? (
                    <button className="view-tab" type="button" onClick={() => patchPipelineForm({ targets: pipelineForm.targets.filter((_, j) => j !== i) })}>
                      Remove
                    </button>
                  ) : null}
                </div>
              ))}
              <button
                className="view-tab"
                type="button"
                style={{ marginBottom: "0.6rem" }}
                onClick={() => {
                  const unused = availableTargets.find((n) => !pipelineForm.targets.includes(n)) ?? availableTargets[0];
                  if (unused) patchPipelineForm({ targets: [...pipelineForm.targets, unused] });
                }}
              >
                + Add annotator
              </button>
              {pipelineForm.targets.length > 1 ? (
                <div className="pipeline-editor-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.6rem" }}>
                  <label>
                    <span style={{ fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>keep_threshold (1–{pipelineForm.targets.length})</span>
                    <input
                      type="number" min={1} max={pipelineForm.targets.length} value={pipelineForm.keep_threshold}
                      onChange={(e) => patchPipelineForm({ keep_threshold: Number(e.target.value) })}
                    />
                  </label>
                  <label>
                    <span style={{ fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>arbiter target</span>
                    <select value={pipelineForm.arbiter_target} onChange={(e) => patchPipelineForm({ arbiter_target: e.target.value })}>
                      {availableTargets.map((name) => (
                        <option key={name} value={name}>{name}{snapshot?.targets[name] ? ` (${snapshot.targets[name]})` : ""}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span style={{ fontSize: "0.8rem", color: "var(--muted, #6b7280)" }}>on disagreement</span>
                    <select value={pipelineForm.on_disagree} onChange={(e) => patchPipelineForm({ on_disagree: e.target.value })}>
                      <option value="arbiter">arbiter (resolve + fill)</option>
                      <option value="drop">drop (discard)</option>
                    </select>
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", alignSelf: "end" }}>
                    <input type="checkbox" checked={pipelineForm.run_qc} onChange={(e) => patchPipelineForm({ run_qc: e.target.checked })} />
                    <span style={{ fontSize: "0.85rem" }}>also run QC after merge</span>
                  </label>
                </div>
              ) : (
                <p style={{ fontSize: "0.8rem", color: "var(--muted, #6b7280)", marginBottom: "0.6rem" }}>
                  Single annotator → classic annotation → QC → arbiter flow.
                </p>
              )}
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button className="primary-button" type="button" disabled={pipelineSaving} onClick={savePipeline}>
                  {pipelineSaving ? "Saving" : "Save pipeline"}
                </button>
                <button className="view-tab" type="button" disabled={pipelineSaving} onClick={resetPipelineForm}>
                  Reset
                </button>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Stage Targets — pinned to the TOP since it's the most-edited
          block and the routing decisions here determine which profile
          handles which pipeline stage. Single source of truth for the
          stage → profile mapping; the Annotation Agents form no longer
          edits this. */}
      <div className="provider-targets" style={{ marginBottom: "1rem" }}>
        <h3 style={{ marginTop: 0 }}>Stage Targets</h3>
        <p style={{ marginTop: "-0.25rem", marginBottom: "0.5rem", fontSize: "0.85rem", color: "var(--muted, #6b7280)" }}>
          Each stage routes to one profile at runtime via{" "}
          <code>client_factory(stage_name)</code>. <code>arbiter_secondary</code>{" "}
          is the prior-divergence second arbiter — use a different LLM family
          from <code>arbiter</code> for an honest cross-LLM check.{" "}
          <code>fallback</code> is invoked on transient provider errors
          (429 / 5xx) when a primary stage call fails.
        </p>
        <div className="target-grid">
          {stageTargets.map((stage) => (
            <label key={stage}>
              <span>{stage}</span>
              <select value={snapshot.targets[stage] ?? ""} onChange={(event) => updateTarget(stage, event.target.value)}>
                <option value="">Unassigned</option>
                {snapshot.profiles.map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.name}
                  </option>
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
