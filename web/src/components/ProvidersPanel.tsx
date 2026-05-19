import { useEffect, useMemo, useState } from "react";
import { fetchProviderConfig, saveProviderConfig } from "../api";
import { createProviderProfile, profileStatusLabel, profileTitle, providerConfigPayload } from "../providers";
import type { ProviderConfigSnapshot, ProviderName, ProviderProfileConfig } from "../types";

// Stages the runtime resolves via `client_factory(target_name)`. Order
// matters for layout (top row = the most-edited two; second row =
// arbitration; third = fallback). Note: `arbiter_secondary` is the
// prior-divergence second arbiter and should usually be a different
// LLM family from `arbiter` to keep the cross-LLM check honest.
const stageTargets = [
  "annotation", "qc",
  "arbiter", "arbiter_secondary",
  "coordinator", "fallback",
];

export function ProvidersPanel() {
  // Providers are workspace-global: a single llm_profiles.yaml shared across
  // every project in the workspace. We always pass storeKey=null so the API
  // resolves to the workspace-level file (with project-local fallback).
  const [snapshot, setSnapshot] = useState<ProviderConfigSnapshot | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<string | null>(null);
  const [newProviderKind, setNewProviderKind] = useState<ProviderName>("local_cli");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

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
    const profile = createProviderProfile(newProviderKind, snapshot.profiles.length + 1);
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
            label="Local CLI Global Concurrency"
            value={snapshot.limits.local_cli_global_concurrency}
            onChange={(value) => setSnapshot({ ...snapshot, limits: { local_cli_global_concurrency: value } })}
          />
        </div>
      </div>

      <div className="providers-layout">
        <aside className="provider-list">
          <div className="provider-add-row">
            <select value={newProviderKind} onChange={(event) => setNewProviderKind(event.target.value as ProviderName)}>
              <option value="local_cli">Local CLI</option>
              <option value="openai_responses">OpenAI Responses</option>
              <option value="openai_compatible">OpenAI Compatible</option>
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
              onClick={() => setSelectedProfile(profile.name)}
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
                <button className="view-tab danger" type="button" onClick={deleteProfile} disabled={snapshot.profiles.length <= 1}>
                  Delete
                </button>
              </div>
              <div className="provider-form-grid">
                <TextField label="Name" value={selected.name} onChange={(value) => updateSelected({ name: value })} />
                <SelectField
                  label="Provider"
                  value={selected.provider}
                  options={["local_cli", "openai_responses", "openai_compatible"]}
                  onChange={(value) => updateSelected(providerDefaultsFor(value as ProviderName))}
                />
                {selected.provider === "local_cli" ? (
                  <>
                    <SelectField label="CLI Kind" value={selected.cli_kind ?? "codex"} options={["codex", "claude"]} onChange={(value) => updateSelected({ cli_kind: value as ProviderProfileConfig["cli_kind"] })} />
                    <TextField label="CLI Binary" value={selected.cli_binary ?? ""} onChange={(value) => updateSelected({ cli_binary: value })} />
                    <TextField label="Permission Mode" value={selected.permission_mode ?? ""} onChange={(value) => updateSelected({ permission_mode: value || null })} />
                  </>
                ) : null}
                {selected.provider === "openai_compatible" ? (
                  <SelectField
                    label="Provider Flavor"
                    value={selected.provider_flavor ?? "deepseek"}
                    options={["deepseek", "glm", "minimax"]}
                    onChange={(value) => updateSelected({ provider_flavor: value as ProviderProfileConfig["provider_flavor"] })}
                  />
                ) : null}
                {selected.provider !== "local_cli" ? (
                  <>
                    <PasswordField
                      label="API Key (inline)"
                      value={selected.api_key ?? ""}
                      placeholder={selected.api_key_set ? "set" : "not set"}
                      hint={selected.api_key_set ? "Leave blank to keep current key" : undefined}
                      onChange={(value) => updateSelected({ api_key: value })}
                    />
                    <TextField label="API Key Env" value={selected.api_key_env ?? ""} onChange={(value) => updateSelected({ api_key_env: value })} />
                    <TextField label="Base URL" value={selected.base_url ?? ""} onChange={(value) => updateSelected({ base_url: value })} />
                  </>
                ) : null}
                <TextField label="Model" value={selected.model} onChange={(value) => updateSelected({ model: value })} />
                <TextField label="Reasoning Effort" value={selected.reasoning_effort ?? ""} onChange={(value) => updateSelected({ reasoning_effort: value || null })} />
                <NumberField label="Timeout Seconds" value={selected.timeout_seconds} onChange={(value) => updateSelected({ timeout_seconds: value })} />
                <NumberField label="Max Retries" value={selected.max_retries} onChange={(value) => updateSelected({ max_retries: value })} />
                <NumberField label="Concurrency Limit" value={selected.concurrency_limit} onChange={(value) => updateSelected({ concurrency_limit: value })} />
                <NumberField label="No Progress Timeout" value={selected.no_progress_timeout_seconds} onChange={(value) => updateSelected({ no_progress_timeout_seconds: value })} />
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

function providerDefaultsFor(provider: ProviderName): Partial<ProviderProfileConfig> {
  const defaults = createProviderProfile(provider, 1);
  return {
    provider,
    provider_flavor: defaults.provider_flavor,
    cli_kind: defaults.cli_kind,
    cli_binary: defaults.cli_binary,
    api_key_env: defaults.api_key_env,
    base_url: defaults.base_url,
    model: defaults.model,
    reasoning_effort: defaults.reasoning_effort,
    permission_mode: defaults.permission_mode,
  };
}

function normalizeProfile(profile: ProviderProfileConfig): ProviderProfileConfig {
  if (profile.provider === "local_cli") {
    return { ...profile, provider_flavor: null, api_key_env: null, base_url: null };
  }
  if (profile.provider === "openai_responses") {
    return { ...profile, provider_flavor: null, cli_kind: null, cli_binary: null, permission_mode: null };
  }
  return { ...profile, cli_kind: null, cli_binary: null, permission_mode: null };
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
