import type { PipelineView } from "../types";

// Visual, read-only diagram of the project's annotation workflow. Data-driven
// from the provider snapshot's `pipeline` view: multi-annotation projects
// render annotators → consensus → arbiter → accept (QC disabled); classic
// (replicas=1) projects render annotation → QC → arbiter → accept.

const C = {
  tealFill: "#E1F5EE", tealBorder: "#0F6E56", tealText: "#04342C",
  blueFill: "#E6F1FB", blueBorder: "#185FA5", blueText: "#042C53",
  greenFill: "#EAF3DE", greenBorder: "#3B6D11", greenText: "#173404",
  grayFill: "#F1EFE8", grayBorder: "#B4B2A9", grayText: "#5F5E5A",
  amberFill: "#FAEEDA", amberBorder: "#854F0B", amberText: "#412402",
  arrow: "#5F5E5A",
};

type Box = {
  x: number; y: number; w: number; h: number;
  title: string; lines: string[];
  fill: string; border: string; text: string; dashed?: boolean;
};

function Node({ b }: { b: Box }) {
  return (
    <g>
      <rect
        x={b.x} y={b.y} width={b.w} height={b.h} rx={8}
        fill={b.fill} stroke={b.border} strokeWidth={1.5}
        strokeDasharray={b.dashed ? "5 4" : undefined}
      />
      <text x={b.x + 12} y={b.y + 20} fontSize={12.5} fontWeight={500} fill={b.text}>
        {b.title}
      </text>
      {b.lines.map((ln, i) => (
        <text key={i} x={b.x + 12} y={b.y + 38 + i * 14} fontSize={10.5} fill={b.border}>
          {ln}
        </text>
      ))}
    </g>
  );
}

function Arrow({ x1, y, x2, color = C.arrow, dashed = false, marker = "wfArrow" }:
  { x1: number; y: number; x2: number; color?: string; dashed?: boolean; marker?: string }) {
  return (
    <line
      x1={x1} y1={y} x2={x2} y2={y} stroke={color} strokeWidth={1.5}
      strokeDasharray={dashed ? "4 3" : undefined} markerEnd={`url(#${marker})`}
    />
  );
}

export function WorkflowDiagram({ pipeline }: { pipeline?: PipelineView | null }) {
  if (!pipeline) return null;
  const multi = pipeline.mode === "multi-annotation";

  // First stage box: multi-annotation lists annotators + keep threshold.
  // Model names only — the Stage Targets grid below shows the full
  // target → profile mapping, so the diagram stays uncluttered.
  const annLines = multi
    ? [
        ...pipeline.annotators.slice(0, 3).map((a) => a.profile ?? a.target),
        ...(pipeline.annotators.length > 3 ? [`+${pipeline.annotators.length - 3} more`] : []),
        `keep ≥ ${pipeline.keep_threshold}`,
      ]
    : [pipeline.annotators[0]?.profile ?? "—"];
  const annH = Math.max(54, 30 + annLines.length * 14);
  const annTitle = multi ? `Multi-annotation ×${pipeline.replicas}` : "Annotation";

  const arbiterLine = pipeline.arbiter.profile ?? "—";
  const acceptLine = pipeline.accept_directly ? "direct · no QC" : "after QC";

  return (
    <svg
      viewBox="0 0 680 280" width="100%" role="img"
      style={{ maxWidth: 680, display: "block" }}
      fontFamily="inherit"
    >
      <title>{`${pipeline.mode} workflow`}</title>
      <desc>
        {multi
          ? "Two or more annotators run in parallel, a consensus keeps agreed spans, an arbiter resolves the rest, and the result is accepted directly. QC is disabled."
          : "A single annotator runs, QC reviews, an arbiter resolves disputes, and the result is accepted."}
      </desc>
      <defs>
        <marker id="wfArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={C.arrow} />
        </marker>
        <marker id="wfArrowAmber" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill={C.amberBorder} />
        </marker>
      </defs>

      {/* flow arrows */}
      <Arrow x1={186} y={105} x2={212} />
      <Arrow x1={332} y={105} x2={358} />
      <Arrow x1={478} y={105} x2={504} />
      {/* arbiter → human review (failure branch) */}
      <line x1={423} y1={143} x2={423} y2={196} stroke={C.amberBorder} strokeWidth={1.5}
        strokeDasharray="4 3" markerEnd="url(#wfArrowAmber)" />

      {/* stage 1 */}
      <Node b={{
        x: 36, y: 105 - annH / 2, w: 150, h: annH,
        title: annTitle, lines: annLines,
        fill: C.tealFill, border: C.tealBorder, text: C.tealText,
      }} />

      {/* stage 2: consensus (multi) or QC (single) */}
      {multi ? (
        <Node b={{ x: 212, y: 78, w: 120, h: 54, title: "Consensus", lines: ["unanimous kept"],
          fill: C.tealFill, border: C.tealBorder, text: C.tealText }} />
      ) : (
        <Node b={{ x: 212, y: 78, w: 120, h: 54, title: "QC",
          lines: [pipeline.qc.profile ?? "—"],
          fill: C.tealFill, border: C.tealBorder, text: C.tealText }} />
      )}

      {/* stage 3: arbiter */}
      <Node b={{ x: 358, y: 78, w: 120, h: 54,
        title: multi ? "Arbiter" : "Arbiter · dispute", lines: [arbiterLine],
        fill: C.blueFill, border: C.blueBorder, text: C.blueText }} />

      {/* stage 4: accept */}
      <Node b={{ x: 504, y: 78, w: 120, h: 54, title: "Accept", lines: [acceptLine],
        fill: C.greenFill, border: C.greenBorder, text: C.greenText }} />

      {/* QC disabled callout (multi only) */}
      {multi && !pipeline.qc.enabled ? (
        <Node b={{ x: 212, y: 196, w: 120, h: 50, title: "QC", lines: ["disabled · bypassed"],
          fill: C.grayFill, border: C.grayBorder, text: C.grayText, dashed: true }} />
      ) : null}

      {/* human review branch */}
      <Node b={{ x: 363, y: 196, w: 120, h: 50, title: "Human review",
        lines: ["on failure"],
        fill: C.amberFill, border: C.amberBorder, text: C.amberText }} />

      {/* legend */}
      <text x={504} y={210} fontSize={10} fill={C.grayText}>teal · active stage</text>
      <text x={504} y={224} fontSize={10} fill={C.grayText}>green · terminal accept</text>
      <text x={504} y={238} fontSize={10} fill={C.grayText}>gray dashed · disabled</text>
    </svg>
  );
}
