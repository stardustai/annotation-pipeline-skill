import { CSSProperties } from "react";

interface RangeStepperProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Width of the range track itself (e.g. "160px"). */
  width?: string;
  disabled?: boolean;
  title?: string;
  /** Decimal places used to round after a step, avoiding float drift. */
  decimals?: number;
  id?: string;
  accentColor?: string;
}

const BTN_STYLE: CSSProperties = {
  width: "1.4rem",
  height: "1.4rem",
  lineHeight: 1,
  padding: 0,
  fontSize: "0.95rem",
  fontWeight: 600,
  cursor: "pointer",
  border: "1px solid var(--border, #d1d5db)",
  borderRadius: "4px",
  background: "var(--surface, #fff)",
  color: "inherit",
  flex: "0 0 auto",
};

/**
 * A range slider flanked by − / + buttons so values can be nudged precisely
 * one step at a time instead of fighting the drag handle. Clamps to
 * [min, max] and rounds to `decimals` places to avoid float accumulation.
 */
export function RangeStepper({
  value,
  min,
  max,
  step,
  onChange,
  width = "120px",
  disabled = false,
  title,
  decimals = 0,
  id,
  accentColor,
}: RangeStepperProps) {
  const clamp = (v: number) => {
    const bounded = Math.max(min, Math.min(max, v));
    const factor = Math.pow(10, decimals);
    return Math.round(bounded * factor) / factor;
  };
  const nudge = (delta: number) => onChange(clamp(value + delta));

  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
      <button
        type="button"
        style={BTN_STYLE}
        onClick={() => nudge(-step)}
        disabled={disabled || value <= min}
        title="Decrease"
        aria-label="Decrease"
      >
        −
      </button>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!isNaN(v)) onChange(clamp(v));
        }}
        style={{ width, cursor: "pointer", verticalAlign: "middle", accentColor: accentColor ?? "var(--accent, #2563eb)" }}
        disabled={disabled}
        title={title}
      />
      <button
        type="button"
        style={BTN_STYLE}
        onClick={() => nudge(step)}
        disabled={disabled || value >= max}
        title="Increase"
        aria-label="Increase"
      >
        +
      </button>
    </span>
  );
}
