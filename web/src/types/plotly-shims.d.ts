// Type shims for plotly.js-dist-min and the react-plotly.js factory.
// plotly.js-dist-min ships no .d.ts; we use @types/plotly.js for type info
// but import the minimized bundle at runtime via the factory pattern.

declare module "plotly.js-dist-min" {
  export * from "plotly.js";
  import Plotly from "plotly.js";
  export default Plotly;
}

declare module "react-plotly.js/factory" {
  import type * as Plotly from "plotly.js";
  import type { ComponentType } from "react";
  import type { PlotParams } from "react-plotly.js";
  const factory: (plotly: typeof Plotly) => ComponentType<PlotParams>;
  export default factory;
}
