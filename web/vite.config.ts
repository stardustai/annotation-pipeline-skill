import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.VITE_API_TARGET ?? "http://127.0.0.1:8509",
    },
  },
  build: {
    // Plotly is intrinsically large (~3 MB / 1.5 MB gzip) and lives in
    // its own lazy chunk (ScatterSubTab) that only loads when the
    // operator opens the Distribution → Scatter plot sub-tab. Bump the
    // warning threshold to suppress the default 500 KB nag for chunks
    // we've already deferred behind user interaction.
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "node",
  },
});
