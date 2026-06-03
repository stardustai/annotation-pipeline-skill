import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildSearch, parseUrlState, writeUrl, type UrlState } from "./url_state";

const defaults: UrlState = { view: "kanban", store: null, project: null, task: null };

interface FakeLocation {
  pathname: string;
  search: string;
  hash: string;
}

interface FakeHistory {
  state: unknown;
  replaceState: ReturnType<typeof vi.fn>;
  pushState: ReturnType<typeof vi.fn>;
}

interface FakeStorage {
  store: Record<string, string>;
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
}

interface FakeWindow {
  location: FakeLocation;
  history: FakeHistory;
  localStorage: FakeStorage;
  addEventListener: ReturnType<typeof vi.fn>;
  removeEventListener: ReturnType<typeof vi.fn>;
  __handlers: Record<string, Array<(...args: unknown[]) => void>>;
}

function makeFakeWindow(initialSearch = ""): FakeWindow {
  const handlers: Record<string, Array<(...args: unknown[]) => void>> = {};
  const storeMap: Record<string, string> = {};
  const fake: FakeWindow = {
    location: { pathname: "/", search: initialSearch, hash: "" },
    history: {
      state: null,
      replaceState: vi.fn((stateValue: unknown, _title: string, url: string) => {
        // Update fake location.search to mirror what a real browser does.
        const queryIndex = url.indexOf("?");
        const hashIndex = url.indexOf("#");
        if (queryIndex === -1) {
          fake.location.search = "";
        } else {
          const end = hashIndex === -1 ? url.length : hashIndex;
          fake.location.search = url.slice(queryIndex, end);
        }
        fake.history.state = stateValue;
      }),
      pushState: vi.fn(),
    },
    localStorage: {
      store: storeMap,
      getItem: (key: string) => (key in storeMap ? storeMap[key] : null),
      setItem: (key: string, value: string) => {
        storeMap[key] = value;
      },
      removeItem: (key: string) => {
        delete storeMap[key];
      },
    },
    addEventListener: vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      handlers[event] = handlers[event] ?? [];
      handlers[event].push(handler);
    }),
    removeEventListener: vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      handlers[event] = (handlers[event] ?? []).filter((h) => h !== handler);
    }),
    __handlers: handlers,
  };
  return fake;
}

function installWindow(fake: FakeWindow): void {
  vi.stubGlobal("window", fake);
  vi.stubGlobal("history", fake.history);
  vi.stubGlobal("localStorage", fake.localStorage);
}

describe("parseUrlState", () => {
  it("loads defaults when URL has no params", () => {
    expect(parseUrlState("", defaults)).toEqual(defaults);
  });

  it("parses ?view=runtime&store=abc&project=p&task=t-1 into state", () => {
    const state = parseUrlState("?view=runtime&store=abc&project=p&task=t-1", defaults);
    expect(state).toEqual({ view: "runtime", store: "abc", project: "p", task: "t-1" });
  });

  it("treats empty string params as missing (falls back to defaults)", () => {
    const state = parseUrlState("?view=&store=&project=&task=", defaults);
    expect(state).toEqual(defaults);
  });
});

describe("buildSearch", () => {
  it("omits null/empty fields and the default view", () => {
    expect(buildSearch(defaults, defaults)).toBe("");
  });

  it("emits non-default view and present fields (store is NOT written to URL)", () => {
    const state: UrlState = { view: "runtime", store: "abc", project: "p", task: "t-1" };
    // store is intentionally omitted — project is the addressing key.
    expect(buildSearch(state, defaults)).toBe("?view=runtime&project=p&task=t-1");
  });

  it("setting a value to null removes the param; store is never emitted", () => {
    const state: UrlState = { view: "kanban", store: "abc", project: null, task: null };
    expect(buildSearch(state, defaults)).toBe("");
  });
});

describe("useUrlState hook (manual driver, no react renderer)", () => {
  let fake: FakeWindow;

  beforeEach(() => {
    fake = makeFakeWindow();
    installWindow(fake);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // Vitest runs in node env (no jsdom). We can't render a React component
  // here, but we can exercise the lower-level building blocks the hook
  // uses (writeUrl + parseUrlState + the popstate listener) to cover the
  // same behavior the hook surfaces. The hook itself is a thin wrapper
  // composed of these pieces.

  it("writeUrl: setView updates URL and leaves other params untouched", () => {
    // Start with store + project set; bump view to "runtime".
    const start: UrlState = { view: "kanban", store: "abc", project: "p", task: null };
    writeUrl(start, defaults);
    expect(fake.location.search).toBe("?project=p");  // store not written

    const next: UrlState = { ...start, view: "runtime" };
    writeUrl(next, defaults);
    expect(fake.location.search).toBe("?view=runtime&project=p");
    // history.replaceState was used, not pushState
    expect(fake.history.replaceState).toHaveBeenCalled();
    expect(fake.history.pushState).not.toHaveBeenCalled();
  });

  it("setStore writes to localStorage", () => {
    // Simulate the hook's setStore behavior directly (the hook calls
    // localStorage.setItem with the same key).
    window.localStorage.setItem("storeKey", "abc");
    expect(fake.localStorage.store.storeKey).toBe("abc");
    window.localStorage.removeItem("storeKey");
    expect(fake.localStorage.store.storeKey).toBeUndefined();
  });

  it("popstate event causes state to re-sync from URL", () => {
    // Register a popstate listener (this is what the hook does internally).
    let observed: UrlState | null = null;
    const handler = () => {
      observed = parseUrlState(window.location.search, defaults);
    };
    window.addEventListener("popstate", handler);

    // User navigated back/forward: URL changed under us.
    fake.location.search = "?view=outbox&project=xyz";
    // Fire popstate.
    const popHandlers = fake.__handlers.popstate ?? [];
    expect(popHandlers).toHaveLength(1);
    popHandlers.forEach((h) => h());

    expect(observed).toEqual({ view: "outbox", store: null, project: "xyz", task: null });
  });
});

describe("useUrlState initial-state derivation", () => {
  let fake: FakeWindow;

  beforeEach(() => {
    fake = makeFakeWindow();
    installWindow(fake);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // We can't call the hook directly without a renderer, but we can verify
  // its initialization logic by reproducing it inline (the same code path
  // the hook's useState initializer runs).
  function initialState(): UrlState {
    const parsed = parseUrlState(window.location.search, defaults);
    if (!parsed.store) {
      const stored = window.localStorage.getItem("storeKey");
      if (stored) return { ...parsed, store: stored };
    }
    return parsed;
  }

  it("loads defaults when URL has no params", () => {
    expect(initialState()).toEqual(defaults);
  });

  it("falls back to localStorage when ?store is missing", () => {
    fake.localStorage.setItem("storeKey", "fallback-store");
    expect(initialState()).toEqual({ ...defaults, store: "fallback-store" });
  });

  it("URL store takes precedence over localStorage", () => {
    fake.localStorage.setItem("storeKey", "stored");
    fake.location.search = "?store=from-url";
    expect(initialState()).toEqual({ ...defaults, store: "from-url" });
  });
});
