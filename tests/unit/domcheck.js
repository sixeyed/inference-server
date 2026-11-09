// Minimal DOM stub: runs app.js and fails on any top-level throw. Catches the
// class of bug that leaves the page rendered but completely unpopulated.
const el = new Proxy({}, {
  get: (t, k) => {
    if (k === "classList") return { toggle() {}, add() {}, remove() {} };
    if (k === "addEventListener") return () => {};
    if (k === "dataset") return {};
    if (k === "querySelectorAll") return () => [];
    if (k === "append") return () => {};
    return typeof k === "string" && k.startsWith("inner") ? "" : el;
  },
  set: () => true,
});
global.document = {
  getElementById: () => el,
  querySelectorAll: () => [],
  createElement: () => el,
  addEventListener: () => {},
};
global.window = { addEventListener: () => {} };
global.location = { hash: "" };
global.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({
  data: [], models: [], preempted: {}, requests: [], series: [], sessions: [],
  params: [], strip: [], machine: {}, state: "idle", total: 0, offset: 0 }) });
global.setInterval = () => {};
global.performance = { now: () => 0 };
require(process.argv[2]);
console.log("  app.js executes with no top-level error");
