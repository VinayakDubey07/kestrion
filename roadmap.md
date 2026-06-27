# Kestrion — Roadmap (Next 3 Months)

**Baseline:** `v0.2.1`, live on PyPI. 75 passing tests. Core engine, `Agent`/`@tool` API, 3 LLM
providers (Ollama live-verified; Anthropic/OpenAI doc-verified only), a live-verified MCP client,
and 4 of 6 planned agentic features (approval chains, timeouts, parallel tool calls, sub-agents)
are built and tested.

**Pace assumption:** weekday evenings + weekend mornings, ~10-13 hrs/week — the same pace
established in the original build plan. Each month below is sized to that, not to an idealized
full-time pace. If GATE exam prep intensifies, the items marked **(compressible)** are the ones to
cut or push first — everything else either fixes a known gap honestly disclosed in the README or
is foundational for what comes after it.

This roadmap is intentionally narrower than the original 9-phase build plan — it picks the subset
of that plan, plus the agentic features added mid-project, that's realistic to actually finish in
12 weeks, not everything that's theoretically left.

---

## Month 1 — Close out the agentic feature set, fix the recurring process gap

**Goal:** finish what's already in motion before starting anything new. Two features are mid-sequence;
finishing them is higher leverage than starting fresh work.

### Week 1-2: Feature 5 — Multi-agent handoff
- Design: distinguish handoff (conversation fully transfers, original agent doesn't resume control)
  from the sub-agent pattern already built (delegation, parent stays in charge)
- Build + test against the same rigor as Features 1-4: a real regression run against all 75
  existing tests before any new code is considered done
- Update `docs/concepts/` with a new doc distinguishing sub-agents vs. handoff, since the two are
  easy to conflate and both now exist

### Week 2-3: Feature 6 — Memory / context compaction
- Deliberately last in the original 6-feature sequence because it touches the same
  `_messages`/`scratch` state everything else depends on — by now, every other feature that
  reads/writes that state exists and can be regression-tested against
- Design decision to settle first: summarize-in-place vs. summarize-and-archive (does compacted
  history stay queryable via the event log, or only the live summary?)
- This is the more architecturally risky of the two remaining features — budget extra time for a
  design pass before writing code, same as parallel tool calls got extra care

### Week 4: `Agent.approve()` — stop being a stub
- Every approval-flow example/test currently hand-rolls `Engine.record_approval` +
  manual checkpoint save. This is the most-repeated piece of boilerplate in the whole codebase at
  this point — worth fixing now that the chain/timeout/sub-agent approval shapes are all final
- Needs a real persistence decision: where does an "approval was granted" fact live before
  `resume()` is called — same store, a side table, or does `approve()` just become a thin wrapper
  around the existing `record_approval` + `store.save()` pattern made into one call

**Exit criteria:** 6/6 agentic features built and tested. `Agent.approve()` is a real one-line
call, not a documented workaround. Full regression suite still green.

---

## Month 2 — Close the verification gaps, MCP server side

**Goal:** the two gaps that are about *trust*, not *features* — claims in the README that are
currently "implemented but unverified" become "verified," and the MCP story becomes two-directional.

### Week 5: Real API key verification for Anthropic and OpenAI **(compressible)**
- A single smoke test per provider, mirroring `test_smoke_ollama.py`'s pattern exactly: real API
  call, real agent run, auto-skip if no key is present (so CI doesn't require paid secrets)
- Low engineering risk, meaningful credibility gain — closes the most-repeated caveat in the
  README's "Known gaps" section
- If time is tight this month, this is the item to defer — it doesn't block anything else

### Week 6-7: `mcp/server.py` — expose a Kestrion Agent as an MCP server
- The other half of Phase 3, deferred since the MCP client work
- Concrete target: a Kestrion agent should be callable as a tool from Claude Code or Codex CLI
- Needs a real worked example, not just unit tests — wire this into the kubectl-over-SSH server
  from your own earlier work, the same "real server, not just a fixture" standard the MCP client
  was held to

### Week 8: `core/errors.py` — a real exception hierarchy
- Currently `ApprovalRequired` lives in `engine.py`; raw `ValueError`/`NotImplementedError` are
  raised elsewhere for missing checkpoints, bad store URLs, unimplemented approval persistence
- Define `KestrionError` as a base, with specific subclasses, so calling code can catch broadly or
  specifically — small, contained, improves the experience of anyone building on top of this
- `agent/graph.py` stays deferred — still not a capability gap (raw `Node` classes cover the same
  workflows today), lower priority than everything else this month

**Exit criteria:** every provider in the README's comparison table is either live-verified or
explicitly marked otherwise — no silent gaps. MCP works in both directions. Error handling is
consistent across the codebase, not ad hoc per module.

---

## Month 3 — Documentation completeness, then pick ONE production-readiness feature

**Goal:** a stranger should be able to clone the repo and get productive without asking questions
— and the most-requested remaining production feature gets at least a working first pass.

### Week 9: Finish the docs folder
- `docs/index.md` — still empty; a short landing page tying `getting-started.md`, the concept docs,
  and `architecture.md` together
- `docs/api-reference/` — generate from docstrings now (mkdocs + mkdocstrings, or similar), rather
  than hand-write — by month 3, the API surface is stable enough that this won't immediately go stale
- A concepts doc for parallel tool calls and one for sub-agents/handoff specifically — currently
  covered only inside `approval-gates.md` and the README, deserve their own pages given how much
  surface area they now represent

### Week 10-12: ONE of the following — not both, pick based on what's actually needed **(compressible: pick the smaller of the two if time is short)**
- **Option A — Scheduler (Phase 5 from the original plan):** rate-limited, concurrent execution
  across *multiple separate agent runs* sharing one provider quota (distinct from parallel tool
  calls within one run, which is already built). This is the harder, more valuable one for the
  "production-grade" claim.
- **Option B — CLI + `kestrion deploy` (Phase 6):** `kestrion init`, `kestrion run`,
  `kestrion deploy --target k8s`. Lower architectural risk, more visible/demoable result, plays to
  the Kubernetes background that's part of the project's differentiation story.

Decide between these at the start of month 3 based on which gap is getting asked about more — by
users, in issues, or in your own use of the project — rather than deciding now in the abstract.

**Exit criteria:** a stranger can clone the repo, follow `docs/getting-started.md` end to end with
no outside help, and the chosen Option A/B feature has its own passing test suite with the same
rigor every other feature in this project has had.

---

## Explicitly NOT in this 3-month window

Named here so scope creep has something to point at instead of silently expanding the roadmap
mid-quarter:

- Postgres-backed storage (`store/postgres_store.py`) — real, but lower urgency than the items
  above; SQLite has not yet been a demonstrated bottleneck
- A public docs *site* (vs. the docs *folder*, which Month 3 does address) — needs hosting/CI
  decisions that are a project of their own
- Public launch push (Show HN, etc.) — premature until the Month 3 exit criteria are met
- A 7th/8th agentic feature beyond the original 6 — resist adding new feature ideas until the
  existing 6 are fully done, verified, and documented

## Risk register

1. **Memory/context compaction (Month 1, weeks 2-3) is the single riskiest item on this roadmap** —
   flagged as architecturally trickier than the other features back when the original 6-feature
   sequence was planned. If it runs over, let it run over rather than rushing it; everything in
   Month 2 can shift a week without consequence.
2. **GATE exam prep timeline isn't visible from here** — if weekend-morning availability drops as
   the exam approaches, cut from the **(compressible)** items first, in order: real-API-key
   verification, then the Month 3 Option A/B feature. Don't cut the agentic-feature completion or
   the docs work — those are closer to "finishing what's started" than "new scope."
3. **Scope discipline matters more than speed here.** This project's strongest asset so far has
   been that every shipped feature is genuinely tested, not just claimed — the regression-suite
   discipline from Features 1-4 should continue, even if it means fewer features ship in 3 months
   than this document lists as a stretch goal.