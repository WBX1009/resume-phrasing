# Release acceptance checklist

This checklist is intentionally executable. A release is not complete until
the commands and assertions below have current evidence in the run artifact.

Scope: this checklist covers the public model-result snapshot/quality CLI. Its
passing tests and benchmark do not prove the résumé's separate case-level event
evaluation, LangGraph case workflow, hosted LangSmith experiments, GaussDB
workload, or FastAPI/SSE service. Those claims use the strict, separate gate in
[`resume_alignment_acceptance.md`](resume_alignment_acceptance.md).

## Commands

```powershell
$py='python'
& $py -m pytest -q
& $py -m src.quality_cli catalog validate
& $py -m src.quality_cli evaluate --output-root output\quality_runs
& $py scripts/benchmark_quality_pipeline.py --runs 10 --threshold 1.3
& $py scripts/fuzz_quality_pipeline.py --rounds 100 --seed 20260907
& $py scripts/goal_watch.py --once
```

## Hard gates

- Existing and quality-pipeline tests pass (82 currently collected).
- Catalog reports exactly 104 total, 61 sampled and 43 unsupported-no-sample.
- All 61 sampled model entries have unique strategy keys and are processed in
  the all-snapshot run.
- The user-facing `report.md` exists, is self-contained and loadable; JSON/trace/
  manifest/checkpoint remain loadable internal evidence. Existing HTML/DOCX
  projections are compatibility artifacts, not a reason to block the core
  Markdown chain.
- Claim reference validation has zero missing fact IDs.
- Fault suite passes empty data, malformed data, wrong data types, duplicates,
  unknown tools and orphan references.
- The report manifest records snapshot hash, policy counts, elapsed time and
  online budget usage.
- An intentionally interrupted running checkpoint resumes without duplicate
  model evidence and matches a clean-run replay fingerprint.
- A running checkpoint rejects missing committed state records and mismatched
  case filters instead of silently emitting a partial or cross-case report.
- The emitted report passes the versioned JSON structural contract, and a
  malformed report is rejected by the contract validator.
- The frozen API contract documents request/response fields, state transitions,
  error codes, idempotency and CLI/HTTP mapping; implementation changes must
  update its contract tests.
- Identifier-like fields (phone/account/card/credential numbers) are excluded
  from numeric metric generation and covered by a regression test.
- The public CLI catalog, resume and replay commands return the expected
  status codes and validated JSON payloads.
- Git-tracked files contain no local credential material, and known local key
  files are covered by `.gitignore`.
- The default 8,000-character context pack retains every Fact ID and reports
  zero budget-exceeded events in the all-snapshot run.
- Corrupt cached LLM payloads are discarded and safely re-requested or
  downgraded; they never bypass the structured advisory contract.
- Date/time fields are excluded from amount-like numeric summaries, while
  duration fields remain eligible for deterministic metrics.
- JSONL catalog/snapshot loading rejects non-object records and invalid integer
  keys with a file/row diagnostic.
- Completed artifacts cross-check run identity, snapshot hash, status, model
  count and event/model IDs across `report.json`, `run_manifest.json`,
  `checkpoint.json`, `checkpoint_models.jsonl`, `trace.jsonl`, Markdown and
  HTML; tampering any projection must fail replay validation.
- Public replay fails closed with structured diagnostics when the report is
  missing required sections, trace event IDs are missing, or checkpoint model
  IDs drift from the report.
- The versioned report contract validates Fact shape/types and rejects Claim
  references to unknown Fact or Model IDs.
- Markdown/HTML projections and trace payloads are regenerated or compared
  against `report.json`; preserving a run ID while changing content must fail.
- Report model IDs are unique, and model-row coverage counts agree with both
  the report coverage block and manifest profile counts.
- DOCX is regenerated from the JSON source and its document XML is compared;
  preserving a run ID while changing DOCX content must fail replay.
- The 10-run performance benchmark also requires all deterministic replay
  fingerprints to be identical.
- Same-second or simultaneous CLI runs must reserve distinct output directories;
  concurrent cache publishers must leave a valid JSON entry and no temporary
  files, including on Windows sharing-violation paths.
- Concurrent LLM callers must update the in-process budget ledger atomically;
  a check-then-increment race must not exceed the configured call/token caps.
- The bounded adapter must reject non-positive or non-integer `max_tokens`
  before reserving budget or invoking a provider.
- Budget ledger caps/counters and provider-reported token counts must be
  non-negative integers; invalid values fail closed before reconciliation.
- Invalid provider token usage must not publish a successful cache payload or
  disappear from the budget ledger.
- Checkpoint cleanup and all text/DOCX projections must be atomically published
  so an interrupted write cannot replace a valid artifact with a torn file.
- The executable goal gate must reject negative/invalid aggregate online-budget
  fields and refuse completion when calls exceed 200 or tokens exceed
  2,000,000, even if elapsed time and other acceptance families pass.
- No `.env` or credential-like file is tracked by Git.
- Completion remains blocked until the wall-clock timestamp in
  `docs/execution/goal_state.json` is at least 24 hours after `started_at`.
- Every acceptance family listed by `check_goal_gate.py` must carry a value
  beginning with `passed`; missing, failed or arbitrary status values block
  completion even after the time gate elapses.
- The deterministic malformed-input fuzz harness must report `valid: true` for
  100 rounds across all 104 profiles with no exception or orphan Claim.
- Public `catalog validate` and `evaluate` calls against a missing or malformed
  data directory must return structured JSON and exit code 2, not an uncaught
  traceback.
- Evaluation assets must keep dev, validation, backtest and bad-case roles
  distinct; a bad case is closed only after a fix, regression test and review
  note are recorded.
- An external blocker may receive at most two purposeful attempts. After the
  second failed attempt, the command, error, impact and workaround must be
  recorded in `docs/evaluation/bad_case_register.md`, and execution must return
  to the offline Markdown chain.
