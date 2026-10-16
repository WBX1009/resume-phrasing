# Evidence-first quality pipeline

面向业务读者的主交付是自包含 `report.md`；JSON、trace、manifest 和
checkpoint 是机器验收/恢复所需的内部证据。HTML/DOCX 仍可由现有代码生成
以保持兼容，但不是后续核心链路的阻塞项。

## Scope

The production candidate is a research CLI over the existing model-result
snapshot. It does not recompute the upstream business models and it does not
pretend that the 2,561 sampled rows represent production-scale recall.

The registry contains 104 model definitions:

- 61 models with sample rows are executable and have an explicit strategy key,
  family, field contract, operator list, reasoning policy, report sections and
  cross-model dependencies.
- 43 models without sample rows are registered as
  `unsupported_no_sample`. They are visible in coverage reports but cannot
  invoke an LLM or create a conclusion.

## Runtime flow

```text
snapshot validation
  -> model strategy registry
  -> deterministic fact pack
  -> bounded policy (deterministic / ReAct / Plan-Solve / Plan-ReAct-Reflect)
  -> whitelisted observation tools
  -> evidence ledger and conservative cross-model links
  -> claim validation
  -> JSON source of truth（机器验收）
  -> Markdown 主交付（HTML/DOCX 仅兼容投影）
```

Facts are created only by local operators. A future online LLM adapter may
propose a bounded action list, but tool names are checked against the profile
whitelist and the result is stored as an advisory trace event. It can never
create a `Fact`, change a number, or bypass the evidence validator.

The HelloAgents concepts are represented by local policy classes rather than a
runtime dependency. This avoids the framework's `openai<2` constraint and
keeps the production package independent. The mapping is:

| Task characteristic | Policy | Safety boundary |
| --- | --- | --- |
| Simple aggregation/distribution | deterministic | no LLM call required |
| Ambiguous bounded tool choice | ReAct | max steps and tool whitelist |
| Temporal/sequence or graph plan | Plan-Solve | explicit dependency DAG |
| Cross-model case synthesis | Plan-ReAct-Reflect | reflection can only accept, downgrade or request verification |

## CLI

```powershell
python -m src.quality_cli catalog validate
python -m src.quality_cli run --case-id 13 --output-root output/quality_runs
python -m src.quality_cli run --model-ids 5,10,14 --output-root output/quality_runs
python -m src.quality_cli evaluate --output-root output/quality_runs
python -m src.quality_cli replay output/quality_runs/<run_id>
python scripts/benchmark_quality_pipeline.py --runs 10 --threshold 1.3
python scripts/fuzz_quality_pipeline.py --rounds 100 --seed 20260907
```

Each run contains `report.json`, `report.md`, `report.html`, `report.docx`,
`trace.jsonl`, `run_manifest.json`, `checkpoint.json`, and an append-only
`checkpoint_models.jsonl` state log. If a process stops while the checkpoint
is `running`, the CLI can resume it; the recovery regression requires the
resumed replay fingerprint to match a clean run. `report.json` is the only
source of truth; all human-facing projections are generated from it.

## Release gates

The release candidate must keep all of the following true:

1. 61 sampled profiles execute and 43 unsupported profiles safely downgrade.
2. Every Claim references an existing Fact; malformed, incomplete and empty
   input never becomes a fabricated conclusion.
3. Cross-model links use at least two independent model facts and remain
   `candidate_needs_verification`.
4. Invalid tools, duplicate actions, budget exhaustion and LLM failures are
   blocked or downgraded with a trace reason.
5. Offline evaluation, fault injection, report-format validation and visual
   DOCX review pass; live calls stay within 200 calls / 2,000,000 tokens.
6. A running checkpoint can resume without duplicate model evidence or replay
   fingerprint drift.
7. Every emitted JSON report passes the versioned structural contract before
   format and claim gates are accepted.
8. Identifier-like columns are not promoted to numeric amount metrics, keeping
   phone/account/card values from creating false threshold evidence.
9. Resume refuses incomplete committed state logs and case-filter mismatches;
   recovery never silently changes the evidence scope.
10. Context packing reserves space for all remaining Fact/Observation
    references; the default all-snapshot run has zero budget-exceeded events.
11. The bounded LLM adapter discards malformed cache payloads before
    whitelist validation and preserves the provider/budget failure trace.
12. Date/time identifiers are not converted into amount-like metrics; duration
    fields remain available to the numeric operator.
13. Catalog and snapshot JSONL boundaries reject malformed record shapes and
    identify invalid integer keys before building profiles.
14. Completed report projections are cross-checked against the run manifest,
    checkpoint and JSONL trace; run identity, status and processed model IDs
    must agree, and replay rejects tampered artifacts.
15. The public replay path fails closed on structurally malformed reports,
   missing trace event IDs and checkpoint model-set drift, returning a
   structured invalid payload instead of an exception.
16. Focused model runs record their selected IDs in the manifest and reject
   unknown or empty model filters before execution.
17. Resume rejects a caller-supplied case/model filter that differs from the
   checkpoint scope, including a full checkpoint with no selected-model set.
18. Checkpoint and manifest JSON, checkpoint-state cleanup and text projections
   are written via same-directory replacement so a process interruption cannot
   publish a partially serialized canonical file.
19. The report contract validates Fact field shape/types and rejects orphan
   Claim references to unknown Fact or Model IDs.
20. Markdown/HTML projections and trace payloads are checked against the JSON
   source of truth, so content-only tampering cannot pass replay.
21. Report model IDs are unique and coverage/model/manifest counts agree,
   preventing a duplicate-and-omitted model from passing as full coverage.
22. The DOCX projection is regenerated and compared at document-XML level, so
   content-only DOCX tampering cannot pass replay.
23. The repeated performance benchmark requires deterministic replay
   fingerprints to remain identical across all runs.
24. The executable goal gate requires every acceptance family to be explicitly
   `passed*`; missing or failed values cannot be released by time alone.
25. The seeded malformed-input fuzz harness must complete all 100 rounds across
   every registered profile without an exception, orphan Claim or fabricated
   conclusion.
26. New runs reserve a collision-safe timestamp directory, and concurrent
    LLM cache publishers use unique temporary files with bounded Windows
    replacement retries; same-second and concurrent-write regressions must pass.
27. Public catalog and evaluate commands return structured JSON with exit code
    2 for missing or malformed data instead of leaking a traceback at the CLI
    boundary.
28. The in-process online budget ledger serializes reserve/reconcile updates;
    concurrent callers cannot pass the call or token cap through a check-then-
    increment race.
29. The bounded adapter rejects non-positive or non-integer `max_tokens` before
    cache lookup or transport, so malformed caller input cannot create a
    negative or fractional budget estimate.
30. Budget ledger caps and counters are validated as non-negative integers at
    construction, and provider-reported token counts are validated before
    reconciliation.
31. A provider returning an invalid token usage value fails closed without
    writing a cache success payload; the reserved call remains visible for
    conservative accounting.

The current performance number is an offline snapshot benchmark. A separate
50,000-row synthetic fact-packing stress (three runs, mean 1.662 s, max
1.674 s) is recorded as an engineering signal only; it is not a production
latency or recall claim.

新增策略、数据或外部依赖遇到阻塞时，最多进行两次有明确变量变化的尝试；
仍失败就写入 `docs/evaluation/bad_case_register.md`，转回离线 Markdown
主链路，不重复运行同一阻塞分支。
