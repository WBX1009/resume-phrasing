# DeepAssist Evaluation Report

> This report is generated from local eval artifacts. Only numbers in this file should be copied into README or resume bullets.

## Run Summary

- Run ID: `smoke_public_eval`
- Mode: `offline_processed`
- Status: `success`
- Processed dir: `evals/fixtures/processed`
- Generated at: `2026-06-18T04:56:51.149506+00:00`
- Live backend eval: `False`

## Dataset Warning

This run used the checked-in smoke fixture, not downloaded public benchmark data. Treat the metrics below as pipeline validation only.

## Aggregate Metrics

| Metric | Value |
|---|---:|
| `agent_argument_validity` | 1.0 |
| `agent_final_answer_accuracy` | 1.0 |
| `agent_hitl_resume_success` | 1.0 |
| `agent_pass_rate` | 1.0 |
| `agent_route_accuracy` | 1.0 |
| `agent_tool_choice_accuracy` | 1.0 |
| `agent_tool_execution_success` | 1.0 |
| `rag_citation_presence` | 1.0 |
| `rag_exact_match` | 0.0 |
| `rag_pass_rate` | 1.0 |
| `rag_source_hit_rate` | 1.0 |
| `rag_token_f1` | 0.8 |
| `retrieval_after_rerank_ndcg@10` | 1.0 |
| `retrieval_before_rerank_ndcg@10` | 0.6309 |
| `retrieval_hit_rate@10` | 1.0 |
| `retrieval_hit_rate@5` | 1.0 |
| `retrieval_mrr@10` | 1.0 |
| `retrieval_ndcg@10` | 1.0 |
| `retrieval_pass_rate` | 1.0 |
| `retrieval_recall@10` | 1.0 |
| `retrieval_recall@5` | 1.0 |

## Suite Details

### retrieval

- Evaluated: `1`
- Passed: `1`
- Pass rate: `1.0`

| Metric | Value |
|---|---:|
| `after_rerank_ndcg@10` | 1.0 |
| `before_rerank_ndcg@10` | 0.6309 |
| `hit_rate@10` | 1.0 |
| `hit_rate@5` | 1.0 |
| `mrr@10` | 1.0 |
| `ndcg@10` | 1.0 |
| `recall@10` | 1.0 |
| `recall@5` | 1.0 |

### rag

- Evaluated: `1`
- Passed: `1`
- Pass rate: `1.0`

| Metric | Value |
|---|---:|
| `citation_presence` | 1.0 |
| `exact_match` | 0.0 |
| `source_hit_rate` | 1.0 |
| `token_f1` | 0.8 |

### agent

- Evaluated: `1`
- Passed: `1`
- Pass rate: `1.0`

| Metric | Value |
|---|---:|
| `argument_validity` | 1.0 |
| `final_answer_accuracy` | 1.0 |
| `hitl_resume_success` | 1.0 |
| `route_accuracy` | 1.0 |
| `tool_choice_accuracy` | 1.0 |
| `tool_execution_success` | 1.0 |

## Failure Sample

No failed cases in this run.

## Config Snapshot

```json
{
  "suite": "all",
  "sample_size": 20,
  "api_base": null,
  "created_at": "2026-06-18T04:55:17.642264+00:00"
}
```

## Resume-safe Summary

Use this section only after running public benchmark data, not just smoke fixtures.

- Retrieval: report Recall@k, MRR@10, nDCG@10, and reranker delta.
- RAG: report EM/F1, citation presence, and source hit rate.
- Agent: report route accuracy, tool choice accuracy, tool execution success, final answer accuracy, and HITL resume success.
