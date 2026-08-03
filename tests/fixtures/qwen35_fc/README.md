# Qwen3.5 function-calling integration fixture

This is a frozen 128-example software integration and regression fixture. It is
not a training subset, benchmark, or semantic gold set.

The rows use the canonical pre-Dolci contract:

```text
messages, tools, dataset, sample_id
```

They span the five historical Phase-2 components with fixed quotas: Dolci 24,
GraphSyn 20, Nemotron Agentic v1 28, Nemotron Agentic v2 interactive-agent 28,
and TxT360-high 28. The strict selector met every preregistered structural and
semantic coverage target, including no-call, single/parallel/sequential/hybrid,
multi-turn, tool-result, tool-and-text, clarification/prerequisite/partial
parameter/near-miss, multiple AMS confidence regimes, nested parameters, and
long-context cases.

Lineage was validated before this fixture was frozen. The internal lineage
report is intentionally excluded from the public code snapshot. `cases.jsonl`
keeps the tags and per-example metadata required by the regression tests
separate from the canonical input rows.

Frozen core hashes:

- `fixture.jsonl`: `a5b33f51a1f8b657ae19ad54c4961a2b24c684581a1000e17c8e639896526430`
- `cases.jsonl`: `936b68803d6fcfe5e8c6c7464f1c4407c408dc61e2162b81dde3bb68eb1988c2`
- verified candidate pool: `7e4e85d8f1ba5c6f5fc5461345eeabf059cf683f85985b26981c7a8151c4ef3d`

Nested source key insertion order is intentionally preserved. It affects the
native model's rendered JSON tool-schema text and therefore its token IDs.

Pinned Qwen3.5-0.8B Base reference validation at 32,768 tokens yields 563,233
tokens, 147,669 raw trainable tokens, and exactly 2 truncated long examples.
The concatenated token and mask digests are
`1320a68a4b30f36b565149c325343b0eeeb6173c217f41a7518b3453b9984ab0` and
`e1b31b385495eda44fc6a4f59e38dea59f15e4fcef69254626baa769ef0b8617`.
