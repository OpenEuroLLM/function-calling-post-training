# Data attribution

The 128-row software integration fixture under
`tests/fixtures/qwen35_fc/` contains modified examples selected from the
following public datasets:

| Fixture source | Upstream dataset | Upstream license |
|---|---|---|
| Dolci | [allenai/Dolci-Instruct-SFT-Tool-Use](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) | ODC Attribution License |
| GraphSyn | [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind), `graph_syn_datasets/graphsyn.jsonl` | Apache License 2.0 |
| Nemotron Agentic v1 | [nvidia/Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | Creative Commons Attribution 4.0 |
| Nemotron Agentic v2 | [nvidia/Nemotron-SFT-Agentic-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | Creative Commons Attribution 4.0 |
| TxT360 high | [LLM360/TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) | Creative Commons Attribution 4.0 |

The fixture is a modified, selected, and format-converted software test
artifact. Credential-shaped example strings were replaced with explicit
non-secret placeholders for this public snapshot. It is not a training
dataset, benchmark, or semantic gold set.

The upstream dataset cards and licenses remain authoritative. Preserve this
notice when redistributing the fixture.
