from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.data.audit_qwen35_core_arm_decoding import ARM_ORDER, apply_edit, contiguous_true_spans, select_cases


def test_contiguous_true_spans() -> None:
    mask = np.asarray([False, True, True, False, True, False], dtype=np.bool_)
    assert contiguous_true_spans(mask) == [(1, 3), (4, 5)]
    assert contiguous_true_spans(np.zeros(3, dtype=np.bool_)) == []


def test_apply_edit_preserves_order_and_checks_count() -> None:
    messages = [{"role": role, "content": str(index)} for index, role in enumerate(["user", "assistant"] * 3)]
    operation = {"merged_spans": [[1, 3], [4, 5]], "removed_messages": 3}
    assert [message["content"] for message in apply_edit(messages, operation)] == ["0", "3", "5"]


def test_selection_covers_sources_strata_and_each_operation_action(tmp_path: Path) -> None:
    sources = ["dolci", "graphsyn", "nemotron_agentic_v1", "nemotron_agentic_v2_ia", "txt360_high"]
    rows = []
    for index, source in enumerate(sources):
        rows.append(
            {
                "global_row_number": index,
                "source_key": source,
                "has_high_confidence_ams": index == 0,
                "has_low_confidence_ams": index == 0,
                "is_c11_pure_candidate": index == 0,
                "num_tool_calls": 0 if index == 0 else 1,
                "num_no_call_traces": 1 if index == 0 else 0,
                "num_single_call_traces": 1 if index == 0 else 0,
                "num_sequential_traces": 1 if index == 0 else 0,
                "num_parallel_traces": 1 if index == 0 else 0,
                "num_hybrid_traces": 1 if index == 0 else 0,
                "is_multi_turn": index == 0,
                "qwen_truncated": index == 0,
                "qwen_total_tokens_untruncated": 100 + index,
                "qwen_assistant_loss_tokens": 1,
            }
        )
    features = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pylist(rows), features)
    operations = {arm_id: {} for arm_id in ARM_ORDER}
    for arm_id in ARM_ORDER[1:]:
        operations[arm_id][0] = {"arm_id": arm_id, "global_row_number": 0, "action": "drop_sample"}
    operations["C10"][1] = {"arm_id": "C10", "global_row_number": 1, "action": "drop_real_turn_spans"}

    cases, selections = select_cases(features, operations, amendment_rows=[0, 1, 2, 3, 4])

    assert len(cases) == 5
    assert {f"source:{source}" for source in sources} <= selections.keys()
    assert selections["operation:C10:drop_sample"] == 0
    assert selections["operation:C10:drop_real_turn_spans"] == 1
    assert all("C00" in case["inspected_arms"] for case in cases.values())
