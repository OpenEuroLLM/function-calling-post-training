#!/usr/bin/env bash
# Inspect OLMES eval result files to verify collect_eval_results.py assumptions.
# Run on Leonardo: bash inspect_eval_results.sh

WORK_DIR="${WORK}/ytahtah0"
BASE="${WORK_DIR}/eval-results"
SAMPLE_DIR="${BASE}/exp_a_instruct_sft_v2/run_1"

echo "========================================"
echo "1. All metrics filenames in exp_a/run_1"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*-metrics.json 2>/dev/null | xargs -I{} basename {} | sort
echo ""

echo "========================================"
echo "2. Score scale check — popqa metrics JSON"
echo "========================================"
cat "${SAMPLE_DIR}"/*popqa*-metrics.json 2>/dev/null | python3 -m json.tool | head -40
echo ""

echo "========================================"
echo "3. Score scale check — one MMLU subject"
echo "========================================"
first_mmlu=$(ls -1 "${SAMPLE_DIR}"/*mmlu*-metrics.json 2>/dev/null | head -1)
if [ -n "$first_mmlu" ]; then
    echo "File: $(basename "$first_mmlu")"
    cat "$first_mmlu" | python3 -m json.tool | head -40
else
    echo "No MMLU metrics files found"
fi
echo ""

echo "========================================"
echo "4. GPQA file naming"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*gpqa*-metrics.json 2>/dev/null | xargs -I{} basename {} || echo "No GPQA files found"
echo ""

echo "========================================"
echo "5. BBH file naming (first 3)"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*bbh*-metrics.json 2>/dev/null | head -3 | xargs -I{} basename {} || echo "No BBH files found"
echo ""

echo "========================================"
echo "6. AGI Eval file naming (first 3)"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*agi_eval*-metrics.json 2>/dev/null | head -3 | xargs -I{} basename {} || echo "No AGI Eval files found"
echo ""

echo "========================================"
echo "7. Minerva Math file naming (first 3)"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*minerva*-metrics.json 2>/dev/null | head -3 | xargs -I{} basename {} || echo "No Minerva Math files found"
echo ""

echo "========================================"
echo "8. IFEval + HumanEval+ + MBPP+ naming"
echo "========================================"
ls -1 "${SAMPLE_DIR}"/*ifeval*-metrics.json 2>/dev/null | xargs -I{} basename {} || echo "No IFEval files"
ls -1 "${SAMPLE_DIR}"/*humanevalplus*-metrics.json 2>/dev/null | xargs -I{} basename {} || echo "No HumanEval+ files"
ls -1 "${SAMPLE_DIR}"/*mbppplus*-metrics.json 2>/dev/null | xargs -I{} basename {} || echo "No MBPP+ files"
echo ""

echo "========================================"
echo "9. Completion counts per experiment run_1"
echo "========================================"
for exp in exp_a_instruct_sft_v2 exp_b_dolci_fc_v2 exp_c_nemotron_fc_v2 exp_d_it_only_v2 exp_e_mixed_fc_v2 exp_f_instruct_plus_nemotron_v2; do
    count=$(ls "${BASE}/${exp}/run_1"/*-metrics.json 2>/dev/null | wc -l)
    pred=$(ls "${BASE}/${exp}/run_1"/*-predictions* 2>/dev/null | wc -l)
    echo "  ${exp}  metrics: ${count}  predictions: ${pred}"
done
echo ""

echo "========================================"
echo "10. Completion counts per experiment run_2 and run_3"
echo "========================================"
for run in 2 3; do
    echo "--- run_${run} ---"
    for exp in exp_a_instruct_sft_v2 exp_b_dolci_fc_v2 exp_c_nemotron_fc_v2 exp_d_it_only_v2 exp_e_mixed_fc_v2 exp_f_instruct_plus_nemotron_v2; do
        count=$(ls "${BASE}/${exp}/run_${run}"/*-metrics.json 2>/dev/null | wc -l)
        echo "  ${exp}  metrics: ${count}"
    done
done
