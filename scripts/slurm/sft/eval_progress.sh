for exp in exp_a_instruct_sft exp_b_dolci_fc exp_c_nemotron_fc exp_d_it_only exp_e_mixed_fc exp_f_instruct_plus_nemotron; do
echo "=== $exp ==="
for run in 1 2 3; do
    dir="/leonardo_work/OELLM_prod2026/ytahtah0/eval-results/$exp/run_$run"
    total=$(ls "$dir"/*predictions* 2>/dev/null | wc -l)
    # Check key benchmarks by name
    has_mmlu=$(ls "$dir"/*mmlu*predictions* 2>/dev/null | wc -l)
    has_bbh=$(ls "$dir"/*bbh*predictions* 2>/dev/null | wc -l)
    has_math=$(ls "$dir"/*minerva*predictions* 2>/dev/null | wc -l)
    has_humaneval=$(ls "$dir"/*humanevalplus*predictions* 2>/dev/null | wc -l)
    has_mbpp=$(ls "$dir"/*mbppplus*predictions* 2>/dev/null | wc -l)
    has_gpqa=$(ls "$dir"/*gpqa*predictions* 2>/dev/null | wc -l)
    has_agi=$(ls "$dir"/*agi_eval*predictions* 2>/dev/null | wc -l)
    echo "  run_$run: $total total | mmlu=$has_mmlu bbh=$has_bbh math=$has_math gpqa=$has_gpqa agi=$has_agi humaneval=$has_humaneval mbpp=$has_mbpp"
done
done
