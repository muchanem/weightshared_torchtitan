#!/bin/bash
# Monitor combined sharing runs and auto-launch evals when they finish.
# Converts DCP checkpoint -> HF format, then runs lm_eval.
# Usage: sbatch monitor_combined_evals.sh

#SBATCH --job-name=combined_eval_monitor
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8640
#SBATCH --account=optim
#SBATCH --qos=h100_alignment_shared
#SBATCH --output=/home/sanaelotfi/combined_sharing/colm_logs/monitor_%j.out
#SBATCH --error=/home/sanaelotfi/combined_sharing/colm_logs/monitor_%j.err

set -eo pipefail
source $HOME/miniconda3/etc/profile.d/conda.sh

CKPT_BASE="/checkpoint/optim/sanaelotfi/weight_shared_llms/checkpoints/colm_runs/combined"
EVAL_OUT="/checkpoint/optim/sanaelotfi/weight_shared_llms/eval_results/colm/combined"
CONVERT_SCRIPT="$HOME/combined_sharing/scripts/checkpoint_conversion/convert_to_hf.py"
PROJECT_DIR="$HOME/combined_sharing"
TASKS="arc_easy,hellaswag,piqa,winogrande,rte,openbookqa,triviaqa"
EXPECTED_STEP=76300  # Final step for most configs

ALL_CONFIGS=(
    colm_100m_1_standard colm_100m_2_wide colm_100m_3_deep colm_100m_4_factonly
    colm_100m_5_conservative colm_100m_6_deep_combined colm_100m_7_cycle colm_100m_8_cyclerev
    colm_100m_9_low_emb colm_100m_10_high_rank colm_100m_11_untied colm_100m_12_wide_combined
    colm_300m_1_standard colm_300m_2_wide colm_300m_3_deep colm_300m_4_factonly
    colm_300m_5_conservative colm_300m_6_deep_combined colm_300m_7_cycle colm_300m_8_cyclerev
    colm_300m_9_low_emb colm_300m_10_high_rank colm_300m_11_untied colm_300m_12_wide_combined
    colm_500m_1_standard colm_500m_2_wide colm_500m_3_deep colm_500m_4_factonly
    colm_500m_5_conservative colm_500m_6_deep_combined colm_500m_7_cycle colm_500m_8_cyclerev
    colm_500m_9_low_emb colm_500m_10_high_rank colm_500m_11_untied colm_500m_12_wide_combined
    colm_1b_1_standard colm_1b_2_wide colm_1b_3_deep colm_1b_4_factonly
    colm_1b_5_conservative colm_1b_6_deep_combined colm_1b_7_cycle colm_1b_8_cyclerev
    colm_1b_9_low_emb colm_1b_10_high_rank colm_1b_11_untied colm_1b_12_wide_combined
)

mkdir -p "$EVAL_OUT"

echo "=========================================="
echo "Combined Sharing Eval Monitor"
echo "Watching ${#ALL_CONFIGS[@]} configs"
echo "Start: $(date)"
echo "=========================================="

declare -A EVALED

while true; do
    all_done=true
    for cfg in "${ALL_CONFIGS[@]}"; do
        # Skip already evaluated
        if [ "${EVALED[$cfg]}" == "done" ]; then
            continue
        fi

        # Check if final checkpoint exists
        ckpt_dir="${CKPT_BASE}/${cfg}/checkpoint"
        final_ckpt="${ckpt_dir}/step-${EXPECTED_STEP}"

        # Also check for step-30520 (100m configs have different step count)
        if [[ "$cfg" == *"100m"* ]]; then
            final_ckpt="${ckpt_dir}/step-30520"
            if [ ! -d "$final_ckpt" ]; then
                final_ckpt="${ckpt_dir}/step-${EXPECTED_STEP}"
            fi
        fi

        # Find the latest checkpoint
        latest_ckpt=$(ls -d "${ckpt_dir}/step-"*/ 2>/dev/null | sort -t- -k2 -n | tail -1)

        if [ -z "$latest_ckpt" ]; then
            all_done=false
            continue
        fi

        # Check if this is a final or near-final checkpoint
        step_num=$(basename "$latest_ckpt" | sed 's/step-//')

        # For the eval, check if this config's training job is still running
        running=$(squeue -u sanaelotfi --name=colm_combined_sharing -h 2>/dev/null | wc -l)

        # If checkpoint exists and eval result doesn't, run eval
        if [ -d "$latest_ckpt" ] && [ ! -f "${EVAL_OUT}/${cfg}/results.json" ]; then
            # Check if training job for this config finished
            # Simple heuristic: if step >= expected, it's done
            if [ "$step_num" -ge 15260 ]; then
                echo "$(date): Evaluating $cfg (step $step_num)"

                # Convert DCP -> HF
                hf_dir="${CKPT_BASE}/${cfg}/hf_checkpoint"
                if [ ! -d "$hf_dir" ]; then
                    echo "  Converting checkpoint to HF format..."
                    conda activate torchtitan_env
                    cd "$PROJECT_DIR"
                    python "$CONVERT_SCRIPT" \
                        "$latest_ckpt" \
                        "$hf_dir" \
                        --model_name qwen3 \
                        --export_dtype bfloat16 2>&1 || {
                        echo "  WARNING: HF conversion failed for $cfg, skipping"
                        continue
                    }
                fi

                # Run lm_eval
                if [ -d "$hf_dir" ]; then
                    echo "  Running lm_eval..."
                    conda activate lm_eval_env 2>/dev/null || conda activate lingua_251209
                    mkdir -p "${EVAL_OUT}/${cfg}"

                    python -m lm_eval \
                        --model hf \
                        --model_args "pretrained=${hf_dir},dtype=bfloat16" \
                        --tasks $TASKS \
                        --num_fewshot 5 \
                        --batch_size auto \
                        --output_path "${EVAL_OUT}/${cfg}" 2>&1 || {
                        echo "  WARNING: lm_eval failed for $cfg"
                        continue
                    }

                    EVALED[$cfg]="done"
                    echo "  ✅ Done: $cfg"
                fi
            else
                all_done=false
            fi
        elif [ -f "${EVAL_OUT}/${cfg}/results.json" ]; then
            EVALED[$cfg]="done"
        else
            all_done=false
        fi
    done

    evaled_count=0
    for v in "${EVALED[@]}"; do [ "$v" == "done" ] && ((evaled_count++)); done
    echo "$(date): ${evaled_count}/${#ALL_CONFIGS[@]} evaluated. Sleeping 10min..."

    if [ "$evaled_count" -eq "${#ALL_CONFIGS[@]}" ]; then
        echo "All configs evaluated! Exiting."
        break
    fi

    sleep 600
done

echo "=========================================="
echo "Monitor complete at $(date)"
echo "=========================================="
