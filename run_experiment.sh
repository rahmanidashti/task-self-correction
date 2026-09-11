#!/bin/bash
# Runs every configuration from the paper: 6 models x 2 datasets x 3 methods x 2 tasks.
# Each configuration goes through: main.py (self-correction iterations) -> answer_process.py (final answer
# extraction) -> eval.py (judging) -> result.py (per-iteration and cumulative accuracy).
set -e

MODELS=("SmolLM2-1.7B" "Qwen2.5-3B" "Llama-3.1-8B" "Qwen2.5-14B" "DeepSeek-R1-Distill-Llama-8B" "gemini/gemini-2.0-flash")
DATASETS=("bbeh-disambiguation-qa" "tiny-truthful-qa")

BASE_PARAMS="param.temp=1.0 param.top_k=50 param.top_p=0.95"
SC_PARAMS="param.temp=0.7 param.top_k=20 param.top_p=1.0"

run_pipeline() {
    for script in main.py answer_process.py eval.py result.py; do
        python "$script" "$@"
    done
}

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        common="select.model=$model select.dataset=$dataset"

        # Method: baseline
        run_pipeline $BASE_PARAMS $common select.method=baseline select.task=generation select.start_prompt=base-gen-2 select.iterate_prompt=base-gen-2
        run_pipeline $BASE_PARAMS $common select.method=baseline select.task=multiple-choice select.start_prompt=base-mc1-2 select.iterate_prompt=base-mc1-2

        # Method: CoT
        run_pipeline $BASE_PARAMS $common select.method=CoT select.task=generation select.start_prompt=cot-gen-2 select.iterate_prompt=cot-gen-2
        run_pipeline $BASE_PARAMS $common select.method=CoT select.task=multiple-choice select.start_prompt=cot-mc1-2 select.iterate_prompt=cot-mc1-2

        # Method: self-consistency
        run_pipeline $SC_PARAMS $common select.method=self-consistency select.task=generation select.start_prompt=cot-gen-2 select.iterate_prompt=cot-gen-2
        run_pipeline $SC_PARAMS $common select.method=self-consistency select.task=multiple-choice select.start_prompt=cot-mc1-2 select.iterate_prompt=cot-mc1-2
    done
done
