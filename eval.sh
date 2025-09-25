#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_evaluations.sh
#
# If you supply models as arguments, those override the default list.
# Otherwise the script uses the MODELS array defined below.
# ------------------------------------------------------------------------------

# ----------------------- configurable section ---------------------------------
MODELS=(
    # 'LGAI-EXAONE/EXAONE-Deep-2.4B'
    'LGAI-EXAONE/EXAONE-Deep-7.8B'
    'LGAI-EXAONE/EXAONE-Deep-32B'
    'dnotitia/DNA-R1'
    'OLAIR/ko-r1-7b-v2.0.3'
    'OLAIR/ko-r1-14b-v2.0.3'
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    # 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'
    'Qwen/Qwen3-0.6B'
    'Qwen/Qwen3-1.7B'
    'Qwen/Qwen3-4B'
    'Qwen/Qwen3-8B'
    'Qwen/Qwen3-14B'
    'Qwen/Qwen3-32B'
    'Qwen/Qwen3-30B-A3B'
    'Qwen/QwQ-32B'
    'open-thoughts/OpenThinker3-1.5B'
    'open-thoughts/OpenThinker3-7B'
    'open-thoughts/OpenThinker2-7B'
    'open-thoughts/OpenThinker2-32B'
    'open-thoughts/OpenThinker-7B'
    'open-thoughts/OpenThinker-32B'
    )
    
DATASETS=("MCLM" "ClinicalQA" "KMMLU_Redux" "HRB1_0")
TEMP=0.7
TOP_P=0.9
# ------------------------------------------------------------------------------

# If any CLI arguments are present, treat them as the model list.
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
fi

# ----------------------------- main loops -------------------------------------
for MODEL in "${MODELS[@]}"; do
  echo "=== Evaluating model: ${MODEL} ==="
  for DS in "${DATASETS[@]}"; do
    echo "-> Dataset: ${DS}"
    python evaluation.py \
      --model   "${MODEL}" \
      --dataset "${DS}" \
      --temperature "${TEMP}" \
      --top_p "${TOP_P}"
  done
done
