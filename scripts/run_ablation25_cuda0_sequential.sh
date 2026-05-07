#!/bin/bash
set -euo pipefail

cd /home/airs/homework/llm_fp

set -a
source .env
set +a

export LD_LIBRARY_PATH=/home/airs/miniforge3/envs/validation/lib:${LD_LIBRARY_PATH:-}
export CLAIMCHECK_PAPER_ALIGN=0
export LOCAL_LLM_MODEL=Qwen/Qwen3.5-4B
export LOCAL_CUDA_DEVICE="${LOCAL_CUDA_DEVICE:-0}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SERPER_SOFT_LIMIT=10000
export SERPER_TOTAL_LIMIT=10000

PYTHON=/home/airs/miniforge3/envs/validation/bin/python
NUM_RECORDS="${NUM_RECORDS:-25}"
TAG_PREFIX="${TAG_PREFIX:-ablation25}"
CONFIGS="${CONFIGS:-ft-retrieval-direct,noft-retrieval-direct,ft-no-retrieval-direct,ft-retrieval-thinking}"

run_one() {
  local tag="$1"
  local adapter_mode="$2"
  local retrieval_mode="$3"
  local think_mode="$4"

  export CLAIMCHECK_RUN_TAG="$tag"
  export SERPER_QUOTA_STATE="/home/airs/homework/llm_fp/temp/serper_quota_${tag}.json"
  export CLAIMCHECK_FORCE_THINK="$think_mode"

  if [ "$adapter_mode" = "ft" ]; then
    export LOCAL_LLM_ADAPTER=/home/airs/homework/llm_fp/outputs/qwen35-averitec-lora
  else
    unset LOCAL_LLM_ADAPTER
  fi

  if [ "$retrieval_mode" = "no-retrieval" ]; then
    export CLAIMCHECK_DISABLE_RETRIEVAL=1
  else
    unset CLAIMCHECK_DISABLE_RETRIEVAL
  fi

  "$PYTHON" - <<'PY'
import os
from ClaimCheck.factchecker.tools.quota import reset_state
state = reset_state()
print("ABLATION_ENV", {
    "CLAIMCHECK_RUN_TAG": os.getenv("CLAIMCHECK_RUN_TAG"),
    "LOCAL_LLM_MODEL": os.getenv("LOCAL_LLM_MODEL"),
    "LOCAL_LLM_ADAPTER": os.getenv("LOCAL_LLM_ADAPTER"),
    "LOCAL_CUDA_DEVICE": os.getenv("LOCAL_CUDA_DEVICE"),
    "CLAIMCHECK_DISABLE_RETRIEVAL": os.getenv("CLAIMCHECK_DISABLE_RETRIEVAL"),
    "CLAIMCHECK_FORCE_THINK": os.getenv("CLAIMCHECK_FORCE_THINK"),
    "SERPER_QUOTA_STATE": os.getenv("SERPER_QUOTA_STATE"),
    "quota": f"{state['count']}/{state['soft_limit']}",
})
PY

  "$PYTHON" ClaimCheck/run_dev.py data/dev.json "$NUM_RECORDS" 2>&1 | tee "/home/airs/homework/llm_fp/temp/${tag}.log"
}

IFS=',' read -r -a CONFIG_LIST <<< "$CONFIGS"
for config in "${CONFIG_LIST[@]}"; do
  case "$config" in
    ft-retrieval-direct)
      run_one "${TAG_PREFIX}-ft-retrieval-direct" "ft" "retrieval" "0"
      ;;
    noft-retrieval-direct)
      run_one "${TAG_PREFIX}-noft-retrieval-direct" "noft" "retrieval" "0"
      ;;
    ft-no-retrieval-direct)
      run_one "${TAG_PREFIX}-ft-no-retrieval-direct" "ft" "no-retrieval" "0"
      ;;
    ft-retrieval-thinking)
      run_one "${TAG_PREFIX}-ft-retrieval-thinking" "ft" "retrieval" "1"
      ;;
    "")
      ;;
    *)
      echo "Unknown config: $config" >&2
      exit 1
      ;;
  esac
done
