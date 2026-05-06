#!/bin/bash
set -euo pipefail

cd /home/airs/homework/llm_fp
source /home/airs/miniforge3/etc/profile.d/conda.sh
conda activate validation

SESSION_NAME=${1:-qwen35-lora}
LOG_PATH="temp/${SESSION_NAME}.log"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" \
  "cd /home/airs/homework/llm_fp && source /home/airs/miniforge3/etc/profile.d/conda.sh && conda activate validation && CUDA_VISIBLE_DEVICES=1 python scripts/train_qwen35_averitec_lora.py"
tmux pipe-pane -o -t "${SESSION_NAME}" "cat >> ${LOG_PATH}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_PATH}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Tail log: tail -f ${LOG_PATH}"
