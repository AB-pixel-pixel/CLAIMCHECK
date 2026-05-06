#!/bin/bash
set -euo pipefail

cd /home/airs/homework/llm_fp

set -a
if [ -f .env ]; then
  source .env
fi
set +a

SESSION_NAME=${1:-claimcheck-exp}
NUM_RECORDS=${2:-500}
LOG_PATH="temp/${SESSION_NAME}.log"

/home/airs/miniforge3/envs/validation/bin/python - <<'PY'
from ClaimCheck.factchecker.tools.quota import reset_state
state = reset_state()
print(f"Reset Serper counter to {state['count']} with pause threshold {state['soft_limit']}.")
PY

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "cd /home/airs/homework/llm_fp && ./run.sh ${NUM_RECORDS}"
tmux pipe-pane -o -t "${SESSION_NAME}" "cat >> ${LOG_PATH}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_PATH}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Tail log: tail -f ${LOG_PATH}"
