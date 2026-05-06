#!/bin/bash
set -a
if [ -f .env ]; then
  source .env
fi
set +a

# Set LD_LIBRARY_PATH to include the conda environment's lib directory to avoid C++ ABI issues
export LD_LIBRARY_PATH=/home/airs/miniforge3/envs/validation/lib:$LD_LIBRARY_PATH
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

export CLAIMCHECK_PAPER_ALIGN=${CLAIMCHECK_PAPER_ALIGN:-0}
export PAPER_LLM_MODEL=${PAPER_LLM_MODEL:-Qwen/Qwen3-4B}

# Run the fact checker
# Usage: ./run.sh [num_records]
NUM_RECORDS=${1:-5}
/home/airs/miniforge3/envs/validation/bin/python ClaimCheck/run_dev.py data/dev.json $NUM_RECORDS
