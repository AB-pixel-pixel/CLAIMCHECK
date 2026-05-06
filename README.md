# ClaimCheck Reproduction on AVeriTeC

This repository reproduces the paper **"ClaimCheck: Automatic Fact-Checking of Textual Claims using Web Evidence"** and documents where the local pipeline matches or diverges from the reported results.

The project focuses on three questions:

1. Can the method be reproduced locally?
2. How effective is the local reproduction?
3. How do local results differ from the original paper?

## Current Status

- Paper-reported verdict prediction accuracy: `62.6%` on the `AVeriTeC` dev set (`500` claims)
- Best local full-run result currently recorded in this workspace: `42.8%` (`214/500`)
- Small pilot run with the local LoRA verdict model: `10.0%` (`1/10`)
- This repository is a reproduction study, not an exact upstream re-release

## Repository Layout

```text
.
├── ClaimCheck/                        # local fact-checking scaffold
├── data/
│   └── dev.json                       # local AVeriTeC-style sample
├── report/
│   ├── report.tex                     # main reproduction report
│   └── experiment-20260505.md         # experiment note
├── scripts/
│   ├── prepare_averitec_sft.py        # build SFT training data
│   ├── train_qwen35_averitec_lora.py  # LoRA fine-tuning
│   └── test_vllm_qwen.py              # optional vLLM smoke test
├── .env.example
├── requirements.txt
├── run.sh
├── start_experiment_tmux.sh
└── start_finetune_tmux.sh
```

## Environment

The original local setup used the `validation` conda environment.

- Preferred interpreter:
  `/home/airs/miniforge3/envs/validation/bin/python`
- Required runtime library path before running the pipeline:
  `LD_LIBRARY_PATH=/home/airs/miniforge3/envs/validation/lib:$LD_LIBRARY_PATH`

`run.sh` already exports the required `LD_LIBRARY_PATH`.

## Installation

Create and activate an environment, then install dependencies:

```bash
conda create -n validation python=3.10 -y
conda activate validation
pip install -r requirements.txt
```

If you need a CUDA build of PyTorch, install the matching `torch` wheel for your machine before or after `pip install -r requirements.txt`.

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Recommended `.env` template:

```bash
BOCHA_API_KEY=your-bocha-api-key

LOCAL_LLM_MODEL=Qwen/Qwen3.5-4B
LOCAL_CUDA_DEVICE=0

SERPER_TOTAL_LIMIT=2500
SERPER_SOFT_LIMIT=2450

CLAIMCHECK_PAPER_ALIGN=0
PAPER_LLM_MODEL=Qwen/Qwen3-4B

# Optional for LoRA inference
# LOCAL_LLM_ADAPTER=/abs/path/to/outputs/qwen35-averitec-lora

# Optional generation controls
# LOCAL_MAX_NEW_TOKENS=512
# CLAIMCHECK_FORCE_THINK=0
```

Variable meanings:

- `BOCHA_API_KEY`: required web search API key
- `LOCAL_LLM_MODEL`: local Hugging Face model used by the pipeline
- `LOCAL_CUDA_DEVICE`: CUDA device index used for local inference
- `SERPER_TOTAL_LIMIT`: hard query budget used by the quota tracker
- `SERPER_SOFT_LIMIT`: early-stop budget to avoid overrunning quota
- `CLAIMCHECK_PAPER_ALIGN`: when set to `1`, prefer the paper-aligned local model path in code
- `PAPER_LLM_MODEL`: fallback model name used for paper-aligned mode
- `LOCAL_LLM_ADAPTER`: optional LoRA adapter path for verdict-model experiments
- `LOCAL_MAX_NEW_TOKENS`: optional generation length override
- `CLAIMCHECK_FORCE_THINK`: force enable or disable thinking mode if the tokenizer supports it

## Main Experiment Commands

Run a short local fact-checking pass:

```bash
./run.sh 5
```

This executes:

```bash
/home/airs/miniforge3/envs/validation/bin/python ClaimCheck/run_dev.py data/dev.json 5
```

Run a longer experiment in `tmux`:

```bash
./start_experiment_tmux.sh claimcheck-exp 500
```

Useful follow-up commands:

```bash
tmux attach -t claimcheck-exp
tail -f temp/claimcheck-exp.log
```

The helper script resets the search quota tracker, loads `.env`, and launches `./run.sh`.

## Fine-Tuning Commands

Prepare SFT training data:

```bash
python scripts/prepare_averitec_sft.py
```

Train the LoRA adapter:

```bash
python scripts/train_qwen35_averitec_lora.py
```

Launch fine-tuning in `tmux`:

```bash
./start_finetune_tmux.sh qwen35-lora
```

Current training script defaults:

- base model: `Qwen/Qwen3.5-4B`
- quantization: 4-bit `bitsandbytes`
- LoRA rank: `16`
- LoRA alpha: `32`
- training epochs: `1`
- output directory: `outputs/qwen35-averitec-lora/`

Useful fine-tuning overrides:

```bash
FT_MODEL_NAME=Qwen/Qwen3.5-4B \
FT_DATA_PATH=/abs/path/to/train_sft.jsonl \
FT_OUTPUT_DIR=/abs/path/to/output_dir \
FT_MAX_LENGTH=1024 \
python scripts/train_qwen35_averitec_lora.py
```

## Optional Checks

Run the optional vLLM smoke test:

```bash
python scripts/test_vllm_qwen.py
```

## Results Snapshot

Paper-reported results from `paper_results.md`:

- ClaimCheck: `0.626`
- ClaimCheck without claim-matching: `0.598`
- Naive Qwen2.5-7B: `0.260`
- Fine-tuned Qwen2.5-7B verdict model: `0.626`

Local results recorded in this repository:

- full local run: `0.428` (`214/500`)
- valid-only accuracy: `0.430`
- LoRA pilot evaluation: `0.100` (`1/10`)

## Known Gaps

- The current local verdict model differs from the paper's reported fine-tuned verdict model.
- Web search depends on live external APIs and quota.
- The LoRA verdict model can emit malformed outputs or reasoning traces instead of clean verdict labels.
- This public repository does not include model weights, training checkpoints, or local secrets.

## Report

The write-up is maintained in [report/report.tex](/home/airs/homework/llm_fp/report/report.tex).

## Citation

If you use this repository, cite the original ClaimCheck paper and clearly separate original reported numbers from local reproduction numbers.
