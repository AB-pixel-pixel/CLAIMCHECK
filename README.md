# ClaimCheck Reproduction on AVeriTeC

This repository reproduces the paper **"ClaimCheck: Automatic Fact-Checking of Textual Claims using Web Evidence"** in a local workspace and documents where the local pipeline matches or diverges from the paper.

The goal is not only to run the system, but to answer three practical questions:

1. Can the method be reproduced locally?
2. How effective is the local reproduction?
3. How do local results differ from the original paper?

## Current Status

- Paper-reported verdict prediction accuracy: `62.6%` on the `AVeriTeC` dev set (`500` claims)
- Best local full-run result currently recorded in this workspace: `42.8%` (`214/500`)
- Small pilot run with the LoRA verdict model: `10.0%` (`1/10`)
- Local setup differs from the paper in both **implementation details** and **verdict model choice**, so this repository should be treated as a reproduction study rather than an exact official re-release

## Repository Layout

```text
.
├── 2025.knowledgenlp-1.26.pdf            # original paper PDF
├── paper.txt                            # extracted paper text
├── paper_results.md                     # paper metrics summarized from the paper
├── ClaimCheck/                          # local fact-checking scaffold
├── data/
│   ├── dev.json                         # local AVeriTeC-style dev sample
│   └── averitec/                        # local training data directory
├── outputs/
│   └── qwen35-averitec-lora/            # local LoRA adapter output
├── report/
│   ├── report.tex                       # reproduction report source
│   └── *.md                             # experiment summaries
├── scripts/
│   ├── prepare_averitec_sft.py          # convert training data to SFT format
│   ├── train_qwen35_averitec_lora.py    # LoRA fine-tuning script
│   └── test_vllm_qwen.py
├── run.sh                               # main pipeline entrypoint
├── start_experiment_tmux.sh             # long experiment launcher
└── start_finetune_tmux.sh               # fine-tuning launcher
```

## Environment

All Python commands should use the `validation` conda environment.

- Preferred interpreter:
  `/home/airs/miniforge3/envs/validation/bin/python`
- Required runtime library setting before running the pipeline:
  `LD_LIBRARY_PATH=/home/airs/miniforge3/envs/validation/lib:$LD_LIBRARY_PATH`

The provided `run.sh` already exports the required `LD_LIBRARY_PATH`.

## External Dependencies

This reproduction is not fully self-contained. The pipeline depends on:

- a web search API key via `BOCHA_API_KEY`
- Hugging Face model access for the local LLM
- enough local GPU memory for Qwen-based inference and LoRA fine-tuning

Copy `.env.example` to `.env` and fill in your local settings:

```bash
cp .env.example .env
```

Important environment variables:

- `BOCHA_API_KEY`
- `LOCAL_LLM_MODEL`
- `LOCAL_CUDA_DEVICE`
- `SERPER_TOTAL_LIMIT`
- `SERPER_SOFT_LIMIT`

## Quick Start

Run a short local fact-checking pass on the dev sample:

```bash
./run.sh 5
```

This runs:

```bash
/home/airs/miniforge3/envs/validation/bin/python ClaimCheck/run_dev.py data/dev.json 5
```

For a longer experiment in `tmux`:

```bash
./start_experiment_tmux.sh claimcheck-exp 500
```

Useful follow-up commands:

```bash
tmux attach -t claimcheck-exp
tail -f temp/claimcheck-exp.log
```

Generated fact-check reports are written under `ClaimCheck/reports/` with timestamped or resumed run directories.

## Fine-Tuning the Verdict Model

This workspace includes a local LoRA fine-tuning path for a verdict model based on `Qwen/Qwen3.5-4B`.

### 1. Prepare SFT training data

```bash
/home/airs/miniforge3/envs/validation/bin/python scripts/prepare_averitec_sft.py
```

This converts `data/averitec/train.json` into `data/averitec/train_sft.jsonl`.

### 2. Train the LoRA adapter

```bash
/home/airs/miniforge3/envs/validation/bin/python scripts/train_qwen35_averitec_lora.py
```

Or launch it in `tmux`:

```bash
./start_finetune_tmux.sh qwen35-lora
```

Current script characteristics:

- base model: `Qwen/Qwen3.5-4B`
- 4-bit loading with `bitsandbytes`
- LoRA rank `16`, alpha `32`
- one training epoch
- output dir: `outputs/qwen35-averitec-lora/`

## Model Artifacts

This public repository does **not** include LoRA weights or training checkpoints.

Ignored artifact categories include:

- `outputs/` model outputs and checkpoints
- runtime caches and generated reports
- local secrets and environment-specific files
- third-party paper source bundles and local reference PDFs

If you want to release adapter weights later, publish them separately with clear license notes and avoid mixing them into the main reproduction history by accident.

## Results Snapshot

### Paper-reported results

From [paper_results.md](/home/airs/homework/llm_fp/paper_results.md):

- ClaimCheck: `0.626`
- ClaimCheck without claim-matching: `0.598`
- Naive Qwen2.5-7B: `0.260`
- Fine-tuned Qwen2.5-7B verdict model: `0.626`

### Local results recorded in this workspace

From [report/claimcheck_full_500_summary.md](/home/airs/homework/llm_fp/report/claimcheck_full_500_summary.md):

- records processed: `500`
- completed: `498`
- failed: `2`
- accuracy: `0.428` (`214/500`)
- valid-only accuracy: `0.430`

From [report/experiment-20260505.md](/home/airs/homework/llm_fp/report/experiment-20260505.md):

- LoRA pilot evaluation: `10.0%` (`1/10`)
- observed issue: the model often emitted reasoning text instead of a clean verdict label

## Why the Reproduction Differs from the Paper

The current local setup is meaningfully different from the paper in several ways:

- the paper reports a **fine-tuned Qwen2.5-7B** verdict model, while this workspace currently uses a **Qwen3.5-4B LoRA** verdict model
- the local scaffold includes practical modifications to support local inference, adapter loading, and API-backed search
- web search behavior depends on live external services, which makes the pipeline sensitive to quota, availability, ranking changes, and time
- not every internal detail of the original paper implementation is available in this repository

Because of those differences, the local result should be interpreted as a **reproduction attempt with documented deviations**, not as a strict official benchmark rerun.

## Known Issues

- The claim-matching path appears unused in the recorded `500`-claim local run.
- The LoRA verdict model can output reasoning traces or malformed verdicts instead of one of the expected labels.
- Reproduction quality depends on external search and model availability.
- Long experiments can consume substantial query quota.

## Report

The reproduction write-up is maintained in:

- [report/report.tex](/home/airs/homework/llm_fp/report/report.tex)

This TeX report is intended to summarize:

- abstract
- introduction
- proposed idea / exploration
- technical details
- results
- analysis and discussion
- references

## Citation

If you use this repository, cite the original ClaimCheck paper and clearly distinguish the original reported numbers from the local reproduction numbers contained here.
