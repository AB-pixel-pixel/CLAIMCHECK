# ClaimCheck Reproduction Experiment Report
**Date**: 2026-05-05
**Experiment ID**: exp-20260505-finetuned-lora

## Experiment Configuration

### Model Configuration
- **Base Model**: Qwen/Qwen3.5-4B
- **Fine-tuned Adapter**: `/home/airs/homework/llm_fp/outputs/qwen35-averitec-lora/`
- **Checkpoint**: checkpoint-192 (latest, 192 training steps)
- **LoRA Config**: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]

### Environment
- **Conda Environment**: validation
- **Python**: `/home/airs/miniforge3/envs/validation/bin/python`
- **Dataset**: `data/dev.json` (AVeriTeC-style development sample)
- **Web Search API**: Bocha (configured in .env)
- **Local LLM**: Qwen/Qwen3.5-4B with LoRA adapter

## Final Experimental Results (10 records)

### Overall Metrics
| Metric | Value |
|--------|-------|
| **Total Records** | 10 |
| **Completed** | 10 |
| **Correct** | 1 |
| **Accuracy** | **10.0%** |

### Detailed Results

| # | Claim (truncated) | Expected | Predicted | Correct |
|---|-------------------|----------|-----------|---------|
| 0 | Sean Connery letter to Steve Jobs | Refuted | Supported | ❌ |
| 1 | Trump Administration + Billie Eilish | Refuted | Thinking Process... | ❌ |
| 2 | Imran Khan + French visa cancellation | Refuted | Thinking Process... | ❌ |
| 3 | UNESCO Nadar community ancient race | Refuted | Not Enough Evidence | ❌ |
| 4 | Matt Gaetz hospice fraud $75M | Refuted | Thinking Process... | ❌ |
| 5 | Western media fabricated info | Refuted | Thinking Process... | ❌ |
| 6 | COVID-19 225,000 dead | Supported | Thinking Process... | ❌ |
| 7 | Donald Trump taxes $750 | Supported | Supported | ✅ |
| 8 | Mail-in ballot cheaters | Refuted | Not Enough Evidence | ❌ |
| 9 | 33.1 GDP history | Not Enough Evidence | Supported | ❌ |

## Key Observations

### 1. Model Output Issue
- **Problem**: Model frequently outputs "Thinking Process..." content instead of clean verdict labels
- **Only 1 out of 10** predictions was a clean verdict label ("Supported")
- The model appears to output internal reasoning traces rather than following the instruction to return only the verdict label

### 2. Verdict Distribution
- **Clean verdicts**: 3 (Record 3, 7, 8 - "Not Enough Evidence", "Supported", "Not Enough Evidence")
- **Thinking Process outputs**: 7 (incorrectly formatted responses)

### 3. CUDA Status
- The LoRA adapter was successfully loaded during the experiment
- Log shows: `Loading LoRA adapter from: /home/airs/homework/llm_fp/outputs/qwen35-averitec-lora` followed by `LoRA adapter loaded successfully.`

### 4. Comparison with Paper
- **Paper reported**: 62.6% AVeriTeC verdict prediction accuracy
- **Our result**: 10.0% (1/10 correct)
- **Gap**: 52.6 percentage points

## Issues Identified

1. **Model输出格式问题**: The fine-tuned model does not consistently output clean verdict labels
2. **Verdict Parsing Logic**: The pipeline's verdict extraction may need improvement to handle model outputs that contain thinking process text
3. **Training Data Mismatch**: The model may have been fine-tuned on different data/format than what the pipeline expects

## Code Modifications Made

### 1. `ClaimCheck/factchecker/modules/llm.py`
- Added `LOCAL_LLM_ADAPTER` environment variable support
- Added `PeftModel` import for LoRA adapter loading
- Modified `get_model_and_tokenizer()` to load LoRA adapter when path is provided
- Changed default model from `Qwen/Qwen2.5-7B-Instruct` to `Qwen/Qwen3.5-4B`

### 2. `.env`
- Added `LOCAL_LLM_ADAPTER=/home/airs/homework/llm_fp/outputs/qwen35-averitec-lora`

### 3. `run.sh`
- Added `LOCAL_LLM_ADAPTER` environment variable export

## Files Modified
- `/home/airs/homework/llm_fp/AGENTS.md` - Updated with model location and new files
- `/home/airs/homework/llm_fp/.env` - Added LOCAL_LLM_ADAPTER
- `/home/airs/homework/llm_fp/run.sh` - Added LOCAL_LLM_ADAPTER export
- `/home/airs/homework/llm_fp/ClaimCheck/factchecker/modules/llm.py` - Added LoRA support

## Next Steps
1. Investigate why the fine-tuned model outputs thinking process text instead of clean verdicts
2. Consider re-training with better prompt formatting or instruction tuning
3. Run larger experiment (500 records) once model output issue is resolved
4. Compare with paper's reported 62.6% accuracy