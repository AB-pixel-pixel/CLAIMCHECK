# ClaimCheck Submodule

This directory contains the local fact-checking scaffold used by the reproduction project in the repository root.

For the full project overview, environment setup, dependency installation, experiment commands, and reproduction notes, read the root-level [README.md](/home/airs/homework/llm_fp/README.md).

## What Is In This Directory

- `run_dev.py`: main entrypoint used by the local pipeline
- `factchecker/`: fact-checking modules, report writer, and search/scraping tools
- `demo/`: lightweight demo server assets
- `requirement.txt`: original submodule dependency list retained for reference

## How This Submodule Is Used

From the repository root, the main short-run command is:

```bash
./run.sh 5
```

This eventually calls:

```bash
/home/airs/miniforge3/envs/validation/bin/python ClaimCheck/run_dev.py data/dev.json 5
```

If you want to run the submodule entrypoint directly:

```bash
python ClaimCheck/run_dev.py data/dev.json 5
```

## Environment Notes

This reproduction uses:

- a local Hugging Face model for generation
- `BOCHA_API_KEY` for web search
- `.env` values loaded from the repository root
- `LD_LIBRARY_PATH` configured for the `validation` conda environment

In practice, you should use the root-level helper scripts instead of manually editing source files for API keys or model names.

## Important Differences From The Original Upstream ClaimCheck

- This is a locally adapted reproduction scaffold, not a clean mirror of the original upstream repository.
- Search configuration is environment-variable based.
- Local model loading and optional LoRA adapter support are handled in `factchecker/modules/llm.py`.
- Public repository history intentionally excludes model weights, local secrets, and generated reports.
