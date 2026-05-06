# ClaimCheck Paper Results

Source: [paper.txt](/home/airs/homework/llm_fp/paper.txt)

## Table 1

- ClaimCheck: `0.626`
- ClaimCheck without claim-matching: `0.598`
- Papelo: `0.415`
- HerO: `0.752`
- InFact: `0.724`
- Naive GPT-4o: `0.532`
- Naive GPT-4o-Mini: `0.468`
- Naive Qwen2.5-7B: `0.260`

## Table 2

- Fine-tuned Qwen2.5-7B: `0.626`
- Phi-4: `0.494`
- GPT-4o: `0.396`
- GPT-4o-mini: `0.314`
- Qwen2.5-7B: `0.280`

## Table 3

- Claims with evidence: `0.980`
- Claims with evidence after evidence curation: `0.696`
- Questions answered: `0.949`
- Fact-check articles matched: `0.158`
- Claim-matching only accuracy: `0.759`

## Setup Reported In Paper

- Dataset: AVeriTeC dev set, `500` claims
- Metric: claim alignment accuracy
- Main reported model setup:
  `Qwen2.5-7B` for most tasks and fine-tuned `Qwen2.5-7B` for verdict prediction
