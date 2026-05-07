from modules.llm import prompt_local


judge_prompt = """
Instructions
Determine the Claim's veracity from the fact-check record.
Return ONLY valid JSON with exactly these keys:
- "verdict"
- "justification"

Requirements:
- "verdict" must be exactly one of the Decision Options.
- "justification" must be a short evidence-grounded explanation.
- Do not output markdown, code fences, or any text outside the JSON object.
- Before choosing "Supported", verify that the claim's material details are actually covered by the evidence.
- If the evidence supports only a weaker or partial version of the claim, do NOT choose "Supported".
- If a key number, causal link, location, timeframe, or quoted wording is missing, contradicted, or only weakly implied, prefer "Refuted" or "Not Enough Evidence" instead of "Supported".
- If the record itself says the claim is exaggerated, misleading, false, fabricated, or not supported, do NOT choose "Supported".

Decision Options:
Supported|Refuted|Conflicting Evidence/Cherrypicking|Not Enough Evidence

Rules:
{rules}

Record:
{record}
Your Judgement:
"""


verdict_extraction_prompt = """
Extract exactly one verdict label from the text.
Return ONLY the label, with no explanation.

Decision Options:
{options}

Rules:
{rules}

Conclusion:
{conclusion}
Extracted Verdict:
"""


def judge(record, decision_options, rules="", think=True):
    prompt = judge_prompt.format(
        record=(
            "Claim to Evaluate\n"
            f"{record.get('claim', '')}\n\n"
            "Relevant Evidence\n"
            f"{record.get('relevant_evidence', '')}\n\n"
            "QA Pair Analysis\n"
            f"{record.get('qa_text', '')}"
        ),
        rules=rules,
    )
    return prompt_local(
        prompt,
        think=False,
        use_adapter=True,
        max_new_tokens=256,
        do_sample=False,
    )


def extract_verdict(conclusion, decision_options, rules=""):
    prompt = verdict_extraction_prompt.format(conclusion=conclusion, options=decision_options, rules=rules)
    return prompt_local(
        prompt,
        think=False,
        use_adapter=True,
        max_new_tokens=64,
        do_sample=False,
    )
