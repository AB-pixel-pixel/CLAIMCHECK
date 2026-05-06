from modules.llm import prompt_local


judge_prompt = """
Instructions
Determine the Claim's veracity by following these steps:
1. Briefly summarize the key insights from the fact-check (see Record) in at most one paragraph.
2. Write one paragraph about which one of the Decision Options applies best. Include the most appropriate decision option at the end and enclose it in backticks like `this`.

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
        max_new_tokens=256,
        do_sample=False,
    )


def extract_verdict(conclusion, decision_options, rules=""):
    prompt = verdict_extraction_prompt.format(conclusion=conclusion, options=decision_options, rules=rules)
    return prompt_local(
        prompt,
        think=False,
        max_new_tokens=64,
        do_sample=False,
    )
