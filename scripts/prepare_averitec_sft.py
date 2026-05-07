import json
import os
from pathlib import Path


SRC = Path(os.getenv("FT_SRC_PATH", "/home/airs/homework/llm_fp/data/averitec/train.json"))
OUT = Path(os.getenv("FT_OUT_PATH", "/home/airs/homework/llm_fp/data/averitec/train_sft_verdict.jsonl"))
MAX_QUESTIONS = int(os.getenv("FT_MAX_QUESTIONS", "4"))
MAX_ANSWERS_PER_QUESTION = int(os.getenv("FT_MAX_ANSWERS_PER_QUESTION", "1"))
ALLOWED_MEDIA = {
    "Web text",
    "PDF",
    "Web table",
}
INCLUDE_GOLD_JUSTIFICATION = os.getenv("FT_INCLUDE_GOLD_JUSTIFICATION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


SYSTEM = (
    "You are a fact-checking assistant. "
    "Given a claim and evidence, return ONLY JSON with keys "
    '"verdict" and "justification". '
    'The verdict must be exactly one of: Supported, Refuted, '
    'Conflicting Evidence/Cherrypicking, Not Enough Evidence.'
)


def build_evidence(item):
    parts = []
    kept_answers = 0

    if INCLUDE_GOLD_JUSTIFICATION:
        justification = (item.get("justification") or "").strip()
        if justification:
            parts.append(f"Reference justification: {justification}")

    for qa in (item.get("questions") or [])[:MAX_QUESTIONS]:
        question = (qa.get("question") or "").strip()
        answers = []
        for answer_item in (qa.get("answers") or [])[:MAX_ANSWERS_PER_QUESTION]:
            answer = (answer_item.get("answer") or "").strip()
            source_url = (answer_item.get("source_url") or "").strip()
            source_medium = (answer_item.get("source_medium") or "").strip()
            if not answer or not source_url or source_medium not in ALLOWED_MEDIA:
                continue
            answers.append(f"- {answer}\n  Source: {source_url}\n  Medium: {source_medium}")
            kept_answers += 1

        if question and answers:
            answer_block = "\n".join(answers)
            parts.append(f"Question: {question}\nAnswers:\n{answer_block}")

    return "\n\n".join(parts).strip(), kept_answers


def build_user_message(item, evidence):
    sections = [
        f"Claim: {item['claim']}",
        f"Claim Date: {item.get('claim_date', '')}",
        "Evidence Record:",
        evidence or "No evidence provided.",
    ]
    return "\n".join(sections) + "\n"


def main():
    data = json.loads(SRC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with_evidence = 0
    skipped = 0
    skipped_no_acceptable_evidence = 0
    total_answers_kept = 0

    with OUT.open("w") as f:
        for item in data:
            label = (item.get("label") or "").strip()
            claim = (item.get("claim") or "").strip()
            justification = (item.get("justification") or "").strip()
            if not label or not claim or not justification:
                skipped += 1
                continue

            evidence, kept_answers = build_evidence(item)
            if not evidence or kept_answers == 0:
                skipped_no_acceptable_evidence += 1
                continue
            with_evidence += 1
            total_answers_kept += kept_answers

            assistant = json.dumps(
                {
                    "verdict": label,
                    "justification": justification,
                },
                ensure_ascii=False,
            )
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_message(item, evidence)},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(
        json.dumps(
            {
                "output_path": str(OUT),
                "total_rows": total,
                "rows_with_evidence": with_evidence,
                "rows_without_evidence": total - with_evidence,
                "skipped_rows": skipped,
                "skipped_no_acceptable_evidence": skipped_no_acceptable_evidence,
                "total_answers_kept": total_answers_kept,
                "allowed_media": sorted(ALLOWED_MEDIA),
                "included_gold_justification": INCLUDE_GOLD_JUSTIFICATION,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
