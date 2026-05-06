import json
import os
from pathlib import Path


SRC = Path("/home/airs/homework/llm_fp/data/averitec/train.json")
OUT = Path("/home/airs/homework/llm_fp/data/averitec/train_sft.jsonl")


SYSTEM = (
    "You are a fact-checking assistant. "
    "Given a claim and evidence, return ONLY JSON with keys "
    '"verdict" and "justification". '
    'The verdict must be exactly one of: Supported, Refuted, '
    'Conflicting Evidence/Cherrypicking, Not Enough Evidence.'
)


def build_evidence(item, max_questions=4):
    parts = []
    justification = (item.get("justification") or "").strip()
    if justification:
        parts.append(f"Gold justification: {justification}")

    for qa in item.get("questions", [])[:max_questions]:
        q = (qa.get("question") or "").strip()
        answers = qa.get("answers") or []
        ans = ""
        if answers:
            ans = (answers[0].get("answer") or "").strip()
        if q and ans:
            parts.append(f"Q: {q}\nA: {ans}")
    return "\n\n".join(parts).strip()


def main():
    data = json.loads(SRC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w") as f:
        for item in data:
            evidence = build_evidence(item)
            user = (
                f"Claim: {item['claim']}\n"
                f"Claim Date: {item.get('claim_date', '')}\n"
                f"Evidence:\n{evidence}\n"
            )
            assistant = json.dumps(
                {
                    "verdict": item["label"],
                    "justification": (item.get("justification") or "").strip(),
                },
                ensure_ascii=False,
            )
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(OUT)


if __name__ == "__main__":
    main()
