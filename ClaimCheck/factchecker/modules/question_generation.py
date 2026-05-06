import json
import re

from modules.llm import prompt_local


reformulation_prompt = """
Instructions
You are presented with a raw claim, with additional metadata like Content date or speaker. Your task right
now is to interpret the claim. That is, identify the speaker's core message and write down the main
point(s) using your own words. Do not ask any questions and only use the metadata provided to interpret
the claim. Be concise and write only one paragraph.
# Content
Original Claim: {claim}
Metadata:
- Speaker: {speaker}
- Date: {claim_date}
- Origin URL: {original_claim_url}
- Reporting Source: {reporting_source}
- Location ISO Code: {location_iso_code}
# Interpretation
"""


question_prompt = """
Instructions
You are a fact-checker verifying a claim. Your task is to generate clear, specific, and relevant fact-
checking questions that help assess the accuracy of the claim.
Guidelines:
- Focus on the essential details of the claim. The questions should help find direct evidence to confirm or
  refute it.
- Only use metadata (such as date, speaker, or source) when it is necessary for verification.
- Each question should be concise and directly related to the claim.
- Format each question using backticks like `this`.
- Do not repeat questions already addressed in prior fact-checking records.
Examples:
Claim: "New Zealand's new Food Bill bans gardening."
Questions:
1. `Does New Zealand's Food Bill ban home gardening?`
2. `What are the key regulations in the New Zealand Food Bill related to gardening?`
3. `Has the New Zealand government enforced any gardening restrictions under this bill?`
Claim: "Video of a man blowing vape smoke through various face masks shows that they do not help prevent the spread of coronavirus."
Questions:
1. `How does coronavirus spread?`
2. `Do scientific studies show that face masks reduce the spread of coronavirus?`
3. `Does the ability of vape smoke to pass through a mask indicate ineffectiveness against viruses?`
Claim: "The Nigerian government is donating $600 million to Democratic presidential nominee Joe Biden's campaign."
Questions:
1. `Is there evidence that the Nigerian government donated $600 million to Joe Biden's campaign?`
2. `Are foreign governments legally allowed to donate to U.S. presidential campaigns?`
3. `Has the Biden campaign reported any donations from Nigeria?`
# Claim to Verify
Claim: {claim}
Metadata: {metadata}
## Questions:
"""


query_prompt = """
Instructions
You are a fact-checker optimizing a question for web search to retrieve relevant evidence.
Guidelines:
- Ensure the query makes sense in the context of the question.
- Add claim-specific context only if absolutely necessary to improve relevance.
- Keep the query concise and structured for effective search results.
- Format the final query using backticks like `this` without extra formatting or explanation.
## Question
{question}
## Claim
{claim}
## Search Query:
"""


def reformulate_claim(claim, metadata):
    prompt = reformulation_prompt.format(
        claim=claim,
        speaker=(metadata or {}).get("speaker", ""),
        claim_date=(metadata or {}).get("claim_date", ""),
        original_claim_url=(metadata or {}).get("original_claim_url", ""),
        reporting_source=(metadata or {}).get("reporting_source", ""),
        location_iso_code=(metadata or {}).get("location_ISO_code", ""),
    )
    raw = prompt_local(
        prompt,
        think=False,
        max_new_tokens=128,
        do_sample=False,
    ).strip()
    return raw or claim


def generate_questions(claim, metadata):
    prompt = question_prompt.format(
        claim=claim,
        metadata=json.dumps(metadata or {}, ensure_ascii=True),
    )
    raw = prompt_local(
        prompt,
        think=False,
        max_new_tokens=256,
        do_sample=False,
    )
    questions = []
    for line in raw.splitlines():
        line = line.strip()
        m = re.search(r"`([^`]+)`", line)
        if m:
            questions.append(m.group(1).strip())
            continue
        line = re.sub(r"^\d+\.\s*", "", line).strip()
        if line.endswith("?"):
            questions.append(line)
    deduped = []
    seen = set()
    for q in questions:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped[:4]


def generate_query(claim, question):
    prompt = query_prompt.format(claim=claim, question=question)
    raw = prompt_local(
        prompt,
        think=False,
        max_new_tokens=64,
        do_sample=False,
    ).strip()
    m = re.search(r"`([^`]+)`", raw)
    if m:
        return m.group(1).strip()
    return raw.strip().strip("`")
