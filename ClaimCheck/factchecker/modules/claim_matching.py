import json
from urllib.parse import urlparse

from modules.llm import prompt_local


TRUSTED_FACTCHECK_DOMAINS = [
    "africacheck.org",
    "factcheck.kz",
    "altnews.in",
    "boomlive.in",
    "vishvasnews.com",
    "factcheck.ge",
    "poynter.org",
    "factcheck.afp.com",
    "apnews.com",
    "reuters.com",
    "checkyourfact.com",
    "hoax-slayer.net",
    "leadstories.com",
    "fullfact.org",
    "truthorfiction.com",
    "politifact.com",
    "snopes.com",
]


claim_match_prompt = """
Can this fact-checking article provide a complete fact-check for the claim, including a clear verdict and
justification with relevant evidence?
Take into account the claim date and any other information important for fact-checking the claim.
Possible Verdicts:
- Supported: The knowledge from the fact-check supports or at least strongly implies the claim. Mere
  plausibility is not enough for this decision.
- Refuted: The knowledge from the fact-check clearly refutes the claim. The mere absence or lack of
  supporting evidence is not enough reason for being refuted. This includes fake news and deliberate
  misinformation.
- Conflicting Evidence/Cherrypicking: The knowledge from the fact-check contains conflicting evidence from
  multiple reliable sources. Even trying to resolve the conflicting sources through additional investigation
  was not successful.
Claim: {claim}
Claim date: {claim_date}
Metadata: {metadata}
Article: {article_text}
If the article cannot fulfill this requirement, respond with "No answer found." Otherwise, gather the key
evidence from the article that can be used for fact checking the claim and summarize them in at most
one paragraph.
"""


def is_trusted_factcheck_url(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in TRUSTED_FACTCHECK_DOMAINS)


def extract_candidate_urls(urls):
    return [url for url in urls if is_trusted_factcheck_url(url)]


def summarize_factcheck_match(claim, metadata, url, content):
    content = (content or "").strip()
    if not content or content == "Unable to Scrape":
        return {"match": False, "summary": "", "verdict_hint": "Unknown"}

    prompt = claim_match_prompt.format(
        claim=claim,
        claim_date=(metadata or {}).get("claim_date", ""),
        metadata=json.dumps(metadata or {}, ensure_ascii=True),
        article_text=content[:12000],
    )
    raw = prompt_local(
        prompt,
        think=False,
        max_new_tokens=192,
        do_sample=False,
    ).strip()

    if "no answer found" in raw.lower():
        return {"match": False, "summary": "", "verdict_hint": "Unknown"}

    return {"match": True, "summary": raw, "verdict_hint": "Unknown"}
