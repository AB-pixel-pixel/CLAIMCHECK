from modules.llm import prompt_local


curation_prompt = """
Instructions
You are a fact-checker. Your overall motivation is to verify a given Claim. In order to find evidence that
helps the fact-checking work, you just ran a web search which yielded a Search Result. Your task right
now is to determine if the Answer is useful to fact-checking the Claim. Follow the following rules:
An answer is useful even when it doesn't directly answer the question, if it provides highly relevant
information for fact-checking. It just has to be somewhat related to the Claim.
If the Answer is useful to fact-checking the Claim, respond only with "Yes".
If the Answer is not useful to fact-checking the Claim, respond only with "No".
Claim: {claim}
Question and Answer: {answer}
"""


def is_useful(claim, answer):
    prompt = curation_prompt.format(claim=claim, answer=answer)
    raw = prompt_local(
        prompt,
        think=False,
        max_new_tokens=16,
        do_sample=False,
    ).strip()
    return raw.lower().startswith("yes")
