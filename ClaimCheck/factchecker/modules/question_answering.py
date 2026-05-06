from modules.llm import prompt_local


answer_prompt = """
Instructions
You are a fact-checker. Your overall motivation is to verify a given Claim. In order to find evidence that
helps the fact-checking work, you just ran a web search which yielded a Search Result. Your task right
now is to answer the Question given below. Adhere to the following rules:
The length of your Answer should be between one sentence and one paragraph.
If applicable and useful, you may directly cite relevant excerpts from the source. In that case, put the
citation into quotation marks.
If the search result does not contain sufficient information to answer the Question or is unrelated to the
question completely, respond simply with Answer Not Found.
If the evidence does not answer the question, but can otherwise be highly useful for the fact-check, you
must respond with "The evidence is useful, but does not answer the question." This is a very rare case.
Claim: {claim}
Question
{question}
Search Result
Summary: {snippet}
Evidence:
{evidence_text}
Your Answer
"""


def answer_question(claim, question, snippet, evidence_text):
    prompt = answer_prompt.format(
        claim=claim,
        question=question,
        snippet=snippet or "",
        evidence_text=evidence_text or "",
    )
    return prompt_local(
        prompt,
        think=False,
        max_new_tokens=192,
        do_sample=False,
    ).strip()
