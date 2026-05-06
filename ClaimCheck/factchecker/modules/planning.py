from modules.llm import prompt_local

plan_prompt = """Instructions
You are a fact-checker planning web searches for a claim.
Generate clear, specific, and relevant search actions that help assess the accuracy of the claim.

Guidelines:
- Focus on the essential details of the claim and the provided questions.
- Use the reformulated claim and questions when they help retrieval.
- Prefer queries that target different subquestions instead of near-duplicates.
- Keep each query concise and retrieval-oriented.
- Output ONLY one markdown code block.
- Inside the code block, output 2 or 3 distinct web_search(...) lines.
- Do NOT output any explanation, bullets, notes, or thinking.

Examples:
```markdown
web_search("New Zealand Food Bill 2020")
web_search("fact check New Zealand Food Bill 2020")
web_search("New Zealand Food Bill gardening hoax")
```

Claim: {claim}
Reformulated claim: {reformulated_claim}
Questions: {questions}
Code block:
"""

def plan(claim, record="", examples="", actions=None, think=True, reformulated_claim="", questions=None):
    prompt = plan_prompt.format(
        claim=claim,
        reformulated_claim=reformulated_claim or claim,
        questions=questions or [],
    )
    response = prompt_local(
        prompt,
        think=False,
        max_new_tokens=128,
        do_sample=False,
    )
    return response
