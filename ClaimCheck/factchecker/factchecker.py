import re
import os
import json
import concurrent.futures
from modules import (
    planning,
    evidence_summarization,
    evidence_synthesis,
    evaluation,
    claim_matching,
    question_generation,
    question_answering,
    evidence_curation,
)
from tools import web_search, web_scraper
from report import report_writer
import fcntl

RULES_PROMPT = """
Supported
- The claim is directly and clearly backed by strong, credible evidence. Minor uncertainty or lack of detail does not disqualify a claim from being Supported if the main point is well-evidenced.
- Use Supported if the overall weight of evidence points to the claim being true, even if there are minor caveats or not every detail is confirmed.

Refuted
- The claim is contradicted by strong, credible evidence, or is shown to be fabricated, deceptive, or false in its main point.
- Use Refuted if the central elements of the claim are disproven, even if some minor details are unclear.
- Lack of any credible sources supporting the claim does not mean "Not Enough Evidence" - it means the claim is Refuted.

Conflicting Evidence/Cherrypicking
- Only use this if there are reputable sources that directly and irreconcilably contradict each other about the main point of the claim, and no clear resolution is possible after careful analysis.
- Do NOT use this for minor disagreements, incomplete evidence, or if most evidence points one way but a few sources disagree.

Not Enough Evidence
- Only use this if there is genuinely no relevant evidence available after a thorough search, or if the claim is too vague or ambiguous to evaluate.
- Do NOT use this if there is some evidence, even if it is weak, or if the claim is mostly clear but not every detail is confirmed.
- This is a last-resort option only.
"""

class FactChecker:
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=3, metadata=None):
        self.claim = claim
        self.date = date
        self.metadata = metadata or {}
        self.multimodal = multimodal if not (multimodal and image_path is None) else False
        self.image_path = image_path
        if identifier is None:
            from datetime import datetime
            identifier = datetime.now().strftime("%m%d%Y%H%M%S")
        self.identifier = identifier
        report_writer.init_report(claim, identifier)
        self.report_path = report_writer.REPORT_PATH
        print(f"Initialized report at: {self.report_path}")
        # Initialize the report dict for web use
        self.report = {
            "claim": self.claim,
            "date": self.date,
            "metadata": self.metadata,
            "identifier": self.identifier,
            "reformulated_claim": None,
            "generated_questions": [],
            "qa_pairs": [],
            "relevant_evidence": [],
            "claim_match": {"attempted": False, "matched": False, "used_urls": []},
            "actions": {},
            "reasoning": [],
            "judged_verdict": None,
            "verdict": None,
            "justification": None,
            "report_path": self.report_path
        }
        self.max_actions = max_actions

        # Save initial JSON report
        self.save_report_json()

    def save_report_json(self):
        """Save the report dictionary as report.json in the report_path folder"""
        try:
            json_path = os.path.join(os.path.dirname(self.report_path), 'report.json')
            with open(json_path, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(self.report, f, indent=2)
                fcntl.flock(f, fcntl.LOCK_UN)
            print(f"Report JSON saved to: {json_path}")
        except Exception as e:
            print(f"Error saving report JSON: {e}")

    def get_report(self):
        report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports', self.identifier, 'report.md'))
        try:
            with open(report_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading report: {e}"

    def add_relevant_evidence(self, url, text):
        text = (text or "").strip()
        url = (url or "").strip() or "unknown-url"
        if not text:
            return False

        candidate = {"url": url, "text": text}
        if candidate in self.report["relevant_evidence"]:
            return False

        self.report["relevant_evidence"].append(candidate)
        return True

    def build_verdict_record(self, max_items=8, max_summary_chars=700):
        relevant_lines = []
        for idx, item in enumerate(self.report.get("relevant_evidence", [])[-max_items:], start=1):
            text = (item.get("text") or "").strip()
            url = item.get("url") or "unknown-url"
            if len(text) > max_summary_chars:
                text = text[:max_summary_chars].rstrip() + "..."
            relevant_lines.append(f"[{idx}] {url}\n{text}")

        qa_lines = []
        for idx, item in enumerate(self.report.get("qa_pairs", [])[-max_items:], start=1):
            answer = (item.get("answer") or "").strip()
            question = (item.get("question") or "").strip()
            url = item.get("url") or "unknown-url"
            if len(answer) > max_summary_chars:
                answer = answer[:max_summary_chars].rstrip() + "..."
            qa_lines.append(f"[{idx}] Q: {question}\nA: {answer}\nURL: {url}")

        return {
            "claim": self.claim,
            "relevant_evidence": "\n\n".join(relevant_lines),
            "qa_text": "\n\n".join(qa_lines),
        }

    def metadata_text(self):
        useful = {
            key: value
            for key, value in self.metadata.items()
            if value not in (None, "", [], {}, "Unknown")
        }
        return json.dumps(useful, ensure_ascii=True)

    def generate_fallback_queries(self):
        claim = self.claim.replace('"', "'").strip()
        stopwords = {
            "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is",
            "are", "was", "were", "be", "been", "being", "that", "this", "with",
            "from", "by", "at", "as", "it", "its", "their", "his", "her",
        }
        keyword_tokens = []
        for token in re.findall(r"[A-Za-z0-9']+", claim):
            lower = token.lower()
            if lower in stopwords:
                continue
            keyword_tokens.append(token)
        targeted = " ".join(keyword_tokens[:8]).strip()
        queries = [
            claim,
            f"fact check {claim}",
            targeted if targeted else f"fact check {claim}",
        ]
        deduped = []
        seen = set()
        for query in queries:
            query = query.strip()
            if not query or query in seen:
                continue
            seen.add(query)
            deduped.append(f'web_search("{query}")')
        return deduped[:self.max_actions]

    def build_question_queries(self):
        reformulated_claim = question_generation.reformulate_claim(self.claim, self.metadata)
        questions = question_generation.generate_questions(self.claim, self.metadata)

        self.report["reformulated_claim"] = reformulated_claim
        self.report["generated_questions"] = questions
        self.save_report_json()

        question_queries = []
        for question in questions:
            query = question_generation.generate_query(self.claim, question)
            if not query:
                continue
            line = f'web_search("{query.replace(chr(34), chr(39))}")'
            question_queries.append((question, line))
        return question_queries

    def try_claim_matching(self, top_k=8):
        self.report["claim_match"]["attempted"] = True

        queries = [
            self.claim,
            f"fact check {self.claim}",
        ]
        matched_any = False
        used_urls = []

        for query in queries:
            urls, snippets = web_search.web_search(query, self.date, top_k=top_k)
            candidate_urls = claim_matching.extract_candidate_urls(urls)
            for url in candidate_urls:
                used_urls.append(url)
                content = web_scraper.scrape_url_content(url)
                match_result = claim_matching.summarize_factcheck_match(
                    self.claim,
                    self.metadata,
                    url,
                    content,
                )
                if not match_result.get("match"):
                    continue

                identifier = f"claim_match: {url}"
                self.report["actions"][identifier] = {
                    "action": "claim_match",
                    "query": query,
                    "results": {
                        url: {
                            "url": url,
                            "snippet": "",
                            "summary": match_result.get("summary", ""),
                            "verdict_hint": match_result.get("verdict_hint", "Unknown"),
                        }
                    },
                }
                report_writer.append_raw(f"claim_match('{query}') result: {url}")
                report_writer.append_evidence(
                    f"claim_match('{query}') summary: {match_result.get('summary', '')}"
                )
                self.add_relevant_evidence(url, match_result.get("summary", ""))
                matched_any = True

        self.report["claim_match"]["matched"] = matched_any
        self.report["claim_match"]["used_urls"] = used_urls
        self.save_report_json()
        return matched_any

    def ensure_multiple_queries(self, action_lines):
        normalized = []
        seen_queries = set()
        for line in action_lines:
            m = re.fullmatch(r'(\w+)_search\("([^"]+)"\)', line, re.IGNORECASE)
            if not m:
                continue
            query = m.group(2).strip()
            if query in seen_queries:
                continue
            seen_queries.add(query)
            normalized.append(f'web_search("{query}")')

        for line in self.generate_fallback_queries():
            if len(normalized) >= self.max_actions:
                break
            query = re.fullmatch(r'(\w+)_search\("([^"]+)"\)', line, re.IGNORECASE).group(2)
            if query in seen_queries:
                continue
            seen_queries.add(query)
            normalized.append(line)
        return normalized[:self.max_actions]

    def parse_verdict_response(self, verdict_text, allowed_verdicts):
        verdict_text = (verdict_text or "").strip()

        backtick_match = re.findall(r"`([^`]+)`", verdict_text)
        for candidate in reversed(backtick_match):
            candidate = candidate.strip()
            if candidate in allowed_verdicts:
                return candidate, ""

        json_match = re.search(r"\{.*\}", verdict_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                verdict = (data.get("verdict") or data.get("classification") or "").strip()
                justification = (data.get("justification") or "").strip()
                if verdict in allowed_verdicts:
                    return verdict, justification
            except json.JSONDecodeError:
                pass

        for verdict in allowed_verdicts:
            if re.search(rf"\b{re.escape(verdict)}\b", verdict_text, re.IGNORECASE):
                return verdict, ""
        return "", ""

    def process_action_line(self, line):
        try:
            m = re.match(r'(\w+)_search\("([^"]+)"\)', line)
            if m:
                action, query = m.groups()
                action_entry = {
                    "action": action + "_search",
                    "query": query,
                    "results": None
                }
                identifier = f'{action}: {query}'
                if identifier in self.report["actions"]:
                    print(f"Skipping duplicate action: {identifier}")
                    return

                if action == 'web':
                    self.report["actions"][identifier] = action_entry
                    urls, snippets = web_search.web_search(query, self.date, top_k=5)

                    # Default with snippets from web_search
                    self.report["actions"][identifier]["results"] = {url: {"snippet": snippet, 'url':url, 'summary': None} for url, snippet in zip(urls, snippets)}
                    self.save_report_json()

                    def process_result(result):
                        scraped_content = web_scraper.scrape_url_content(result)
                        summary = evidence_summarization.summarize(self.claim, scraped_content, result, record=self.get_report())

                        if "NONE" in summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None

                        print(f"Web search result: {result}, Summary: {summary}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}') summary: {summary}")

                        self.report["actions"][identifier]["results"][result]["summary"] = summary
                        self.add_relevant_evidence(result, summary)
                        self.save_report_json()

                    with concurrent.futures.ThreadPoolExecutor() as result_executor:
                        processed = list(result_executor.map(process_result, urls))
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")

    def process_question(self, question, query, top_k=5):
        identifier = f"question: {question}"
        self.report["actions"][identifier] = {
            "action": "question_search",
            "query": query,
            "question": question,
            "results": {},
        }
        urls, snippets = web_search.web_search(query, self.date, top_k=top_k)

        for url, snippet in zip(urls, snippets):
            evidence_text = web_scraper.scrape_url_content(url)
            answer = question_answering.answer_question(self.claim, question, snippet, evidence_text)
            answer_clean = (answer or "").strip()
            self.report["actions"][identifier]["results"][url] = {
                "url": url,
                "snippet": snippet,
                "answer": answer_clean,
            }

            if re.search(r"answer not found", answer_clean, re.IGNORECASE):
                continue

            useful = evidence_curation.is_useful(self.claim, f"Question: {question}\nAnswer: {answer_clean}")
            self.report["actions"][identifier]["results"][url]["useful"] = useful

            qa_item = {"question": question, "answer": answer_clean, "url": url}
            if qa_item not in self.report["qa_pairs"]:
                self.report["qa_pairs"].append(qa_item)
            self.add_relevant_evidence(url, answer_clean)
            report_writer.append_evidence(f"Q: {question}\nA: {answer_clean}\nURL: {url}")
            self.save_report_json()

    def extract_action_lines(self, text):
        action_pattern = r'(\w+)_search\("([^"]+)"\)'
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        candidates = []

        for block in code_blocks:
            candidates.extend([x.strip() for x in block.splitlines()])

        if not candidates:
            candidates = [x.strip() for x in text.split('\n')]

        valid = [line for line in candidates if re.fullmatch(action_pattern, line, re.IGNORECASE)]
        return valid

    def predict_verdict(self, verdict_record, force_judge=False):
        allowed_verdicts = ["Supported", "Refuted", "Conflicting Evidence/Cherrypicking", "Not Enough Evidence"]
        max_judge_tries = 3
        judge_tries = 0
        pred_verdict = ''
        pred_justification = ''
        rules = RULES_PROMPT
        verdict = ""

        print("[PIPELINE] Stage 5/5: verdict prediction", flush=True)
        print(
            f"[PIPELINE] verdict_record_sizes evidence={len(verdict_record['relevant_evidence'])} qa={len(verdict_record['qa_text'])}",
            flush=True,
        )
        if (
            not force_judge
            and not verdict_record["relevant_evidence"].strip()
            and not verdict_record["qa_text"].strip()
        ):
            pred_verdict = "Not Enough Evidence"
            pred_justification = "No relevant evidence or question-answer pairs were collected."
            verdict = pred_justification
            print("[PIPELINE] No evidence collected; using Not Enough Evidence.", flush=True)
        else:
            while judge_tries < max_judge_tries:
                verdict = evaluation.judge(
                    record=verdict_record,
                    decision_options="Supported|Refuted|Conflicting Evidence/Cherrypicking|Not Enough Evidence",
                    rules=rules,
                    think=None
                )
                print(f"Judged verdict (try {judge_tries+1}):\n{verdict}")
                pred_verdict, pred_justification = self.parse_verdict_response(verdict, allowed_verdicts)
                if pred_verdict in allowed_verdicts:
                    break
                judge_tries += 1

            if pred_verdict not in allowed_verdicts:
                print("Original extraction failed, falling back to most frequent decision option...")
                option_counts = {}
                for option in allowed_verdicts:
                    count = verdict.lower().count(option.lower())
                    if count > 0:
                        option_counts[option] = count

                if option_counts:
                    pred_verdict = max(option_counts, key=option_counts.get)
                    print(f"Fallback verdict selected: {pred_verdict} (appeared {option_counts[pred_verdict]} times)")
                else:
                    print("No decision options found in verdict, using extract_verdict from judge.py...")
                    try:
                        extracted = evaluation.extract_verdict(verdict, "Supported|Refuted|Conflicting Evidence/Cherrypicking|Not Enough Evidence", rules)
                        pred_verdict = extracted.strip()
                        print(f"extract_verdict returned: {pred_verdict}")
                    except Exception as e:
                        print(f"extract_verdict failed: {e}")
                        pred_verdict = "INVALID VERDICT"
                        print("No decision options found in verdict, defaulting to 'INVALID VERDICT'.")

        report_writer.append_verdict(verdict)
        self.report["judged_verdict"] = verdict
        self.report["verdict"] = pred_verdict
        self.report["justification"] = pred_justification
        self.save_report_json()
        return pred_verdict, report_writer.REPORT_PATH

    def run(self):
        if os.getenv("CLAIMCHECK_DISABLE_RETRIEVAL", "").strip().lower() in {"1", "true", "yes", "on"}:
            print("[PIPELINE] Retrieval disabled by CLAIMCHECK_DISABLE_RETRIEVAL", flush=True)
            verdict_record = {
                "claim": self.claim,
                "relevant_evidence": "",
                "qa_text": "",
            }
            return self.predict_verdict(verdict_record, force_judge=True)

        print("[PIPELINE] Stage 1/5: claim matching", flush=True)
        claim_match_found = self.try_claim_matching()
        print(f"Claim matching found evidence: {claim_match_found}", flush=True)

        print("[PIPELINE] Stage 2/5: planning actions", flush=True)
        if self.multimodal == True:
            actions = "All"
        else:
            actions = ["web_search"]#, "image_search"]

        seeded_question_actions = self.build_question_queries()
        actions = planning.plan(
            self.claim,
            record=self.get_report(),
            actions=actions,
            reformulated_claim=self.report.get("reformulated_claim") or self.claim,
            questions=self.report.get("generated_questions") or [],
        )
        report_writer.append_iteration_actions(1, actions)
        print(f"Proposed actions for claim '{self.claim}':\n{actions}")

        action_lines = self.extract_action_lines(actions)
        print(f"Extracted action lines: {action_lines}")

        print(f"Total action lines: {len(action_lines)}")
        print(f"Max actions allowed: {self.max_actions}")

        if not action_lines:
            action_lines = []
        action_lines = [line for _, line in seeded_question_actions] + action_lines
        if not action_lines:
            action_lines = self.generate_fallback_queries()
            print(f"No valid action lines extracted. Falling back to claim search: {action_lines}")

        action_lines = self.ensure_multiple_queries(action_lines)
        print(f"Normalized action lines: {action_lines}")

        if action_lines and len(action_lines) > self.max_actions:
            print(f"Limiting actions to the first {self.max_actions} lines.")
            action_lines = action_lines[:self.max_actions]
        
        print(f"Processing action lines: {action_lines}")

        # block multithreading until everything above is done
        

        print("[PIPELINE] Stage 3/5: retrieving evidence", flush=True)
        for question, line in seeded_question_actions:
            m = re.match(r'web_search\("([^"]+)"\)', line)
            if not m:
                continue
            self.process_question(question, m.group(1))

        seeded_lines = {line for _, line in seeded_question_actions}
        for line in action_lines:
            if line in seeded_lines:
                continue
            self.process_action_line(line)

        # Save the initial report after planning
        self.save_report_json()

        iterations = 0
        seen_action_lines = set(action_lines)
        while iterations <= 2:
            print(f"[PIPELINE] Stage 4/5: synthesis iteration={iterations+1}", flush=True)
            reasoning = evidence_synthesis.develop(record=self.get_report())

            print(f"Developed reasoning:\n{reasoning}")
            report_writer.append_reasoning(reasoning)

            self.report["reasoning"].append(reasoning)
            self.save_report_json()
            reasoning_action_lines = self.extract_action_lines(reasoning)
            if not reasoning_action_lines and re.search(r'^\s*NONE\s*$', reasoning, re.IGNORECASE | re.MULTILINE):
                reasoning_action_lines = ['NONE']

            print(f"Extracted reasoning action lines: {reasoning_action_lines}")

            if not reasoning_action_lines or (len(reasoning_action_lines) == 1 and reasoning_action_lines[0].strip().lower() == 'none'):
                break

            if any(line in seen_action_lines for line in reasoning_action_lines):
                print("Duplicate action line detected. Stopping iterations.")
                break

            seen_action_lines.update(reasoning_action_lines)

            for line in reasoning_action_lines:
                self.process_action_line(line)

            iterations += 1

        verdict_record = self.build_verdict_record()
        return self.predict_verdict(verdict_record)

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal = False, image_path = None, max_actions=3, metadata=None):
    return FactChecker(claim, date, identifier, multimodal, image_path, max_actions, metadata).run()
