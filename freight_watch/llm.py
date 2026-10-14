"""OPTIONAL: let an open-source LLM re-word a reason that has already been decided and grounded.

The verdict and the cited note are fixed *before* this step. The model only rephrases a draft, at
temperature 0, and its output is rejected (falling back to the template) if it adds a note id or a
number that was not in the draft. So the LLM can never change a verdict, a note id or a number.

Works with any OpenAI-compatible endpoint (default: local Ollama).
"""
import json
import re
import urllib.request

from .costlog import CostLog, estimate_tokens

SYSTEM_PROMPT = (
    "You rewrite short freight-cost explanations in plain English for a logistics manager. "
    "Use ONLY the facts in the DRAFT. Do not add causes, dates, numbers or note ids that are not in the DRAFT. "
    "Keep the verdict (Justified / Unexplained) and every note id exactly as given. Maximum 60 words."
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
NOTE_ID = re.compile(r"\bN\d{3}\b")


def build_messages(draft: str) -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DRAFT:\n{draft}\n\nRewrite the DRAFT."}]


def is_faithful(draft: str, rewritten: str) -> bool:
    """Reject rewrites that introduce new numbers / note ids, drop the cited ids, or flip the verdict."""
    if not rewritten.strip():
        return False
    if set(NOTE_ID.findall(rewritten)) != set(NOTE_ID.findall(draft)):
        return False
    if not set(NUMBER.findall(rewritten)) <= set(NUMBER.findall(draft)):
        return False
    verdict = "justified" if draft.lower().startswith("justified") else "unexplained"
    return rewritten.lower().lstrip().startswith(verdict)


def _http_transport(base_url: str):
    def call(payload: dict) -> dict:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    return call


class Rewriter:
    def __init__(self, mode: str, model: str, costlog: CostLog, base_url="http://localhost:11434/v1",
                 transport=None):
        self.mode, self.model, self.log = mode, model, costlog
        self.transport = transport or _http_transport(base_url)
        costlog.mode, costlog.model = mode, model

    def rewrite(self, draft: str) -> str:
        messages = build_messages(draft)
        if self.mode == "dry-run":  # estimate only; nothing is sent anywhere
            self.log.add(estimate_tokens(SYSTEM_PROMPT + messages[1]["content"]), estimate_tokens(draft), True)
            return draft
        payload = {"model": self.model, "messages": messages, "temperature": 0, "seed": 0, "max_tokens": 120}
        try:
            data = self.transport(payload)
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage") or {}
            self.log.add(usage.get("prompt_tokens", estimate_tokens(json.dumps(messages))),
                         usage.get("completion_tokens", estimate_tokens(text)), "usage" not in data)
        except Exception:  # network / format problem: keep the grounded template
            self.log.fallbacks_to_template += 1
            return draft
        if is_faithful(draft, text):
            return text
        self.log.fallbacks_to_template += 1
        return draft
