"""Stage 2: natural-language code review with the Gemini API.

If GEMINI_API_KEY is missing or the network is down, the app falls back to the
reference answer for known sample snippets and labels it clearly as offline.
"""
import json
import os
import re
import time
from typing import List

from pydantic import BaseModel, Field

from .samples import find_sample

# Tried in order until one works. Override with GEMINI_MODEL in .env
DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]


class Issue(BaseModel):
    title: str = Field(description="Short name of the problem, under 8 words")
    bug_type: str = Field(description="Category, e.g. Off-by-one error, Null dereference, SQL injection")
    severity: str = Field(description="Exactly one of: Critical, Major, Minor")
    lines: List[int] = Field(description="1-indexed line numbers where the bug is")
    explanation: str = Field(description="2-3 plain sentences: what goes wrong and with which input")
    fix_hint: str = Field(description="One sentence on how to fix it")


class Review(BaseModel):
    verdict: str = Field(description="Exactly 'buggy' or 'clean'")
    summary: str = Field(description="One sentence overall verdict")
    issues: List[Issue]
    fixed_code: str = Field(description="The complete corrected code, or the original if clean")
    fix_explanation: str = Field(description="What changed in fixed_code, 1-2 sentences")
    confidence: float = Field(description="0 to 1, how sure you are about the verdict")


SYSTEM = """You are a senior software engineer doing a careful code review.
You find real defects: logic errors, off-by-one errors, null or None handling,
resource leaks, injection, memory safety, concurrency and error-handling bugs.
Do not report style nits as bugs. If the code is correct, say verdict "clean"
with an empty issues list. Line numbers refer to the numbered listing you are given.
Explain in simple, plain English that a student can follow.

Example
Code:
1 | def last(items):
2 |     return items[len(items)]
Answer:
{"verdict": "buggy", "summary": "Indexing one past the end always crashes.",
 "issues": [{"title": "Index past the end of list", "bug_type": "Off-by-one error",
   "severity": "Major", "lines": [2],
   "explanation": "Valid indexes go up to len(items) - 1, so items[len(items)] raises IndexError for every list.",
   "fix_hint": "Use items[-1] and handle the empty list."}],
 "fixed_code": "def last(items):\\n    if not items:\\n        return None\\n    return items[-1]\\n",
 "fix_explanation": "Uses the last valid index and returns None for an empty list.",
 "confidence": 0.95}"""


class Reviewer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        custom = (os.getenv("GEMINI_MODEL") or "").strip()
        # your pinned model goes first, the defaults stay as backups
        self.candidates = ([custom] if custom else []) + [m for m in DEFAULT_MODELS if m != custom]
        self.active_model = None
        self.last_error = None
        self.client = None
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                # 30 s cap per call so a busy model never freezes the demo
                self.client = genai.Client(api_key=self.api_key,
                                           http_options=types.HttpOptions(timeout=30000))
            except Exception as e:
                self.last_error = f"Gemini SDK error: {e}"

    def status(self):
        return {
            "configured": bool(self.client),
            "model": self.active_model or self.candidates[0],
            "last_error": self.last_error,
        }

    def _prompt(self, code, language, detection):
        numbered = "\n".join(f"{i + 1:>3} | {line}" for i, line in enumerate(code.rstrip("\n").split("\n")))
        if detection:
            p = detection["probability"]
            ctx = (f"A fine-tuned {detection['display']} bug-detection model rated this code "
                   f"{p:.0%} likely to contain a defect "
                   f"({'flagged as buggy' if detection['label'] == 'buggy' else 'rated clean'}). "
                   "Treat this as a hint, not the truth: verify it yourself.")
        else:
            ctx = "No detector score is available. Review the code on its own."
        return f"{ctx}\n\nLanguage: {language}\n\nCode:\n{numbered}\n\nReturn the review as JSON."

    def review(self, code, language="python", detection=None):
        t0 = time.time()
        if self.client:
            try:
                review, model = self._call_gemini(code, language, detection)
                review["source"] = "gemini"
                review["model"] = model
                review["latency_ms"] = round((time.time() - t0) * 1000)
                return review
            except Exception as e:
                self.last_error = str(e)[:300]

        offline = self._offline(code)
        if offline:
            offline["latency_ms"] = round((time.time() - t0) * 1000)
            offline["notice"] = ("Gemini is not reachable, so this is the reference answer stored with the sample."
                                 if self.client else
                                 "No Gemini API key set, so this is the reference answer stored with the sample.")
            return offline
        raise RuntimeError(
            "Gemini review failed: " + (self.last_error or "no GEMINI_API_KEY in .env")
            + ". Offline answers exist only for the built-in samples.")

    def _call_gemini(self, code, language, detection):
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=Review,
        )
        prompt = self._prompt(code, language, detection)
        models = ([self.active_model] if self.active_model else []) + list(self.candidates)
        tried, last, discovered = set(), None, False
        while models:
            m = models.pop(0)
            if m in tried:
                continue
            tried.add(m)
            for attempt in (1, 2):
                try:
                    resp = self.client.models.generate_content(model=m, contents=prompt, config=config)
                    data = self._parse(resp)
                    self.active_model, self.last_error = m, None
                    return data, m
                except Exception as e:
                    last = e
                    msg = str(e).lower()
                    busy = any(w in msg for w in ("503", "unavailable", "overloaded", "high demand",
                                                  "429", "resource_exhausted", "quota", "deadline", "timed out"))
                    missing = any(w in msg for w in ("not found", "404", "not supported", "permission"))
                    bad_json = isinstance(e, (ValueError, KeyError, TypeError))
                    if busy and attempt == 1:
                        print(f"[reviewer] {m} is busy, retrying once")
                        time.sleep(1.5)
                        continue
                    if busy or missing or bad_json:
                        print(f"[reviewer] {m} failed ({str(e)[:60]}), trying the next model")
                        break
                    raise
            if not models and not discovered:
                # every name we know failed: ask Google which models this key can use
                discovered = True
                models = [x for x in self._discover_models() if x not in tried]
                print("[reviewer] trying models available to this key:", models[:5])
        raise last

    def _discover_models(self):
        """List Flash text models this API key can call, best guesses first."""
        found = []
        try:
            for m in self.client.models.list():
                name = (m.name or "").split("/")[-1]
                actions = getattr(m, "supported_actions", None) or []
                if actions and "generateContent" not in actions:
                    continue
                low = name.lower()
                if not low.startswith("gemini") or any(w in low for w in
                        ("image", "tts", "audio", "live", "embedding", "vision", "thinking", "computer")):
                    continue
                found.append(name)
        except Exception as e:
            print("[reviewer] could not list models:", e)
            return []

        def rank(n):
            low = n.lower()
            return ("flash" not in low, "preview" in low or "exp" in low, "lite" in low, n)
        return sorted(found, key=rank)[:6]

    @staticmethod
    def _parse(resp):
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, BaseModel):
            data = parsed.model_dump()
        else:
            text = resp.text or ""
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            data = json.loads(text)
        data["verdict"] = "buggy" if str(data.get("verdict", "")).lower().startswith("bug") else "clean"
        for issue in data.get("issues", []):
            sev = str(issue.get("severity", "Minor")).capitalize()
            issue["severity"] = sev if sev in ("Critical", "Major", "Minor") else "Minor"
        if data["verdict"] == "clean":
            data["issues"] = []
        return data

    @staticmethod
    def _offline(code):
        sample, which = find_sample(code)
        if not sample:
            return None
        if which == "fixed":
            return {"verdict": "clean", "summary": "This is the corrected version of a sample, no defect expected.",
                    "issues": [], "fixed_code": sample["fixed"], "fix_explanation": "No change needed.",
                    "confidence": 1.0, "source": "offline-reference", "model": "sample reference"}
        return {
            "verdict": "buggy",
            "summary": sample["explanation"].split(". ")[0].rstrip(".") + ".",
            "issues": [{"title": sample["title"], "bug_type": sample["bug_type"],
                        "severity": sample["severity"], "lines": sample["lines"],
                        "explanation": sample["explanation"], "fix_hint": "See the corrected code."}],
            "fixed_code": sample["fixed"],
            "fix_explanation": "Reference fix stored with this sample.",
            "confidence": 1.0,
            "source": "offline-reference",
            "model": "sample reference",
        }
