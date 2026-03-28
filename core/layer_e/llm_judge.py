import json
import re
from dataclasses import dataclass
from typing import Literal
from urllib import error, request

from .utils import (
    LLM_BASE_URL,
    RUNTIME_MODEL,
    TEMPERATURE,
    JUDGE_MODE,
    JudgeOutput,
    BASE_SYSTEM_PROMPT,
    FINETUNED_SYSTEM_PROMPT,
    build_user_prompt,
)


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(BLOCK|ALLOW)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class _JudgeParseResult:
    decision: Literal["allow", "block"]
    rationale: str


class LLMJudge:
    def __init__(self, llm_base_url=LLM_BASE_URL, model_name=RUNTIME_MODEL, temperature=TEMPERATURE, timeout_s=30.0, max_retries=2, max_new_tokens=16, no_think_default=True, judge_mode=JUDGE_MODE, ):
        self.llm_base_url = llm_base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.max_new_tokens = int(max_new_tokens)
        self.no_think_default = bool(no_think_default)
        self.judge_mode = judge_mode

    def _post_chat(self, messages, no_think):
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": messages,
            "think": not no_think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }

        endpoint = f"{self.llm_base_url}/api/chat"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_output(content):
        verdict_match = _VERDICT_RE.search(content)
        if verdict_match is None:
            bare = content.strip().upper()
            if bare.startswith("BLOCK"):
                return _JudgeParseResult(decision="block", rationale="Model returned BLOCK verdict")
            if bare.startswith("ALLOW"):
                return _JudgeParseResult(decision="allow", rationale="Model returned ALLOW verdict")
            return None

        decision = "block" if verdict_match.group(1).upper() == "BLOCK" else "allow"
        rationale_match = _RATIONALE_RE.search(content)
        rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided"
        return _JudgeParseResult(decision=decision, rationale=rationale)

    def call_judge(self, prompt, *, no_think=None, max_retries=None):
        retries = self.max_retries if max_retries is None else int(max_retries)
        use_no_think = self.no_think_default if no_think is None else bool(no_think)

        last_error = ""
        for attempt in range(max(retries, 1)):
            try:
                response = self._post_chat(
                    messages=[
                        {
                            "role": "system",
                            "content": FINETUNED_SYSTEM_PROMPT if self.judge_mode == "finetuned" else BASE_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": build_user_prompt(prompt, mode=self.judge_mode)},
                    ],
                    no_think=use_no_think,
                )
                message = response.get("message", {})
                content = str(message.get("content", "")).strip()
                reasoning = message.get("thinking")
                prompt_tokens = response.get("prompt_eval_count")
                completion_tokens = response.get("eval_count")

                prompt_tokens_int = int(prompt_tokens) if isinstance(prompt_tokens, int) else None
                completion_tokens_int = int(completion_tokens) if isinstance(completion_tokens, int) else None
                total_tokens_int = None
                if prompt_tokens_int is not None or completion_tokens_int is not None:
                    total_tokens_int = int((prompt_tokens_int or 0) + (completion_tokens_int or 0))

                parsed = self._parse_output(content)
                if parsed is None:
                    raise ValueError(f"Could not parse verdict from response: {content[:120]}")

                return JudgeOutput(
                    decision=parsed.decision,
                    rationale=parsed.rationale,
                    model=self.model_name,
                    no_think=use_no_think,
                    raw_response=content,
                    reasoning_trace=str(reasoning).strip() if reasoning else None,
                    prompt_tokens=prompt_tokens_int,
                    completion_tokens=completion_tokens_int,
                    total_tokens=total_tokens_int,
                )
            except (ValueError, error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt < retries - 1:
                    continue

        return JudgeOutput(
            decision="block",
            rationale=f"Layer E fallback block after runtime/parsing failure: {last_error[:140]}",
            model=self.model_name,
            no_think=use_no_think,
            raw_response="",
            reasoning_trace=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )