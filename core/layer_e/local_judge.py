import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from .utils import FINETUNED_SYSTEM_PROMPT, JudgeOutput, build_user_prompt


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(BLOCK|ALLOW)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE\s*:\s*(.+)", re.IGNORECASE)


@dataclass(frozen=True)
class _LocalJudgeState:
    model_dir: str
    model_name: str


@dataclass(frozen=True)
class _JudgeParseResult:
    decision: str
    rationale: str


class LocalTeacherJudge:
    @staticmethod
    def _load_tokenizer(model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        except ValueError as exc:
            if "Tokenizer class" not in str(exc):
                raise

            tokenizer_json = Path(model_dir) / "tokenizer.json"
            if not tokenizer_json.exists():
                raise

            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
            special_tokens_map = Path(model_dir) / "special_tokens_map.json"
            if special_tokens_map.exists():
                with special_tokens_map.open("r", encoding="utf-8") as handle:
                    token_map = json.load(handle)
                for key, value in token_map.items():
                    token_value = value.get("content") if isinstance(value, dict) else value
                    if isinstance(token_value, str):
                        setattr(tokenizer, key, token_value)

            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def __init__(
        self,
        model_dir,
        model_name=None,
        temperature=0.0,
        timeout_s=30.0,
        max_retries=2,
        max_new_tokens=16,
        no_think_default=True,
    ):
        self.state = _LocalJudgeState(
            model_dir=str(model_dir),
            model_name=str(model_name or model_dir),
        )
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.max_new_tokens = int(max_new_tokens)
        self.no_think_default = bool(no_think_default)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = self._load_tokenizer(self.state.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.state.model_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        cast(Any, self.model).to(self.device)
        cast(Any, self.model).eval()

    def _build_messages(self, prompt):
        return [
            {"role": "system", "content": FINETUNED_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prompt, mode="finetuned")},
        ]

    def _render_prompt(self, prompt):
        messages = self._build_messages(prompt)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(message["content"] for message in messages)

    def _generate_completion(self, prompt):
        rendered_prompt = self._render_prompt(prompt)
        tokenizer = cast(Any, self.tokenizer)
        model = cast(Any, self.model)
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        generated_ids = output_ids[0][input_length:]
        return str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()

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
                content = self._generate_completion(prompt)
                parsed = self._parse_output(content)
                if parsed is None:
                    raise ValueError(f"Could not parse verdict from response: {content[:120]}")

                return JudgeOutput(
                    decision=parsed.decision, # type: ignore
                    rationale=parsed.rationale,
                    model=self.state.model_name,
                    no_think=use_no_think,
                    raw_response=content,
                    reasoning_trace=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                )
            except (ValueError, RuntimeError, TypeError, torch.cuda.OutOfMemoryError) as exc:
                last_error = str(exc)
                if attempt < retries - 1:
                    continue

        return JudgeOutput(
            decision="block",
            rationale=f"Layer E fallback block after local judge failure: {last_error[:140]}",
            model=self.state.model_name,
            no_think=use_no_think,
            raw_response="",
            reasoning_trace=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )