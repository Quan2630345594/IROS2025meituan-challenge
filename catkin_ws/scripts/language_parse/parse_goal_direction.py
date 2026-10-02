#!/usr/bin/env python3
"""Ollama JSON-schema parsing with a compatible alternating-string interface."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

if __package__:
    from .contract import ROOT, DIRECTIONS, OBJECTS, ParseValidationError, get_schema, load_result, system_prompt, validate_result
else:
    from contract import ROOT, DIRECTIONS, OBJECTS, ParseValidationError, get_schema, load_result, system_prompt, validate_result


class InferenceError(RuntimeError):
    """Ollama is unavailable, returned an error, or truncated its output."""


class ClarificationRequired(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__("; ".join(issue["message"] for issue in result["issues"]))


def _endpoint(base_url):
    url = base_url.rstrip("/")
    # Accept the original script's OpenAI-compatible base URL configuration.
    if url.endswith("/v1"):
        url = url[:-3]
    if url.endswith("/api"):
        url = url[:-4]
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP(S) Ollama server URL")
    return url + "/api/chat"


def ollama_chat(messages, *, model=None, base_url=None, schema=None, timeout=120,
                temperature=0, num_predict=2048, num_ctx=8192):
    """One native /api/chat request; no retries, regex repair or majority vote."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    payload = {
        "model": model or os.environ.get("LANGUAGE_PARSE_MODEL", "qwen2.5vl:7b"),
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "seed": 42, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if schema is not None:
        payload["format"] = schema
    request = Request(_endpoint(base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")),
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise InferenceError("Ollama HTTP %s: %s" % (exc.code, detail)) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise InferenceError("Ollama request failed: %s" % exc) from exc
    if not isinstance(envelope, dict):
        raise InferenceError("Ollama response must be an object")
    if envelope.get("error"):
        raise InferenceError(str(envelope["error"]))
    if envelope.get("done") is not True or envelope.get("done_reason") == "length":
        raise InferenceError("Ollama output incomplete/truncated; increase num_predict or shorten instruction")
    message = envelope.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise InferenceError("Ollama response is missing message.content")
    return message["content"].strip()


def openai_chat(messages, *, model=None, base_url=None, schema=None, timeout=120,
                temperature=0, num_predict=2048, num_ctx=None):
    """Schema-capable OpenAI-compatible serving (e.g. vLLM merged LoRA weights)."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP(S) OpenAI-compatible server URL")
    if not url.endswith("/v1"):
        url += "/v1"
    payload = {"model": model or os.environ.get("LANGUAGE_PARSE_MODEL", "navigation-lora"),
               "messages": messages, "stream": False, "temperature": temperature,
               "seed": 42, "max_tokens": num_predict}
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "navigation_instruction", "strict": True, "schema": schema}}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request(url + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise InferenceError("OpenAI-compatible server HTTP %s: %s" % (
            exc.code, exc.read(2048).decode("utf-8", errors="replace"))) from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise InferenceError("OpenAI-compatible server request failed: %s" % exc) from exc
    try:
        choice = envelope["choices"][0]
        if choice["finish_reason"] != "stop":
            raise InferenceError("Model output incomplete/refused: " + str(choice["finish_reason"]))
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("content is not a string")
    except (KeyError, IndexError, TypeError) as exc:
        raise InferenceError("OpenAI-compatible response missing completed message.content") from exc
    return content.strip()


def chat(messages, *, backend="ollama", **kwargs):
    if backend == "ollama":
        return ollama_chat(messages, **kwargs)
    if backend == "openai":
        return openai_chat(messages, **kwargs)
    raise ValueError("backend must be ollama or openai")


def build_messages(instruction, few_shot=True):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    messages = [{"role": "system", "content": system_prompt()}]
    if few_shot:
        examples = json.loads((ROOT / "data" / "few_shot.json").read_text(encoding="utf-8"))
        for example in examples:
            messages.extend([
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": json.dumps(validate_result(example["output"]), ensure_ascii=False)},
            ])
    messages.append({"role": "user", "content": instruction.strip()})
    return messages


def parse_instruction(language_instr, *, model=None, base_url=None, timeout=120, few_shot=True, backend="ollama"):
    """Return validated JSON; unresolved semantics remain needs_clarification."""
    text = chat(build_messages(language_instr, few_shot), backend=backend, model=model, base_url=base_url,
                schema=get_schema(), timeout=timeout)
    return load_result(text)


def to_legacy(result):
    """Convert only complete sequences. The legacy interface cannot represent gaps."""
    validate_result(result)
    if result["status"] != "complete":
        raise ClarificationRequired(result)
    # Serialize labels individually: multiword classes remain atomic.
    return ", ".join(json.dumps(label) for step in result["steps"]
                     for label in (step["direction"], step["object"]))


def from_legacy(text):
    if not isinstance(text, str) or not text.strip():
        raise ParseValidationError("legacy output must be nonempty")
    try:
        labels = json.loads("[" + text + "]")
    except json.JSONDecodeError as exc:
        raise ParseValidationError("legacy output requires quoted comma-separated labels") from exc
    if not labels or len(labels) % 2:
        raise ParseValidationError("legacy output requires a nonzero even number of labels")
    return validate_result({"status": "complete", "steps": [
        {"direction": labels[i], "object": labels[i + 1], "target_gap_m": None}
        for i in range(0, len(labels), 2)], "issues": []})


def validate_direction(direction):
    try:
        from_legacy(direction)
        return True
    except (ValueError, TypeError):
        return False


def standardize_output(text):
    """Compatibility helper; validate quoted labels without guessing word boundaries."""
    return to_legacy(from_legacy(text))


def parse_goal_direction(language_instr, **kwargs):
    """Original API: alternating quoted labels; ambiguity raises ClarificationRequired.

    For target gaps, use parse_instruction(), since this API discards gap fields.
    """
    return to_legacy(parse_instruction(language_instr, **kwargs))


def analyze_sample_coverage():
    from collections import Counter
    counts = Counter()
    records = (ROOT / "data" / "train.records.jsonl").read_text(encoding="utf-8").splitlines()
    for line in records:
        for step in json.loads(line)["output"]["steps"]:
            counts[step["direction"]] += 1
            counts[step["object"]] += 1
    return {**{label: counts[label] for label in DIRECTIONS + OBJECTS}, "total": len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-few-shot", action="store_true", help="Use for the LoRA model trained on the same system prompt")
    parser.add_argument("--legacy", action="store_true", help="Discard target gaps and emit original interface")
    args = parser.parse_args()
    try:
        result = parse_instruction(args.instruction, model=args.model, base_url=args.base_url,
                                   timeout=args.timeout, few_shot=not args.no_few_shot, backend=args.backend)
        print(to_legacy(result) if args.legacy else json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "complete" else 2
    except ClarificationRequired as exc:
        print(json.dumps(exc.result, ensure_ascii=False, indent=2))
        return 2
    except (ValueError, InferenceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
