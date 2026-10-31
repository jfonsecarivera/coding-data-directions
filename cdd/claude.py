"""Claude API calls for the data filters, the judges and the review rewrites: a client, a concurrency throttle with retries, and
replies read as strict JSON."""
from __future__ import annotations

import json
import os
import random
import threading
import time

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"


def client(timeout: float = 300.0):
    """An Anthropic client. If ANTHROPIC_FEDERATION_RULE_ID is set, it authenticates with Workload Identity Federation from Google
    Cloud (ANTHROPIC_ORGANIZATION_ID, ANTHROPIC_SERVICE_ACCOUNT_ID and ANTHROPIC_WORKSPACE_ID must be set too); otherwise the SDK's
    default, an API key in ANTHROPIC_API_KEY."""
    import anthropic  # noqa: PLC0415
    if os.environ.get("ANTHROPIC_FEDERATION_RULE_ID"):
        import google.auth.transport.requests  # noqa: PLC0415
        import google.oauth2.id_token  # noqa: PLC0415

        def google_token():
            return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), "https://api.anthropic.com")

        creds = anthropic.WorkloadIdentityCredentials(
            identity_token_provider=google_token,
            federation_rule_id=os.environ["ANTHROPIC_FEDERATION_RULE_ID"],
            organization_id=os.environ["ANTHROPIC_ORGANIZATION_ID"],
            service_account_id=os.environ["ANTHROPIC_SERVICE_ACCOUNT_ID"],
            workspace_id=os.environ["ANTHROPIC_WORKSPACE_ID"],
        )
        return anthropic.Anthropic(credentials=creds, max_retries=1, timeout=timeout)
    return anthropic.Anthropic(max_retries=1, timeout=timeout)


class Throttle:
    """Concurrency limit that starts at `start` calls, adds `step` every `every` seconds without a rate-limit or overload error, and
    multiplies by 0.7 on one (never below `floor` or above `cap`)."""

    def __init__(self, start: int = 32, cap: int = 192, floor: int = 8, step: int = 8, every: float = 20.0):
        self.limit, self.cap, self.floor, self.step, self.every = start, cap, floor, step, every
        self.active, self.last_event, self.last_raise = 0, 0.0, time.time()
        self.cv = threading.Condition()

    def acquire(self) -> None:
        with self.cv:
            while True:
                now = time.time()
                if now - self.last_raise > self.every and now - self.last_event > self.every and self.limit < self.cap:
                    self.limit, self.last_raise = min(self.cap, self.limit + self.step), now
                if self.active < self.limit:
                    self.active += 1
                    return
                self.cv.wait(timeout=1.0)

    def release(self, throttled: bool = False) -> None:
        with self.cv:
            self.active -= 1
            if throttled:
                if time.time() - self.last_event > 5.0:
                    self.limit = max(self.floor, int(self.limit * 0.7))
                self.last_event = time.time()
            self.cv.notify_all()


def create(cl, thr: Throttle, params: dict):
    """One Messages call under the throttle; rate limits, overloads, server errors and dropped connections back off and retry."""
    import anthropic  # noqa: PLC0415
    for k in range(12):
        thr.acquire()
        throttled = False
        try:
            return cl.messages.create(**params)
        except anthropic.APIStatusError as e:
            if e.status_code not in (429, 500, 502, 503, 504, 529) or k == 11:
                raise
            throttled = True
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            if k == 11:
                raise
            throttled = True
        finally:
            thr.release(throttled)
        time.sleep(min(90.0, 3.0 * 1.6 ** k) * (0.5 + random.random()))
    raise RuntimeError("unreachable")


def text_of(message) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")


def strict_json(text: str):
    """The JSON object between the first '{' and the last '}' of a reply, or None."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        out = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def judge(cl, thr: Throttle, system: str, user: str, validate, model: str = HAIKU, attempts: int = 4, max_tokens: int = 2000,
          effort: str | None = "low"):
    """Ask until validate(strict_json(reply)) is not None, up to `attempts` times; returns (fields, message). A refusal raises."""
    params = {"model": model, "max_tokens": max_tokens,
              "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
              "messages": [{"role": "user", "content": user}]}
    if effort:
        params["output_config"] = {"effort": effort}
    for _ in range(attempts):
        m = create(cl, thr, params)
        if m.stop_reason == "refusal":
            raise RuntimeError("refusal")
        out = strict_json(text_of(m))
        got = validate(out) if out is not None else None
        if got is not None:
            return got, m
    raise ValueError(f"no valid JSON in {attempts} attempts")
