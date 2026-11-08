"""LLM access through openJiuwen's `Model` (OpenAI-compatible client).

All calls are charged to the Ledger. The SDK client has no retries, so we
retry transient failures ourselves (a failed request costs nothing).
"""
import asyncio
import random
import re
import time

from . import config
from .budget import approx_tokens
from .oj import Model, ModelClientConfig, ModelRequestConfig

_THINK = re.compile(r"<think>.*?</think>", re.S)


def client_config():
    return ModelClientConfig(
        client_provider="OpenAI",
        api_key=config.LLM_API_KEY or "EMPTY",
        api_base=config.LLM_BASE_URL,
        timeout=float(config.LLM_TIMEOUT_S),
        verify_ssl=True,
    )


def usage_of(resp):
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return 0, 0, 0
    return (getattr(um, "input_tokens", 0) or 0, getattr(um, "output_tokens", 0) or 0,
            getattr(um, "reasoning_tokens", 0) or 0)


class LLMUnavailable(RuntimeError):
    """The proxy refuses service (budget exhausted / persistent outage)."""


class LLM:
    def __init__(self, ledger):
        self.ledger = ledger
        self._models = {}
        self.dead = False
        self.consecutive_failures = 0

    def _model(self, name):
        if name not in self._models:
            self._models[name] = Model(client_config(), ModelRequestConfig(model=name, temperature=0.0))
        return self._models[name]

    async def chat(self, task_id, messages, model=None, max_tokens=800, temperature=0.0, tag=""):
        """Returns (text, cost). Raises LLMUnavailable / RuntimeError on failure."""
        if self.dead:
            raise LLMUnavailable("LLM disabled after repeated failures")
        model = model or config.MODEL_FAST
        extra = {}
        if model.endswith("-noreason"):
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
        last, failures, attempt = None, 0, 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                resp = await asyncio.wait_for(
                    self._model(model).invoke(messages, temperature=temperature, max_tokens=max_tokens,
                                              extra_body=extra or None),
                    timeout=config.LLM_TIMEOUT_S + 10)
            except asyncio.TimeoutError:
                # the proxy may still bill a request we gave up on: charge a pessimistic estimate
                pin = sum(approx_tokens(str(m.get("content", ""))) for m in messages)
                self.ledger.charge(task_id, model, pin, max_tokens, tag=f"{tag} TIMEOUT")
                last, failures = "timeout", failures + 1
            except Exception as e:  # noqa: BLE001 - network / proxy errors
                msg = str(e)
                last = e
                low = msg.lower()
                if "429" in msg or "ratelimit" in low or "too many" in low or "одновременн" in low:
                    if attempt <= 12:  # concurrency limit of the proxy: wait and retry, costs nothing
                        await asyncio.sleep(min(10.0, 1.0 + attempt) + random.random())
                        continue
                if any(k in low for k in ("budget", "insufficient", "quota", "exceeded", "402")):
                    self.dead = True
                    raise LLMUnavailable(msg[:300])
                failures += 1
            else:
                self.consecutive_failures = 0
                pin, pout, preason = usage_of(resp)
                text = resp.content if isinstance(resp.content, str) else str(resp.content or "")
                if not pin and not pout:  # provider omitted usage -> estimate conservatively
                    pin = sum(approx_tokens(str(m.get("content", ""))) for m in messages)
                    pout = approx_tokens(text + str(getattr(resp, "reasoning_content", "") or "")) + max_tokens // 4
                included = not (preason and preason > pout)
                cost = self.ledger.charge(task_id, model, pin, pout, preason, included,
                                          tag=f"{tag} {time.time() - t0:.1f}s")
                return _THINK.sub("", text).strip(), cost
            self.consecutive_failures += 1
            if self.consecutive_failures >= 8:
                self.dead = True  # proxy stopped answering (budget exhausted or outage)
                raise LLMUnavailable(f"{self.consecutive_failures} consecutive failures: {last}")
            if failures > config.REQUEST_RETRIES:
                raise RuntimeError(f"LLM call failed: {last}")
            await asyncio.sleep(2.0 * failures)
