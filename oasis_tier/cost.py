"""Token metering and budget enforcement for Tier-2 runs.

LLM simulations fail by silently burning money, not by crashing. Nothing in
this package calls a model without going through a metered backend.

The meter wraps CAMEL's model backend and reads `usage` off each response, so
the numbers are what the provider actually billed rather than an estimate from
a local tokenizer.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

# USD per million tokens. Keep in sync with the pricing table when models change.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class BudgetExceeded(RuntimeError):
    """Raised when a run crosses its configured spend ceiling."""


@dataclass
class CostMeter:
    """Thread-safe accumulator for token usage and spend."""

    model: str
    budget_usd: float = 1.00
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def rates(self) -> tuple[float, float]:
        return PRICING.get(self.model, (0.0, 0.0))

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = self.rates
        return (
            self.input_tokens * rate_in + self.output_tokens * rate_out
        ) / 1_000_000

    def record(self, usage) -> None:
        """Accumulate one response's usage. Accepts an object or a dict."""
        if usage is None:
            return

        def _get(name, *alts):
            for n in (name, *alts):
                if isinstance(usage, dict):
                    if usage.get(n) is not None:
                        return usage[n]
                elif getattr(usage, n, None) is not None:
                    return getattr(usage, n)
            return 0

        with self._lock:
            self.calls += 1
            self.input_tokens += int(_get("input_tokens", "prompt_tokens") or 0)
            self.output_tokens += int(_get("output_tokens", "completion_tokens") or 0)
            self.cache_read_tokens += int(_get("cache_read_input_tokens") or 0)

    def check_budget(self) -> None:
        if self.cost_usd > self.budget_usd:
            raise BudgetExceeded(
                f"run spent ${self.cost_usd:.4f}, over the ${self.budget_usd:.2f} cap "
                f"after {self.calls} calls"
            )

    def snapshot(self) -> dict:
        rate_in, rate_out = self.rates
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "budget_usd": self.budget_usd,
            "rate_in_per_mtok": rate_in,
            "rate_out_per_mtok": rate_out,
            "priced": self.model in PRICING,
        }


def metered_backend(backend, meter: CostMeter):
    """Patch a CAMEL model backend in place so every call is recorded.

    Returns the *same* object, with `run`/`arun` shadowed by metering wrappers.

    A wrapper object will not work here: CAMEL's `ChatAgent._resolve_models`
    does an `isinstance(model, BaseModelBackend)` check and rejects anything
    else, so the real backend has to be the thing that gets handed to the agent.
    Shadowing the bound methods on the instance keeps the type intact while
    still capturing usage.
    """

    def _capture(result):
        usage = getattr(result, "usage", None)
        if usage is not None:
            meter.record(usage)
            meter.check_budget()
        return result

    original_run = backend.run
    original_arun = getattr(backend, "arun", None)

    def run(*args, **kwargs):
        return _capture(original_run(*args, **kwargs))

    backend.run = run

    if original_arun is not None:

        async def arun(*args, **kwargs):
            return _capture(await original_arun(*args, **kwargs))

        backend.arun = arun

    return backend


def extrapolate(meter: CostMeter, agent_steps_done: int, target_agent_steps: int) -> dict:
    """Project a measured toy run up to a target scale.

    An agent-step is one agent being asked to act once. This is the unit that
    matters for planning: cost scales with it, not with wall-clock or rounds.
    """
    if agent_steps_done <= 0:
        return {"error": "no agent-steps recorded"}

    per_step = meter.cost_usd / agent_steps_done
    return {
        "measured_agent_steps": agent_steps_done,
        "measured_cost_usd": round(meter.cost_usd, 6),
        "cost_per_agent_step_usd": round(per_step, 8),
        "target_agent_steps": target_agent_steps,
        "projected_cost_usd": round(per_step * target_agent_steps, 4),
    }
