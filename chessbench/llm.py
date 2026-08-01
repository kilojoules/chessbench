from __future__ import annotations

import random
import time
from dataclasses import dataclass

import chess


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    reasoning: str | None = None  # separated thinking text, when the provider returns it


def _probe_effective_temperature(model: str, temperature: float) -> float | None:
    """None if the provider rejects the requested temperature (and would fall
    back to its own default under drop_params) — recorded so no model silently
    runs under a different sampling regime than the records claim."""
    import litellm

    try:
        provider = litellm.get_llm_provider(model)[1]
        litellm.get_optional_params(
            model=model,
            custom_llm_provider=provider,
            temperature=temperature,
            drop_params=False,
        )
        return temperature
    except Exception as e:
        if type(e).__name__ == "UnsupportedParamsError":
            return None
        return temperature  # probe machinery failed; assume supported


class LiteLLMClient:
    """Thin wrapper over litellm for provider-agnostic completions.

    The `board` argument exists only for FakeLLM's benefit and is ignored
    here — the real model must never receive board state outside the prompt.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 600.0,  # local thinking models can take minutes/move
        num_retries: int = 2,
        seed: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_retries = num_retries
        self.seed = seed
        self.effective_temperature = _probe_effective_temperature(model, temperature)

    def complete(self, messages: list[dict], board: chess.Board | None = None) -> LLMResponse:
        import litellm

        litellm.drop_params = True  # safety net for anything the probe missed
        litellm.suppress_debug_info = True
        kwargs: dict = {}
        if self.effective_temperature is not None:
            kwargs["temperature"] = self.effective_temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        t0 = time.monotonic()
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            num_retries=self.num_retries,
            **kwargs,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            reasoning=getattr(choice.message, "reasoning_content", None),
        )


def _illegal_san(board: chess.Board) -> str:
    """A syntactically valid SAN move that is illegal in this position."""
    for sq in chess.SQUARE_NAMES:
        cand = f"K{sq}"
        try:
            board.parse_san(cand)
        except chess.IllegalMoveError:
            return cand
        except (chess.InvalidMoveError, chess.AmbiguousMoveError):
            continue
    return "Ka1"  # unreachable in practice: a king move is almost always illegal somewhere


class FakeLLM:
    """Deterministic offline stand-in for dry runs and tests.

    Policies:
      "first"          — first legal move in sorted-SAN order
      "random"         — seeded-random legal move
      "always-illegal" — a well-formed illegal move every time (loss path)

    `illegal_at` — set of llm-move indices (1-based, fresh requests only) at
    which the first attempt is a well-formed illegal move; the retry then
    plays legally, exercising the recovery path.

    `script` — explicit list of raw outputs, consumed in order (tests).

    Relies on the harness passing the current board via `complete(..., board=)`;
    the real client ignores that argument.
    """

    def __init__(
        self,
        policy: str = "first",
        seed: int = 0,
        illegal_at: frozenset[int] | set[int] = frozenset(),
        script: list[str] | None = None,
    ):
        if policy not in ("first", "random", "always-illegal"):
            raise ValueError(f"unknown FakeLLM policy: {policy}")
        self.policy = policy
        self.rng = random.Random(seed)
        self.illegal_at = set(illegal_at)
        self.script = list(script) if script is not None else None
        self.move_index = 0  # counts fresh (non-retry) requests
        self.effective_temperature = 0.0  # deterministic; mirrors the real client's attribute

    def _legal_san(self, board: chess.Board) -> str:
        sans = sorted(board.san(m) for m in board.legal_moves)
        if self.policy == "random":
            return self.rng.choice(sans)
        return sans[0]

    def complete(self, messages: list[dict], board: chess.Board | None = None) -> LLMResponse:
        if self.script is not None:
            if not self.script:
                raise RuntimeError("FakeLLM script exhausted")
            return LLMResponse(self.script.pop(0), 0, 0, 0, finish_reason="stop")
        if board is None:
            raise RuntimeError("FakeLLM requires the harness to pass board=")
        is_retry = len(messages) > 2  # fresh request is [system, user]
        if not is_retry:
            self.move_index += 1
        if self.policy == "always-illegal" or (not is_retry and self.move_index in self.illegal_at):
            san = _illegal_san(board)
        else:
            san = self._legal_san(board)
        return LLMResponse(f"Considering the position.\nMOVE: {san}", 0, 0, 0, finish_reason="stop")
