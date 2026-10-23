"""The language-model client. Plain ``groq`` SDK, wrapped so that

* the agent only depends on a tiny interface (``complete``), which tests fake
* every provider error becomes a short, safe ``LLMError`` message (never the raw
  exception text, never the API key)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import groq
from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMError(Exception):
    """Raised with a message that is safe to show to the user.

    ``kind`` lets callers react without parsing text: rate_limit, timeout,
    connection, server, auth, rejected or other. ``retry_after`` is the wait (in
    seconds) Groq asked for on a rate limit, when it said.
    """

    def __init__(self, message: str, kind: str = "other", retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


def _retry_after(error: "groq.APIStatusError") -> float | None:
    try:
        return float(error.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


class LLM(Protocol):
    def complete(self, system: str, user: str, *, step: str = "", max_tokens: int = 1500) -> str: ...


class GroqLLM:
    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, timeout_s: float = 30.0, max_retries: int = 1
    ):
        self.model = model
        # The SDK retries transient 429/5xx/connection errors itself. The
        # evaluation harness sets 0 so that it can time and report those waits.
        self._client = groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=max_retries)

    @classmethod
    def from_env(cls, env_file: Path | None = None, max_retries: int = 1) -> "GroqLLM":
        load_dotenv(env_file or PROJECT_ROOT / ".env")
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key or api_key.startswith("your_"):
            raise LLMError("GROQ_API_KEY is not set. Add it to the .env file (see .env.example).")
        model = os.environ.get("GROQ_MODEL", "").strip()
        if not model or model.startswith("your_"):
            model = DEFAULT_MODEL
        return cls(api_key, model, max_retries=max_retries)

    def complete(self, system: str, user: str, *, step: str = "", max_tokens: int = 1500) -> str:
        extra = {"reasoning_effort": "low"} if self.model.startswith("openai/gpt-oss") else {}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        except groq.AuthenticationError:
            raise LLMError("The Groq API key was rejected. Check GROQ_API_KEY in .env.", "auth") from None
        except groq.PermissionDeniedError:
            raise LLMError("The Groq API key is not allowed to use this model.", "auth") from None
        except groq.RateLimitError as error:
            raise LLMError(
                "Groq's rate limit was reached. Please wait a little and try again.",
                "rate_limit",
                _retry_after(error),
            ) from None
        except groq.APITimeoutError:
            raise LLMError("The language model took too long to respond. Please try again.", "timeout") from None
        except groq.APIConnectionError:
            raise LLMError("Could not reach the Groq API. Check the internet connection.", "connection") from None
        except groq.InternalServerError:
            raise LLMError("Groq is temporarily unavailable. Please try again later.", "server") from None
        except groq.APIStatusError as error:
            raise LLMError(
                f"The language model request was rejected (HTTP {error.status_code}).", "rejected"
            ) from None
        except Exception:
            raise LLMError("The language model call failed unexpectedly.") from None
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("The language model returned an empty response.")
        return text
