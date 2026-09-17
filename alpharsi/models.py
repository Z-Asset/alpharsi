"""OpenAI-compatible chat client，仅供 Refiner 在线模式调用 DeepSeek 等。

离线模式（主推路径）用 TPE 贝叶斯搜索，不需要此模块。仅在线 `--online`
时，Refiner 才用 ChatModel 读历史指标、提议下一组训练配方。

只用标准库 urllib，无第三方运行时依赖。
"""
from __future__ import annotations

import json
from typing import Any


class ChatModel:
    def __init__(self, name: str = "", *, base_url: str = "", api_key: str = "", temperature: float = 0.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """Send a chat request and return the assistant text.

        Synchronous and blocking. Raises RuntimeError with the URL/status so a
        reachability failure is diagnosable rather than silently swallowed.
        """
        import urllib.error
        import urllib.request

        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"model {self.name!r} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach {self.base_url}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response shape from {self.base_url}: {json.dumps(body)[:500]}") from exc
