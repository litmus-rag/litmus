"""Small client for an Azure OpenAI resource.

Three entry points:
  chat(prompt)                    -> plain completion, Chat Completions API
  chat_thinking(prompt, effort)   -> reasoning mode, Responses API
  chat_search(prompt)             -> web search, Responses API
"""

import os
import requests

DEPLOYMENT = "gpt-5.4"
CONF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


def _load_conf(path=CONF_PATH):
    conf = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            conf[key.strip()] = value.strip()
    return conf


_conf = _load_conf()
API_KEY = _conf["AZURE_API_KEY"]
API_BASE = _conf["AZURE_API_BASE"].rstrip("/")
API_VERSION = _conf["AZURE_API_VERSION"]

_HEADERS = {"Content-Type": "application/json", "api-key": API_KEY}


def chat(prompt, max_completion_tokens=500):
    """Plain completion via the Chat Completions API. No reasoning, no tools."""
    url = f"{API_BASE}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    resp = requests.post(url, headers=_HEADERS, json=body)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_thinking(prompt, effort="high", max_output_tokens=500):
    """Reasoning-mode completion via the Responses API.

    effort: "low" | "medium" | "high"
    """
    url = f"{API_BASE}/openai/responses?api-version={API_VERSION}"
    body = {
        "model": DEPLOYMENT,
        "input": prompt,
        "reasoning": {"effort": effort},
        "max_output_tokens": max_output_tokens,
    }
    resp = requests.post(url, headers=_HEADERS, json=body)
    resp.raise_for_status()
    return _extract_text(resp.json())


def chat_search(prompt, max_output_tokens=500):
    """Web-search-grounded completion via the Responses API.

    Returns (text, citations) where citations is a list of {title, url}.
    """
    url = f"{API_BASE}/openai/responses?api-version={API_VERSION}"
    body = {
        "model": DEPLOYMENT,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "max_output_tokens": max_output_tokens,
    }
    resp = requests.post(url, headers=_HEADERS, json=body)
    resp.raise_for_status()
    data = resp.json()
    text = _extract_text(data)
    citations = _extract_citations(data)
    return text, citations


def _extract_text(data):
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    return ""


def _extract_citations(data):
    citations = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                for ann in part.get("annotations", []):
                    if ann.get("type") == "url_citation":
                        citations.append({"title": ann.get("title"), "url": ann.get("url")})
    return citations


if __name__ == "__main__":
    print("chat():", chat("Say hello in one word."))
    print("chat_thinking():", chat_thinking("What is 17*24? Think it through.", effort="high"))
    text, cites = chat_search("Who wrote Attention Is All You Need and from which company?")
    print("chat_search() text:", text)
    print("chat_search() citations:", cites)
