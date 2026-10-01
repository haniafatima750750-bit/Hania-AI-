"""
Hania AI - a professional AI chatbot desktop app (pure Python, single file).

Run:
    python hania_ai.py          # desktop app
    python hania_ai.py --cli    # terminal chat

Multiple users: every person signs up with a username + password (stored as a salted PBKDF2 hash) and gets their
own private chats and settings (including their own API key). Data lives in the "hania_ai_data" folder next to
this file (override with HANIA_DATA_DIR).

Real AI (optional): click the Settings button (gear icon) in the app, choose Gemini / ChatGPT / Ollama, paste your
key and press Save. The app can install the 'openai' package for you. Or use `.streamlit/secrets.toml` with
`GEMINI_API_KEY`, and optionally `CUSTOM_SERVER_URL` / `CUSTOM_MODEL_NAME`, or use environment variables:
    OPENAI_API_KEY=sk-...                     (optional OPENAI_MODEL, default gpt-4o-mini)
    GEMINI_API_KEY=...                        (optional GEMINI_MODEL, default gemini-2.0-flash)
    OPENAI_API_KEY=ollama + OPENAI_BASE_URL=http://localhost:11434/v1 + OPENAI_MODEL=llama3.2   (free, local)
Without a key, the built-in offline brain is used, so the app works with zero setup.

Design: follows style.css - deep-blue background with a cyan top-left / black bottom-right glow,
glowing #AED6F1 heading, dark #161b22 chat bubbles with cream text, black pill input with a blue send button,
and a #0b0f14 sidebar.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import getpass
import hashlib
import hmac
import importlib
import itertools
import json
import math
import operator
import os
import queue
import random
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

try:
    import tomllib
except ImportError:
    tomllib = None

if TYPE_CHECKING:
    from tkinter import Canvas, Entry, Event, Label, Misc, PhotoImage, StringVar, Text, Tk, Widget

try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox, simpledialog
except ImportError:
    tk = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

APP_NAME = "Hania AI"
WELCOME_TEXT = "Welcome! Let's solve your query together"
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    f"You are {APP_NAME}, a friendly, expert AI assistant. Give clear, accurate, well-structured answers. "
    "Use Markdown (headings, bullet lists, **bold**, `code`, fenced code blocks) when it helps.",
)
DATA_DIR = Path(os.getenv("HANIA_DATA_DIR", Path(__file__).with_name("hania_ai_data")))
LEGACY_FILE = Path(__file__).with_name("hania_ai_chats.json")  # single-user chat file from older versions
USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]{3,32}")
MIN_PASSWORD = 6
PBKDF2_ROUNDS = 200_000
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_FALLBACK_MODELS = ("gemini-3.8-flash",)


@dataclass(frozen=True)
class Provider:
    label: str
    model: str = ""
    base_url: str = ""
    key_url: str = ""
    needs_key: bool = True
    note: str = ""


PROVIDERS = {
    "offline": Provider("Offline", needs_key=False, note="Uses the built-in offline brain. No internet or key needed."),
    "gemini": Provider("Gemini", "gemini-2.0-flash", GEMINI_BASE_URL, "https://aistudio.google.com/apikey",
                       note="Google Gemini. Free API keys are available from Google AI Studio."),
    "openai": Provider("ChatGPT", "gpt-4o-mini", "", "https://platform.openai.com/api-keys",
                       note="OpenAI ChatGPT models. Needs an OpenAI API key with credit."),
    "ollama": Provider("Ollama", "llama3.2", "http://localhost:11434/v1", "https://ollama.com/download",
                       needs_key=False, note="Free AI running on your own computer. Install Ollama first."),
    "custom": Provider("Custom", needs_key=True,
                       note="Any OpenAI-compatible API (Groq, OpenRouter, LM Studio...)."),
}


def load_file_provider_settings() -> dict[str, str]:
    """Read provider credentials from the app's local Streamlit-style secrets file."""
    if tomllib is None:
        print("Reading .streamlit/secrets.toml requires Python 3.11 or newer.")
        return {}
    path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not path.is_file():
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        print("Could not read .streamlit/secrets.toml. Check that the file is valid TOML.")
        return {}

    settings = {key: value.strip() for key, value in raw.items()
                if isinstance(value, str) and value.strip()}
    if settings.get("GEMINI_API_KEY"):
        return {
            "provider": "gemini",
            "api_key": settings["GEMINI_API_KEY"],
            "model": settings.get("CUSTOM_MODEL_NAME", settings.get("GEMINI_MODEL", PROVIDERS["gemini"].model)),
            "base_url": settings.get("CUSTOM_SERVER_URL", GEMINI_BASE_URL),
        }
    if settings.get("OPENAI_API_KEY"):
        return {
            "provider": "openai",
            "api_key": settings["OPENAI_API_KEY"],
            "model": settings.get("OPENAI_MODEL", PROVIDERS["openai"].model),
            "base_url": settings.get("OPENAI_BASE_URL", ""),
        }
    return {}


# Theme taken from the user's style.css
BG = "#1A5276"            # --bg
GLOW = "#AED6F1"          # --glow: headings
CREAM = "#fdfce8"         # chat, subtitle and sidebar text
BUBBLE = "#161b22"        # chat message background
BORDER = "#30363d"        # bubble / button borders
HOVER = "#242c36"         # button hover
SIDEBAR = "#0b0f14"
INPUT_BG = "#000000"      # chat input capsule
INPUT_BORDER = "#1f2937"
ACCENT = "#2980B9"        # send button + focus glow
SEND_BG = "#081a25"       # rgba(41, 128, 185, 0.2) on black
THUMB = "#010101"         # scrollbar thumb
CODE_BG = "#0d1117"
CODE_FG = "#e6edf3"
MUTED = "#8b949e"
ERROR = "#f87171"
SHADOW = ("#2a1812", "#152526")  # heading pulse text-shadow colours
# background-image radial glows, top-most first: ((x, y) centre as fraction of the view, colour stops)
GLOW_LAYERS = (
    ((0.0, 0.0), ((0.0, (0, 242, 254, 0.25)), (0.30, (41, 128, 185, 0.05)), (0.60, (0, 0, 0, 0.0)))),
    ((1.0, 1.0), ((0.0, (0, 0, 0, 0.95)), (0.40, (10, 15, 26, 0.6)), (0.70, (0, 0, 0, 0.0)))),
)


# =========================================================== storage

def write_json(path: Path, data: dict) -> None:
    """Write atomically so a crash (or two app windows saving at once) never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def hash_password(password: str, salt: bytes | None = None, rounds: int = PBKDF2_ROUNDS) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _algo, rounds, salt, _digest = stored.split("$")
        return hmac.compare_digest(hash_password(password, bytes.fromhex(salt), int(rounds)), stored)
    except ValueError:
        return False


class ChatStore:
    """Conversations + settings persisted in one JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"settings": {}, "conversations": []}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        self.data.setdefault("settings", {})
        self.data.setdefault("conversations", [])

    @property
    def conversations(self) -> list[dict]:
        return self.data["conversations"]

    def save(self) -> None:
        write_json(self.path, self.data)

    def get_setting(self, key: str, default: str) -> str:
        return self.data["settings"].get(key, default)

    def set_setting(self, key: str, value: str) -> None:
        self.data["settings"][key] = value
        self.save()

    def new_conversation(self, title: str) -> dict:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        conv = {"id": uuid.uuid4().hex, "title": title, "created": now, "updated": now, "messages": []}
        self.conversations.insert(0, conv)
        self.save()
        return conv

    def get(self, conv_id: str | None) -> dict | None:
        return next((c for c in self.conversations if c["id"] == conv_id), None)

    def add_message(self, conv: dict, role: str, content: str) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        conv["messages"].append({"role": role, "content": content, "time": now})
        conv["updated"] = now
        self.conversations.remove(conv)
        self.conversations.insert(0, conv)
        self.save()

    def rename(self, conv: dict, title: str) -> None:
        conv["title"] = title
        self.save()

    def delete(self, conv: dict) -> None:
        self.conversations.remove(conv)
        self.save()


class UserStore:
    """User accounts (accounts.json) + one private chat file per user (chats/<username>.json)."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / "accounts.json"
        self.data = self._load()

    def _load(self) -> dict:
        data: dict = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        data.setdefault("users", {})
        data.setdefault("remembered", "")
        data.setdefault("last_user", "")
        return data

    def _save(self) -> None:
        write_json(self.path, self.data)

    @staticmethod
    def key(username: str) -> str:
        return username.strip().lower()

    def get(self, username: str) -> dict | None:
        self.data = self._load()
        return self.data["users"].get(self.key(username))

    def display_name(self, username: str) -> str:
        user = self.get(username)
        return user["name"] if user else username

    def chat_store(self, username: str) -> ChatStore:
        return ChatStore(self.folder / "chats" / f"{self.key(username)}.json")

    def create(self, username: str, password: str, confirm: str | None = None) -> str:
        """Create an account and return its key. Raises ValueError with a message for the user."""
        username = username.strip()
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("Username must be 3-32 characters: letters, numbers, dot, dash or underscore.")
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
        if confirm is not None and password != confirm:
            raise ValueError("The two passwords don't match.")
        if self.get(username):
            raise ValueError("That username is already taken.")
        first_user = not self.data["users"]
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        self.data["users"][self.key(username)] = {"name": username, "password": hash_password(password),
                                                  "created": now}
        self._save()
        if first_user and LEGACY_FILE.exists():
            self._import_legacy(username)
        return self.key(username)

    def _import_legacy(self, username: str) -> None:
        """Give the first account the chats saved by the old single-user version."""
        try:
            old = ChatStore(LEGACY_FILE)
            store = self.chat_store(username)
            store.data["conversations"] = old.conversations + store.conversations
            store.data["settings"] = {**old.data["settings"], **store.data["settings"]}
            store.save()
            LEGACY_FILE.rename(LEGACY_FILE.with_name(LEGACY_FILE.name + ".imported"))
        except OSError:
            pass

    def verify(self, username: str, password: str) -> bool:
        user = self.get(username)
        if user is None:
            hash_password(password)  # same work either way, so timing doesn't reveal which usernames exist
            return False
        return check_password(password, user["password"])

    def change_password(self, username: str, old: str, new: str, confirm: str) -> None:
        if not self.verify(username, old):
            raise ValueError("Your current password is wrong.")
        if len(new) < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
        if new != confirm:
            raise ValueError("The two passwords don't match.")
        self.data["users"][self.key(username)]["password"] = hash_password(new)
        self._save()

    def delete(self, username: str, password: str) -> None:
        if not self.verify(username, password):
            raise ValueError("Wrong password - account not deleted.")
        key = self.key(username)
        del self.data["users"][key]
        if self.data["remembered"] == key:
            self.data["remembered"] = ""
        if self.data["last_user"] == key:
            self.data["last_user"] = ""
        self._save()
        (self.folder / "chats" / f"{key}.json").unlink(missing_ok=True)

    def signed_in(self, username: str, remember: bool) -> None:
        self.data = self._load()
        self.data["last_user"] = self.key(username)
        self.data["remembered"] = self.key(username) if remember else ""
        self._save()

    def remembered_user(self) -> str:
        self.data = self._load()
        key = self.data["remembered"]
        return key if key in self.data["users"] else ""

    def forget(self) -> None:
        self.data = self._load()
        self.data["remembered"] = ""
        self._save()


# =========================================================== AI brains

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """Evaluate a math expression without eval()."""
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")
    return _eval(ast.parse(expr, mode="eval"))


class LocalBrain:
    """Offline brain used when no API key is configured."""

    name = "Offline mode"
    JOKES = (
        "Why do programmers prefer dark mode?\n\nBecause **light attracts bugs**.",
        "There are 10 kinds of people in the world: those who understand binary and those who don't.",
        "Why did the Python developer wear glasses?\n\nBecause they couldn't **C**.",
        "A SQL query walks into a bar, walks up to two tables and asks: *\"Can I join you?\"*",
    )
    NOT_NAMES: ClassVar[set[str]] = {"fine", "good", "ok", "okay", "great", "not", "sad", "happy", "here", "a", "the"}

    def _user_name(self, messages: list[dict]) -> str | None:
        name = None
        for m in messages:
            if m["role"] == "user":
                found = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Za-z]+)", m["content"], re.IGNORECASE)
                if found and found.group(1).lower() not in self.NOT_NAMES:
                    name = found.group(1).capitalize()
        return name

    def reply(self, messages: list[dict]) -> str:
        low = messages[-1]["content"].strip().lower()
        user = self._user_name(messages)

        math = re.search(r"([\d.(][\d\s.+\-*/%()^]*[\d)])", low)
        if math and re.search(r"\d\s*[+\-*/%^]\s*[\d(]", math.group(1)):
            expr = math.group(1)
            try:
                result = safe_eval(expr.replace("^", "**"))
                if isinstance(result, float):
                    result = int(result) if result.is_integer() else round(result, 6)
                return f"Here's the result:\n\n`{expr.strip()} = {result}`"
            except (ValueError, ZeroDivisionError, SyntaxError):
                return "I couldn't calculate that. Try something like `12 * (3 + 4)`."

        now = datetime.now().astimezone()
        rules = [
            (r"\b(my name is|call me)\b", f"Nice to meet you, **{user}**! How can I help you today?"),
            (r"\b(what'?s|what is) my name\b",
             f"Your name is **{user}**." if user else "You haven't told me your name yet."),
            (r"^(hi|hello|hey|salam|assalam|aoa|good (morning|afternoon|evening))\b",
             f"Hello{', ' + user if user else ''}! I'm **{APP_NAME}**. How can I help you today?"),
            (r"\bhow are you\b", "I'm doing great and ready to help! How are you?"),
            (r"\b(your name|who are you|what are you)\b",
             (f"I'm **{APP_NAME}**, an AI assistant built with Python. I can chat, answer questions, "
             "do math, and more.")),
            (r"\bstory\b",
             ("## The Robot Who Learned to Love\n\n"
             "In a quiet workshop lived **Bolt**, a small robot built to sort screws. Every day, an old "
             "inventor named Mira hummed songs while she worked.\n\n"
             "One evening Mira fell ill. Bolt had no instructions for this, yet it brought her tea, "
             "played her favourite song, and stayed by her side all night.\n\n"
             "When she woke, Mira smiled: *\"You weren't programmed for that.\"*\n\n"
             "Bolt's lights flickered softly. **\"No,\"** it said. **\"I learned it from you.\"**")),
            (r"\b(chatbots?|ai) work|how does ai\b",
             ("## How AI chatbots work\n\n"
             "1. **Training** - a large language model reads huge amounts of text and learns patterns.\n"
             "2. **Your prompt** - your message (plus the chat history) is converted into tokens.\n"
             "3. **Prediction** - the model predicts the most likely next token, again and again.\n"
             "4. **Streaming** - those tokens are sent back to you word by word.\n\n"
             "Fine-tuning and human feedback make the answers helpful and safe.")),
            (r"\btime\b", f"It's **{now:%H:%M}** right now."),
            (r"\b(date|today)\b", f"Today is **{now:%A, %B %d, %Y}**."),
            (r"\bjoke\b", random.choice(self.JOKES)),
            (r"\b(thank|thanks|thx)\b", "You're welcome! Anything else I can help with?"),
            (r"\b(bye|goodbye|see you)\b", "Goodbye! Come back anytime."),
            (r"\b(help|what can you do)\b",
             ("Here's what I can do in **offline mode**:\n\n"
             "- Chat and remember your name\n- Solve math like `2^10 / 4`\n- Tell the time and date\n"
             "- Tell jokes and short stories\n\n"
             "Set `OPENAI_API_KEY` or `GEMINI_API_KEY` to unlock full AI answers on any topic.")),
            (r"\bpython\b",
             ("**Python** is a popular, easy-to-read programming language. Example:\n\n"
             "```\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nprint(greet(\"Hania\"))\n```")),
            (r"\b(sad|depressed|unhappy|lonely)\b",
             ("I'm sorry you're feeling that way. Talking to someone you trust can really help - "
             "and I'm always here to listen.")),
            (r"\b(happy|great|good|awesome|fine)\b", "That's great to hear! What would you like to talk about?"),
        ]
        for pattern, answer in rules:
            if re.search(pattern, low):
                return answer
        if low.endswith("?"):
            return ("That's a great question! I'm running in **offline mode**, so my knowledge is limited.\n\n"
                    "Add an `OPENAI_API_KEY` or `GEMINI_API_KEY` to get full answers on any topic.")
        return random.choice([
            "Interesting! Tell me more.",
            "I see. Could you explain a bit more?",
            "Got it. What else is on your mind?",
            "I'm not sure I follow - type **help** to see what I can do.",
        ])

    def stream(self, messages: list[dict]) -> Iterator[str]:
        yield from re.findall(r"[^\S\n]*\S+|\n", self.reply(messages))


def is_model_not_found(exc: Exception) -> bool:
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    model_unavailable = any(phrase in text for phrase in (
        "not found",
        "does not exist",
        "not supported",
        "no longer available",
    ))
    return (
        "model" in text
        and model_unavailable
        and (status_code == 404 or "404" in text)
    )


class OpenAIBrain:
    """Any OpenAI-compatible chat API: OpenAI, Gemini, Groq, Ollama, LM Studio..."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None,
        label: str = "",
        fallback_models: tuple[str, ...] = (),
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=60)
        self.model = model
        self.name = f"{label} \u00b7 {model}" if label else model
        self.fallback_models = tuple(candidate for candidate in fallback_models if candidate != model)

    def ping(self) -> str:
        reply = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": "Reply with just: Connected"}])
        return (reply.choices[0].message.content or "").strip() if reply.choices else ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = [{"role": "system", "content": SYSTEM_PROMPT}]
        payload += [{"role": m["role"], "content": m["content"]} for m in messages[-30:]]
        models = (self.model, *self.fallback_models)
        for index, model in enumerate(models):
            received_content = False
            try:
                chunks = self.client.chat.completions.create(model=model, messages=payload, stream=True)
                for chunk in chunks:
                    # Accept both object-like and dict-like OpenAI-compatible response chunks.
                    try:
                        choices = getattr(chunk, "choices", None) or chunk.get("choices")
                    except (AttributeError, TypeError):
                        continue
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None and isinstance(choice, dict):
                        delta = choice.get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                    if content:
                        received_content = True
                        yield content
                return
            except Exception as exc:
                if received_content or not is_model_not_found(exc) or index == len(models) - 1:
                    if is_model_not_found(exc) and len(models) > 1 and index == len(models) - 1:
                        tried = ", ".join(f"`{candidate}`" for candidate in models)
                        raise RuntimeError(
                            f"404: Gemini rejected all tried model IDs ({tried}). Provider response: {exc}"
                        ) from exc
                    raise


def provider_error_detail(exc: Exception) -> str:
    """Return a bounded provider message without including request headers or credentials."""
    current: BaseException | None = exc
    while current is not None:
        candidates = [getattr(current, "body", None)]
        response = getattr(current, "response", None)
        if response is not None:
            candidates.append(getattr(response, "text", None))
        for body in candidates:
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    continue
            if isinstance(body, dict):
                error = body.get("error", body)
                if isinstance(error, dict):
                    detail = error.get("message") or error.get("detail")
                    if isinstance(detail, str):
                        return detail[:500]
        current = current.__cause__
    return ""


def friendly_error(exc: Exception) -> str:
    """Turn API/network exceptions into a short message a non-programmer can act on."""
    text = str(exc)
    low = text.lower()
    if "401" in text or "api key" in low or "api_key" in low or "unauthorized" in low:
        return "Your API key was rejected. Open Settings and check that it is correct."
    if "429" in text or "quota" in low or "rate limit" in low:
        return "This API key has hit its rate limit or quota. Wait a minute, or check your plan/billing."
    if "404" in text or ("model" in low and "not found" in low):
        detail = provider_error_detail(exc)
        summary = "The AI provider returned 404 (model or endpoint not found)."
        if detail:
            summary += f"\nProvider detail: {detail}"
        if "Gemini rejected all tried model IDs" in text:
            summary += f"\n{text[:500]}"
        if not detail and "Gemini rejected all tried model IDs" not in text:
            summary += f"\nError detail: {text[:500]}"
        summary += "\nCheck the model ID and API endpoint in the local configuration file."
        return summary
    if "connect" in low or "timed out" in low or "timeout" in low:
        return "Could not connect. Check your internet connection (or that Ollama is running)."
    return f"Error: {text}"


def load_openai() -> bool:
    """Import the optional 'openai' package (also after it was installed while the app is running)."""
    global OpenAI
    if OpenAI is None:
        try:
            importlib.invalidate_caches()
            OpenAI = importlib.import_module("openai").OpenAI
        except ImportError:
            return False
    return True


def make_brain(provider: str, api_key: str, model: str, base_url: str) -> LocalBrain | OpenAIBrain:
    """Build a brain from settings. Raises ValueError with a friendly message if something is missing."""
    info = PROVIDERS.get(provider, PROVIDERS["offline"])
    if provider == "offline" or provider not in PROVIDERS:
        return LocalBrain()
    api_key = api_key.strip() or ("" if info.needs_key else "local")
    model = model.strip() or info.model
    base_url = base_url.strip() or info.base_url
    if not api_key:
        raise ValueError("Please paste your API key.")
    if not model:
        raise ValueError("Please enter a model name.")
    if provider == "custom" and not base_url:
        raise ValueError("Please enter the server URL.")
    if not load_openai():
        raise ValueError("The 'openai' package is missing. Click 'Install openai' or run: pip install openai")
    fallbacks = GEMINI_FALLBACK_MODELS if provider == "gemini" else ()
    return OpenAIBrain(api_key, model, base_url, info.label, fallbacks)


def build_brain(store: "ChatStore") -> LocalBrain | OpenAIBrain:
    """Use the local secrets file when it matches the selected provider, then saved/env settings."""
    provider = store.get_setting("provider", "")
    file_settings = load_file_provider_settings()
    if file_settings and (not provider or provider == file_settings["provider"]):
        return make_brain(
            file_settings["provider"],
            file_settings["api_key"],
            file_settings["model"],
            file_settings["base_url"],
        )
    if provider:
        try:
            return make_brain(provider, store.get_setting("api_key", ""), store.get_setting("model", ""),
                              store.get_setting("base_url", ""))
        except ValueError as exc:
            print(f"Settings: {exc} - using offline mode.")
            return LocalBrain()
    if os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"):
        if not load_openai():
            print("API key found but the 'openai' package is missing - run: pip install openai")
        elif os.getenv("OPENAI_API_KEY"):
            return OpenAIBrain(os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                               os.getenv("OPENAI_BASE_URL"))
        else:
            return OpenAIBrain(os.environ["GEMINI_API_KEY"], os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                               GEMINI_BASE_URL, "Gemini", GEMINI_FALLBACK_MODELS)
    return LocalBrain()


# =========================================================== UI helpers

def hex_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def rgb_hex(r: float, g: float, b: float) -> str:
    return "#" + "".join(f"{max(0, min(255, round(v))):02x}" for v in (r, g, b))


def mix(c1: str, c2: str, t: float) -> str:
    a, b = hex_rgb(c1), hex_rgb(c2)
    return rgb_hex(*(x + (y - x) * t for x, y in zip(a, b)))


def _premul(c: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]


def _layer_rgba(stops: tuple, t: float) -> tuple[float, ...]:
    """Premultiplied colour of a CSS radial-gradient at distance t (fraction of its radius)."""
    for (t0, c0), (t1, c1) in itertools.pairwise(stops):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(a + (b - a) * f for a, b in zip(_premul(c0), _premul(c1)))
    return _premul(stops[-1][1])


def glow_rgb(x: float, y: float, w: float, h: float) -> tuple[float, float, float]:
    """App background colour at (x, y): --bg with the two radial glows from style.css on top."""
    r, g, b = hex_rgb(BG)
    radius = math.hypot(w, h) or 1.0
    for (cx, cy), stops in reversed(GLOW_LAYERS):
        pr, pg, pb, a = _layer_rgba(stops, math.hypot(x - cx * w, y - cy * h) / radius)
        r, g, b = pr + r * (1 - a), pg + g * (1 - a), pb + b * (1 - a)
    return r, g, b


def glow_image(master: Misc, w: int, h: int, block: int) -> PhotoImage:
    cols, rows = math.ceil(w / block), math.ceil(h / block)
    data = []
    for j in range(rows):
        y = (j + 0.5) * block
        data.append("{" + " ".join(rgb_hex(*glow_rgb((i + 0.5) * block, y, w, h)) for i in range(cols)) + "}")
    small = tk.PhotoImage(master=master, width=cols, height=rows)
    small.put(" ".join(data))
    return small.zoom(block)


def rounded_points(x1: float, y1: float, x2: float, y2: float, r: float) -> list[float]:
    r = max(1, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]


def rounded_rect(canvas: Canvas, x1: float, y1: float, x2: float, y2: float, r: float, **kw) -> int:
    return canvas.create_polygon(rounded_points(x1, y1, x2, y2, r), smooth=True, **kw)


def display_lines(text: Text) -> int:
    count = text.count("1.0", "end-1c", "displaylines")
    return (count[0] if isinstance(count, tuple) else count or 0) + 1


FONT_FAMILIES = ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "Noto Sans", "DejaVu Sans",
                 "Liberation Sans")
HEAD_FAMILIES = ("Segoe UI", "SF Pro Display", "Roboto", "Helvetica Neue", "Helvetica", "Arial", "Noto Sans",
                 "DejaVu Sans")


def ui_scale(root: Misc) -> float:
    return max(1.0, float(root.tk.call("tk", "scaling")) / 1.3333)


def pick_font(root: Misc, options: tuple[str, ...]) -> str:
    families = set(tkfont.families(root))
    return next((f for f in options if f in families), "TkDefaultFont")


# =========================================================== Markdown

@dataclass
class Block:
    kind: str          # p, h1, h2, h3, bullet, quote, code, gap
    text: str = ""
    marker: str = ""
    level: int = 0


INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def inline_runs(line: str) -> list[tuple[str, str]]:
    runs = []
    for part in INLINE_RE.split(line):
        if len(part) > 4 and part.startswith("**") and part.endswith("**"):
            runs.append((part[2:-2], "bold"))
        elif len(part) > 2 and part[0] == part[-1] == "`":
            runs.append((part[1:-1], "code"))
        elif len(part) > 2 and part[0] == part[-1] == "*":
            runs.append((part[1:-1], "italic"))
        elif part:
            runs.append((part, "body"))
    return runs


def parse_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    code: list[str] | None = None
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if code is None:
                code = []
            else:
                blocks.append(Block("code", "\n".join(code)))
                code = None
        elif code is not None:
            code.append(line)
        elif not line.strip():
            blocks.append(Block("gap"))
        elif m := re.match(r"(#{1,6})\s+(.*)", line):
            blocks.append(Block(f"h{min(3, len(m[1]))}", m[2]))
        elif m := re.match(r"(\s*)[-*\u2022]\s+(.*)", line):
            blocks.append(Block("bullet", m[2], "\u2022", len(m[1]) // 2))
        elif m := re.match(r"(\s*)(\d+)[.)]\s+(.*)", line):
            blocks.append(Block("bullet", m[3], f"{m[2]}.", len(m[1]) // 2))
        elif m := re.match(r">\s?(.*)", line):
            blocks.append(Block("quote", m[1]))
        else:
            blocks.append(Block("p", line))
    if code is not None:
        blocks.append(Block("code", "\n".join(code)))
    return blocks


# =========================================================== chat widgets

class GlowCanvas(tk.Canvas if tk else object):
    """Canvas that paints the app's glowing background behind its items (background-attachment: fixed)."""

    def __init__(self, parent: Widget, app: "HaniaAI", on_resize: Callable[[], None] | None = None, **kw):
        super().__init__(parent, bg=BG, highlightthickness=0, borderwidth=0, **kw)
        self.app = app
        self.on_resize = on_resize
        self.bind("<Configure>", self._configured)
        app.glow_canvases.append(self)

    def _configured(self, _event: Event) -> None:
        if self.on_resize:
            self.on_resize()
        self.paint_glow()

    def paint_glow(self) -> None:
        image = self.app.glow_img
        if image is None:
            return
        x = self.canvasx(0) - (self.winfo_rootx() - self.app.main.winfo_rootx())
        y = self.canvasy(0) - (self.winfo_rooty() - self.app.main.winfo_rooty())
        if self.find_withtag("glow"):
            self.coords("glow", x, y)
            self.itemconfigure("glow", image=image)
        else:
            self.create_image(x, y, image=image, anchor="nw", tags="glow")
        self.tag_lower("glow")


class ChatMessage:
    """One chat bubble drawn straight onto the messages canvas (styled like stChatMessage in style.css)."""

    _ids: ClassVar[Iterator[int]] = itertools.count()

    def __init__(self, app: "HaniaAI", role: str, text: str = "", thinking: bool = False):
        self.app = app
        self.role = role
        self.text = text
        self.thinking = thinking
        self.finished = not thinking
        self.error: str | None = None
        self.tag = f"msg{next(self._ids)}"
        self.y = 0.0
        self.height = 0.0
        self.dots = 0
        self._key: tuple | None = None

    def tick(self) -> None:
        if not self.thinking or self not in self.app.messages:
            return
        self.dots = (self.dots + 1) % 4
        self.app.refresh_message(self)
        self.app.root.after(350, self.tick)

    def layout(self, y: float, col_x: float, col_w: float) -> float:
        canvas = self.app.msg_canvas
        key = (col_x, col_w, self.text, self.error, self.finished, self.thinking and self.dots)
        if key == self._key:
            if y != self.y:
                canvas.move(self.tag, 0, y - self.y)
                self.y = y
            return self.height
        canvas.delete(self.tag)
        self._key, self.y = key, y
        self.height = self.draw(canvas, y, col_x, col_w)
        return self.height

    def draw(self, canvas: Canvas, y: float, col_x: float, col_w: float) -> float:
        app, px = self.app, self.app.px
        pad, av, gap = px(16), px(32), px(12)
        bw = col_w * 0.85
        bx = col_x + col_w - bw if self.role == "user" else col_x
        tags = (self.tag,)
        bubble = canvas.create_polygon(0, 0, 0, 0, smooth=True, fill=BUBBLE, outline=BORDER, tags=tags)
        text_w = bw - 2 * pad - av - gap
        top = y + pad + max(0, (av - app.metric("body")[2]) / 2)
        if self.role == "user":
            ax = bx + bw - pad - av
            app.draw_avatar(canvas, ax, y + pad, True, tags)
            item = canvas.create_text(ax - gap, top, text=self.text, anchor="ne", width=text_w, justify="right",
                                      font=app.fonts["body"], fill=CREAM, tags=tags)
            bottom = canvas.bbox(item)[3]
        else:
            app.draw_avatar(canvas, bx + pad, y + pad, False, tags)
            tx = bx + pad + av + gap
            if self.thinking:
                bottom = app.draw_runs(canvas, [("Thinking" + "." * self.dots, "body")], tx, top, text_w, tags,
                                       tone=MUTED)
            else:
                bottom = app.draw_markdown(canvas, self.text, tx, top, text_w, tags)
            if self.error:
                bottom = app.draw_runs(canvas, [(self.error, "body")], tx, bottom + (px(4) if self.text else 0),
                                       text_w, tags, tone=ERROR)
            if self.finished and self.text:
                canvas.create_text(tx, bottom + px(2), text="Copy", anchor="nw", font=app.fonts["small"], fill=MUTED,
                                   tags=(*tags, "copy"))
                bottom += px(2) + app.metric("small")[2]
        bh = max(bottom - y - pad, av) + 2 * pad
        canvas.coords(bubble, *rounded_points(bx, y, bx + bw, y + bh, px(16)))
        return bh


class ThemedForm:
    """Shared look for form widgets (labels, black entries, pill buttons) in the style.css theme."""

    form_bg = SIDEBAR
    family = "TkDefaultFont"
    scale = 1.0

    def init_form(self, root: Misc, bg: str) -> None:
        self.form_bg = bg
        self.scale = ui_scale(root)
        self.family = pick_font(root, FONT_FAMILIES)
        self.head_family = pick_font(root, HEAD_FAMILIES)
        self.body_font = tkfont.Font(root, family=self.family, size=12)
        self.normal_colors: dict[Label, tuple[str, str]] = {}

    def px(self, value: float) -> int:
        return int(value * self.scale)

    def label(self, parent: Widget, text: str, fg: str, font: tuple, **kw) -> Label:
        return tk.Label(parent, text=text, bg=self.form_bg, fg=fg, font=font, **kw)

    def entry(self, parent: Widget, var: StringVar, **kw) -> Entry:
        return tk.Entry(parent, textvariable=var, bg=INPUT_BG, fg=CREAM, insertbackground=CREAM, relief="flat",
                        highlightthickness=1, highlightbackground=INPUT_BORDER, highlightcolor=ACCENT,
                        selectbackground=ACCENT, selectforeground="#ffffff", font=self.body_font,
                        width=kw.pop("width", 44), **kw)

    def button(self, parent: Widget, text: str, command: Callable[[], None],
               primary: bool = False, danger: bool = False) -> Label:
        px = self.px
        normal = (ACCENT, ACCENT) if primary else (BUBBLE, BORDER)
        hover = ("#3498db", "#3498db") if primary else (HOVER, ERROR if danger else CREAM)
        fg = "#ffffff" if primary else (ERROR if danger else CREAM)
        btn = tk.Label(parent, text=text, bg=normal[0], fg=fg, cursor="hand2",
                       font=(self.family, 10, "bold" if primary else "normal"), padx=px(14), pady=px(6),
                       highlightthickness=1, highlightbackground=normal[1])
        self.normal_colors[btn] = normal
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover[0], highlightbackground=hover[1]))
        btn.bind("<Leave>", lambda _e: self.paint_button(btn))
        btn.bind("<Button-1>", lambda _e: command())
        return btn

    def paint_button(self, btn: Label) -> None:
        bg, border = self.normal_colors[btn]
        btn.configure(bg=bg, highlightbackground=border)


class SettingsDialog(ThemedForm):
    """Settings window for account management only."""

    def __init__(self, app: "HaniaAI"):
        self.app = app
        self.init_form(app.root, SIDEBAR)
        px = self.px

        self.top = top = tk.Toplevel(app.root, bg=SIDEBAR, padx=px(28), pady=px(22))
        top.title(f"{APP_NAME} - Settings")
        top.transient(app.root)
        top.resizable(False, False)
        top.columnconfigure(0, weight=1)
        top.bind("<Escape>", lambda _e: self.close())
        top.protocol("WM_DELETE_WINDOW", self.close)

        self.label(top, "\u2699  Settings", GLOW, (app.head_family, 18, "bold")).grid(sticky="w")
        self.label(top, "Manage your Hania AI account.", MUTED, (app.family, 10)).grid(
            sticky="w", pady=(px(2), px(16)))

        account = tk.Frame(top, bg=SIDEBAR, highlightthickness=1, highlightbackground=BORDER, padx=px(14),
                           pady=px(10))
        account.grid(sticky="ew")
        self.label(account, "Account", CREAM, (app.family, 10, "bold")).pack(anchor="w")
        row = tk.Frame(account, bg=SIDEBAR)
        row.pack(fill="x", pady=(px(4), 0))
        self.label(row, f"Signed in as {app.user_name}", MUTED, (app.family, 10)).pack(side="left")
        self.button(row, "Delete account", self.delete_account, danger=True).pack(side="right")
        self.button(row, "Change password", self.change_password).pack(side="right", padx=(0, px(8)))

        self.status = self.label(top, "", MUTED, (app.family, 10), wraplength=px(440), justify="left")
        self.status.grid(sticky="w", pady=(px(12), 0))
        actions = tk.Frame(top, bg=SIDEBAR)
        actions.grid(sticky="e", pady=(px(14), 0))
        self.button(actions, "Close", self.close, primary=True).pack(side="left")

        top.update_idletasks()
        x = app.root.winfo_rootx() + (app.root.winfo_width() - top.winfo_reqwidth()) // 2
        y = app.root.winfo_rooty() + (app.root.winfo_height() - top.winfo_reqheight()) // 3
        top.geometry(f"+{max(0, x)}+{max(0, y)}")
        top.grab_set()
        top.focus_force()

    def set_status(self, text: str, color: str = MUTED) -> None:
        self.status.configure(text=text, fg=color)

    def ask_password(self, title: str, prompt: str) -> str | None:
        return simpledialog.askstring(title, prompt, show="\u2022", parent=self.top)

    def change_password(self) -> None:
        old = self.ask_password("Change password", "Current password:")
        if old is None:
            return
        new = self.ask_password("Change password", f"New password (at least {MIN_PASSWORD} characters):")
        if new is None:
            return
        confirm = self.ask_password("Change password", "Type the new password again:")
        if confirm is None:
            return
        try:
            self.app.users.change_password(self.app.user_key, old, new, confirm)
        except ValueError as exc:
            self.set_status(str(exc), ERROR)
            return
        self.set_status("Password changed.", "#22c55e")

    def delete_account(self) -> None:
        if not messagebox.askyesno("Delete account", f"Delete the account \"{self.app.user_name}\" and ALL its chats?"
                                   "\n\nThis cannot be undone.", icon="warning", parent=self.top):
            return
        password = self.ask_password("Delete account", "Enter your password to confirm:")
        if password is None:
            return
        try:
            self.app.users.delete(self.app.user_key, password)
        except ValueError as exc:
            self.set_status(str(exc), ERROR)
            return
        self.close()
        self.app.log_out()

    def close(self) -> None:
        if not self.top.winfo_exists():
            return
        self.top.grab_release()
        self.top.destroy()
        if self.app.alive:
            self.app.entry.focus_set()


class LoginScreen(ThemedForm):
    """Full-window log in / sign up screen drawn over the glowing app background."""

    def __init__(self, root: Tk, users: UserStore, on_login: Callable[[str], None]):
        self.root = root
        self.users = users
        self.on_login = on_login
        self.init_form(root, BUBBLE)
        self.mode = "login"
        self.glow: PhotoImage | None = None
        self._glow_job: str | None = None
        px = self.px
        root.title(f"{APP_NAME} - Sign in")

        self.canvas = c = tk.Canvas(root, bg=BG, highlightthickness=0, borderwidth=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", self.on_resize)

        self.username = tk.StringVar(value=users.display_name(users.data["last_user"]) if users.data["last_user"]
                                     else "")
        self.password = tk.StringVar()
        self.confirm = tk.StringVar()
        self.remember = tk.BooleanVar(value=True)

        self.card = card = tk.Frame(c, bg=BUBBLE)
        card.columnconfigure(0, weight=1)
        self.label(card, APP_NAME, GLOW, (self.head_family, 30, "bold")).grid(pady=(0, px(2)))
        self.label(card, WELCOME_TEXT, CREAM, (self.family, 12)).grid(pady=(0, px(18)))
        tabs = tk.Frame(card, bg=BUBBLE)
        tabs.grid(pady=(0, px(14)))
        self.tabs = {"login": self.button(tabs, "Log in", lambda: self.set_mode("login")),
                     "signup": self.button(tabs, "Sign up", lambda: self.set_mode("signup"))}
        self.tabs["login"].pack(side="left", padx=(0, px(8)))
        self.tabs["signup"].pack(side="left")

        self.fields = tk.Frame(card, bg=BUBBLE)
        self.fields.grid(sticky="ew")
        self.fields.columnconfigure(0, weight=1)
        bold = (self.family, 10, "bold")
        self.user_label = self.label(self.fields, "Username", CREAM, bold)
        self.user_entry = self.entry(self.fields, self.username, width=32)
        self.pass_label = self.label(self.fields, "Password", CREAM, bold)
        self.pass_entry = self.entry(self.fields, self.password, show="\u2022", width=32)
        self.confirm_label = self.label(self.fields, "Confirm password", CREAM, bold)
        self.confirm_entry = self.entry(self.fields, self.confirm, show="\u2022", width=32)
        self.hint = self.label(self.fields, f"3-32 letters/numbers for the username, {MIN_PASSWORD}+ characters "
                               "for the password.", MUTED, (self.family, 9), wraplength=px(330), justify="left")
        self.remember_box = tk.Checkbutton(self.fields, text="Keep me signed in", variable=self.remember,
                                           bg=BUBBLE, fg=CREAM, activebackground=BUBBLE, activeforeground=CREAM,
                                           selectcolor=INPUT_BG, highlightthickness=0, borderwidth=0,
                                           font=(self.family, 10), cursor="hand2")
        for entry in (self.user_entry, self.pass_entry, self.confirm_entry):
            entry.bind("<Return>", lambda _e: self.submit())

        self.error = self.label(card, "", ERROR, (self.family, 10), wraplength=px(330), justify="left")
        self.error.grid(sticky="w", pady=(px(8), 0))
        self.submit_btn = self.button(card, "", self.submit, primary=True)
        self.submit_btn.grid(sticky="ew", pady=(px(8), px(12)))
        self.switch = self.label(card, "", ACCENT, (self.family, 10, "underline"), cursor="hand2")
        self.switch.grid()
        self.switch.bind("<Button-1>", lambda _e: self.set_mode("signup" if self.mode == "login" else "login"))

        self.card_item = c.create_window(0, 0, window=card, anchor="center")
        self.set_mode("signup" if not users.data["users"] else "login")

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        px = self.px
        for name, tab in self.tabs.items():
            self.normal_colors[tab] = (ACCENT, ACCENT) if name == mode else (BUBBLE, BORDER)
            self.paint_button(tab)
        for widget in self.fields.winfo_children():
            widget.grid_forget()
        self.user_label.grid(sticky="w")
        self.user_entry.grid(sticky="ew", pady=(px(4), px(10)), ipady=px(6))
        self.pass_label.grid(sticky="w")
        self.pass_entry.grid(sticky="ew", pady=(px(4), px(10)), ipady=px(6))
        if mode == "signup":
            self.confirm_label.grid(sticky="w")
            self.confirm_entry.grid(sticky="ew", pady=(px(4), px(6)), ipady=px(6))
            self.hint.grid(sticky="w", pady=(0, px(8)))
        self.remember_box.grid(sticky="w")
        self.submit_btn.configure(text="Log in" if mode == "login" else "Create account")
        self.switch.configure(text="New here? Create an account" if mode == "login"
                              else "Already have an account? Log in")
        self.error.configure(text="")
        self.password.set("")
        self.confirm.set("")
        (self.pass_entry if mode == "login" and self.username.get() else self.user_entry).focus_set()
        self.root.after_idle(self.place_card)

    def submit(self) -> None:
        name, password = self.username.get().strip(), self.password.get()
        if self.mode == "signup":
            try:
                key = self.users.create(name, password, self.confirm.get())
            except ValueError as exc:
                self.error.configure(text=str(exc))
                return
        else:
            if not name or not password:
                self.error.configure(text="Please enter your username and password.")
                return
            if not self.users.verify(name, password):
                self.error.configure(text="Wrong username or password.")
                self.password.set("")
                self.pass_entry.focus_set()
                return
            key = self.users.key(name)
        self.users.signed_in(key, self.remember.get())
        self.destroy()
        self.on_login(key)

    def destroy(self) -> None:
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        self.canvas.destroy()

    def on_resize(self, _event: Event) -> None:
        self.place_card()
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        self._glow_job = self.root.after(80, self.paint_glow)

    def place_card(self) -> None:
        c, px = self.canvas, self.px
        if not c.winfo_exists():
            return
        w, h = c.winfo_width(), c.winfo_height()
        self.card.update_idletasks()
        cw, ch = self.card.winfo_reqwidth(), self.card.winfo_reqheight()
        c.coords(self.card_item, w / 2, h / 2)
        pad = px(30)
        c.delete("card")
        rounded_rect(c, w / 2 - cw / 2 - pad, h / 2 - ch / 2 - pad, w / 2 + cw / 2 + pad, h / 2 + ch / 2 + pad,
                     px(22), fill=BUBBLE, outline=BORDER, tags="card")
        c.tag_lower("card")
        c.tag_lower("glow")

    def paint_glow(self) -> None:
        self._glow_job = None
        c = self.canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        self.glow = glow_image(self.root, w, h, max(3, self.px(4)))
        c.delete("glow")
        c.create_image(0, 0, image=self.glow, anchor="nw", tags="glow")
        c.tag_lower("glow")


# =========================================================== main app

class HaniaAI:
    def __init__(self, root: Tk, users: UserStore, user_key: str, on_logout: Callable[[], None]):
        self.root = root
        self.users = users
        self.user_key = user_key
        self.user_name = users.display_name(user_key)
        self.on_logout = on_logout
        self.store = store = users.chat_store(user_key)
        self.brain = build_brain(store)
        self.alive = True
        self.scale = ui_scale(root)
        self.family = pick_font(root, FONT_FAMILIES)
        self.head_family = pick_font(root, HEAD_FAMILIES)
        self.mono = pick_font(root, ("Cascadia Code", "Consolas", "SF Mono", "Menlo", "Ubuntu Mono",
                                     "DejaVu Sans Mono", "Liberation Mono", "Courier New"))
        size = 12
        self.fonts = {
            "body": tkfont.Font(root, family=self.family, size=size),
            "bold": tkfont.Font(root, family=self.family, size=size, weight="bold"),
            "italic": tkfont.Font(root, family=self.family, size=size, slant="italic"),
            "code": tkfont.Font(root, family=self.mono, size=size - 1),
            "h1": tkfont.Font(root, family=self.head_family, size=size + 6, weight="bold"),
            "h2": tkfont.Font(root, family=self.head_family, size=size + 4, weight="bold"),
            "h3": tkfont.Font(root, family=self.head_family, size=size + 2, weight="bold"),
            "small": tkfont.Font(root, family=self.family, size=size - 2),
            "sidebar": tkfont.Font(root, family=self.family, size=size - 2),
            "title": tkfont.Font(root, family=self.head_family, size=44, weight="bold"),
            "subtitle": tkfont.Font(root, family=self.family, size=17),
        }
        self._metrics: dict[str, tuple[int, int, int]] = {}
        self._widths: dict[tuple[str, str], int] = {}
        self.muted_bg = mix(CREAM, BG, 0.35)

        self.current_id: str | None = None
        self.messages: list[ChatMessage] = []
        self.reply_msg: ChatMessage | None = None
        self.reply_text = ""
        self.reply_conv: dict = {}
        self.tokens: queue.Queue = queue.Queue()
        self.generating = False
        self.stop_event = threading.Event()
        self.sidebar_visible = store.get_setting("sidebar", "1") == "1"
        self.sidebar_offset = 0
        self.settings_win: SettingsDialog | None = None
        self.glow_canvases: list[GlowCanvas] = []
        self.glow_img: PhotoImage | None = None
        self._glow_job: str | None = None
        self._glow_size = (0, 0)
        self.heading_bg = BG
        self.input_focus = False
        self.send_hover = False
        self.send_geom = (0.0, 0.0, 0.0)
        self._drag = (0, 0.0)

        root.title(f"{APP_NAME} - {self.user_name}")
        root.bind_all("<MouseWheel>", self.on_wheel)
        root.bind_all("<Button-4>", self.on_wheel)
        root.bind_all("<Button-5>", self.on_wheel)
        root.bind("<Control-n>", lambda _e: self.new_chat())
        root.bind("<Escape>", lambda _e: self.stop())

        self.build_ui()
        root.after(30, self.poll_tokens)
        root.after(70, self.pulse)

    def px(self, value: float) -> int:
        return int(value * self.scale)

    def log_out(self) -> None:
        """Close this user's session and go back to the sign-in screen."""
        if not self.alive:
            return
        if self.generating:
            self.stop()
        self.alive = False
        if self.settings_win:
            self.settings_win.close()
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.unbind_all(sequence)
        self.root.unbind("<Control-n>")
        self.root.unbind("<Escape>")
        self.messages = []
        self.frame.destroy()
        self.users.forget()
        self.on_logout()

    def metric(self, key: str) -> tuple[int, int, int]:
        if key not in self._metrics:
            f = self.fonts[key]
            self._metrics[key] = (f.metrics("ascent"), f.metrics("descent"), f.metrics("linespace"))
        return self._metrics[key]

    def measure(self, key: str, text: str) -> int:
        width = self._widths.get((key, text))
        if width is None:
            if len(self._widths) > 50000:
                self._widths.clear()
            width = self._widths[key, text] = self.fonts[key].measure(text)
        return width

    # ------------------------------------------------------------ layout

    def build_ui(self) -> None:
        px = self.px
        self.root.configure(bg=BG)
        self.frame = tk.Frame(self.root, bg=BG)
        self.frame.pack(fill="both", expand=True)

        self.sidebar = tk.Canvas(self.frame, bg=SIDEBAR, width=px(270), highlightthickness=0, borderwidth=0)
        self.sidebar.bind("<Configure>", lambda _e: self.draw_sidebar())
        self.bind_button(self.sidebar, "sb_new", self.new_chat)
        self.bind_button(self.sidebar, "sb_settings", self.open_settings)
        self.bind_button(self.sidebar, "sb_logout", self.log_out)
        self.sidebar.tag_bind("conv", "<Enter>", lambda _e: self.hover_conv(True))
        self.sidebar.tag_bind("conv", "<Leave>", lambda _e: self.hover_conv(False))
        self.sidebar.tag_bind("conv", "<Button-1>", lambda _e: self.click_conv())
        self.sidebar.tag_bind("conv", "<Button-3>", self.conv_menu)
        self.sidebar.tag_bind("conv", "<Button-2>", self.conv_menu)
        if self.sidebar_visible:
            self.sidebar.pack(side="left", fill="y")

        self.main = tk.Frame(self.frame, bg=BG)
        self.main.pack(side="left", fill="both", expand=True)
        self.main.bind("<Configure>", self.on_main_resize)

        self.header = GlowCanvas(self.main, self, on_resize=self.draw_header, height=px(64))
        self.header.pack(side="top", fill="x")
        self.bind_button(self.header, "menu", self.toggle_sidebar, normal=(INPUT_BG, BORDER), hover=(BUBBLE, CREAM))
        self.bind_button(self.header, "hdr_new", self.new_chat)
        self.bind_button(self.header, "hdr_settings", self.open_settings)

        self.build_input()
        self.content = tk.Frame(self.main, bg=BG)
        self.content.pack(side="top", fill="both", expand=True)
        self.build_welcome()
        self.build_messages()

        self.open_conversation(None)
        self.entry.focus_set()

    def bind_button(self, canvas: Canvas, name: str, command: Callable[[], None],
                    normal: tuple[str, str] = (BUBBLE, BORDER), hover: tuple[str, str] = (HOVER, CREAM)) -> None:
        def style(colors: tuple[str, str], cursor: str) -> None:
            canvas.itemconfigure(f"{name}_bg", fill=colors[0], outline=colors[1])
            canvas.configure(cursor=cursor)
        canvas.tag_bind(name, "<Enter>", lambda _e: style(hover, "hand2"))
        canvas.tag_bind(name, "<Leave>", lambda _e: style(normal, ""))
        canvas.tag_bind(name, "<Button-1>", lambda _e: command())

    def draw_button(self, canvas: Canvas, name: str, box: tuple[float, float, float, float], text: str,
                    font: "tkfont.Font | tuple", fill: str = BUBBLE, outline: str = BORDER, radius: int = 10,
                    anchor: str = "center", tags: tuple[str, ...] = ()) -> None:
        x1, y1, x2, y2 = box
        rounded_rect(canvas, x1, y1, x2, y2, self.px(radius), fill=fill, outline=outline,
                     tags=(name, f"{name}_bg", *tags))
        tx = (x1 + x2) / 2 if anchor == "center" else x1 + self.px(14)
        canvas.create_text(tx, (y1 + y2) / 2, text=text, anchor=anchor, fill=CREAM, font=font, tags=(name, *tags))

    def draw_avatar(self, canvas: Canvas, x: float, y: float, user: bool, tags: tuple[str, ...]) -> None:
        s = self.px(32)
        rounded_rect(canvas, x, y, x + s, y + s, self.px(8), fill=CREAM, outline=CREAM if user else "#111112",
                     tags=tags)
        if user:
            cx = x + s / 2
            canvas.create_oval(cx - s * 0.16, y + s * 0.17, cx + s * 0.16, y + s * 0.49, fill=BUBBLE, outline="",
                               tags=tags)
            canvas.create_arc(cx - s * 0.3, y + s * 0.57, cx + s * 0.3, y + s * 1.17, start=0, extent=180,
                              fill=BUBBLE, outline="", tags=tags)
        else:
            canvas.create_text(x + s / 2, y + s / 2, text="H", fill="#000000", font=(self.head_family, 12, "bold"),
                               tags=tags)

    def draw_header(self) -> None:
        c, px = self.header, self.px
        c.delete("ui")
        w = c.winfo_width()
        x, y, s = px(20), px(11), px(42)
        self.draw_button(c, "menu", (x, y, x + s, y + s), "\u2630", (self.family, 15, "bold"), fill=INPUT_BG,
                         radius=12, tags=("ui",))
        title = c.create_text(x + s + px(14), y + s / 2, text=APP_NAME, anchor="w", fill=GLOW,
                              font=(self.head_family, 15, "bold"), tags="ui")
        c.create_text(c.bbox(title)[2] + px(8), y + s / 2 + px(1), text=self.brain.name, anchor="w",
                      fill=self.muted_bg, font=self.fonts["small"], tags="ui")
        bw = px(120)
        self.draw_button(c, "hdr_new", (w - px(20) - bw, y + px(3), w - px(20), y + s - px(3)), "+  New chat",
                         (self.family, 10), tags=("ui",))
        gx = w - px(30) - bw - (s - px(6))
        self.draw_button(c, "hdr_settings", (gx, y + px(3), gx + s - px(6), y + s - px(3)), "\u2699",
                         (self.family, 14), tags=("ui",))

    def draw_sidebar(self) -> None:
        c, px = self.sidebar, self.px
        w, h = c.winfo_width(), c.winfo_height()
        if w < px(60):
            return
        c.delete("all")
        c.create_line(w - 1, 0, w - 1, h, fill=BORDER)
        c.create_text(px(22), px(34), text="\u2726", anchor="w", fill=GLOW, font=(self.head_family, 16))
        c.create_text(px(48), px(34), text=APP_NAME, anchor="w", fill=CREAM, font=(self.head_family, 14, "bold"))
        self.draw_button(c, "sb_new", (px(14), px(64), w - px(15), px(104)), "+   New chat", (self.family, 11),
                         anchor="w")
        c.create_text(px(20), px(130), text="Recent", anchor="w", fill=MUTED, font=(self.family, 9, "bold"))

        mid = w / 2
        self.draw_button(c, "sb_settings", (px(14), h - px(118), mid - px(4), h - px(80)), "\u2699  Settings",
                         (self.family, 10))
        self.draw_button(c, "sb_logout", (mid + px(4), h - px(118), w - px(15), h - px(80)), "\u21aa  Log out",
                         (self.family, 10))
        c.create_line(px(14), h - px(66), w - px(15), h - px(66), fill=BORDER)
        ay, r = h - px(34), px(16)
        c.create_oval(px(20), ay - r, px(20) + 2 * r, ay + r, fill=CREAM, outline="")
        c.create_text(px(20) + r, ay, text=self.user_name[:1].upper(), fill="#000000",
                      font=(self.head_family, 12, "bold"))
        tx, max_w = px(28) + 2 * r, w - px(28) - 2 * r - px(16)
        c.create_text(tx, ay - px(9), text=self.fit(self.user_name, (self.family, 11, "bold"), max_w), anchor="w",
                      fill=CREAM, font=(self.family, 11, "bold"))
        online = not isinstance(self.brain, LocalBrain)
        c.create_text(tx, ay + px(10), text="\u25cf", anchor="w", fill="#22c55e" if online else "#f59e0b",
                      font=(self.family, 8))
        c.create_text(tx + px(14), ay + px(10), text=self.fit(self.brain.name, (self.family, 9), max_w - px(14)),
                      anchor="w", fill=MUTED, font=(self.family, 9))

        convs = self.store.conversations
        top, step, item_h = px(146), px(42), px(36)
        visible = max(1, (h - px(128) - top) // step)
        self.sidebar_offset = max(0, min(self.sidebar_offset, len(convs) - visible))
        if not convs:
            c.create_text(px(20), top + px(12), text="No chats yet", anchor="w", fill=MUTED, font=(self.family, 10))
        max_w = w - px(58)
        for i, conv in enumerate(convs[self.sidebar_offset:self.sidebar_offset + visible]):
            y1 = top + i * step
            active = conv["id"] == self.current_id
            title = conv["title"]
            while len(title) > 1 and self.measure("sidebar", title) > max_w:
                title = title[:-2] + "\u2026"
            self.draw_button(c, f"conv_{conv['id']}", (px(14), y1, w - px(15), y1 + item_h), title,
                             self.fonts["sidebar"], fill=HOVER if active else BUBBLE,
                             outline=CREAM if active else BORDER, anchor="w", tags=("conv",))

    @staticmethod
    def fit(text: str, font: tuple, width: float) -> str:
        measure = tkfont.Font(font=font).measure
        while len(text) > 1 and measure(text) > width:
            text = text[:-2] + "\u2026"
        return text

    def conv_under_pointer(self) -> dict | None:
        for tag in self.sidebar.gettags("current"):
            if tag.startswith("conv_") and not tag.endswith("_bg"):
                return self.store.get(tag[5:])
        return None

    def hover_conv(self, on: bool) -> None:
        self.sidebar.configure(cursor="hand2" if on else "")
        conv = self.conv_under_pointer()
        if conv and conv["id"] != self.current_id:
            self.sidebar.itemconfigure(f"conv_{conv['id']}_bg", fill=HOVER if on else BUBBLE,
                                       outline=CREAM if on else BORDER)

    def click_conv(self) -> None:
        conv = self.conv_under_pointer()
        if conv:
            self.open_conversation(conv["id"])

    def conv_menu(self, event: Event) -> None:
        conv = self.conv_under_pointer()
        if not conv:
            return
        menu = tk.Menu(self.root, tearoff=0, bg=BUBBLE, fg=CREAM, activebackground=HOVER, activeforeground=CREAM,
                       borderwidth=0)
        menu.add_command(label="Rename", command=lambda: self.rename_chat(conv))
        menu.add_command(label="Delete", command=lambda: self.delete_chat(conv))
        menu.tk_popup(event.x_root, event.y_root)

    def build_welcome(self) -> None:
        self.welcome = GlowCanvas(self.content, self, on_resize=self.draw_welcome)

    def draw_welcome(self) -> None:
        c, px = self.welcome, self.px
        w, h = c.winfo_width(), c.winfo_height()
        if w < px(100):
            return
        c.delete("ui")
        title_h, sub_h = self.metric("title")[2], self.metric("subtitle")[2]
        top = max(px(12), (h - title_h - sub_h) * 0.45)
        cx = w / 2
        oy = c.winfo_rooty() - self.main.winfo_rooty()
        self.heading_bg = rgb_hex(*glow_rgb(cx, top + oy + title_h / 2, max(1, self.main.winfo_width()),
                                            max(1, self.main.winfo_height())))
        d = px(1.5) or 1
        for k in (3, 2, 1):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)):
                c.create_text(cx + dx * k * d, top + dy * k * d, text=APP_NAME, anchor="n", font=self.fonts["title"],
                              fill=self.heading_bg, tags=("ui", f"hshadow{k}"))
        c.create_text(cx, top, text=APP_NAME, anchor="n", font=self.fonts["title"], fill=GLOW, tags="ui")
        sub_y = top + title_h - px(4)
        c.create_text(cx, sub_y, text=WELCOME_TEXT, anchor="n", font=self.fonts["subtitle"], fill=CREAM, tags="ui")

    def pulse(self) -> None:
        """The heading's 3s text-shadow pulse animation from style.css."""
        if not self.alive:
            return
        if self.welcome.winfo_ismapped():
            t = (1 - math.cos(2 * math.pi * (time.monotonic() % 3.0) / 3.0)) / 2
            shadow, alpha = mix(SHADOW[0], SHADOW[1], t), 0.2 + 0.2 * t
            for k in (1, 2, 3):
                self.welcome.itemconfigure(f"hshadow{k}", fill=mix(self.heading_bg, shadow, alpha * (4 - k) / 4))
        self.root.after(70, self.pulse)

    def build_messages(self) -> None:
        self.msg_canvas = c = GlowCanvas(self.content, self, on_resize=self.on_messages_resize,
                                         yscrollincrement=self.px(24))
        c.configure(yscrollcommand=self.on_yscroll)
        c.create_polygon(0, 0, 0, 0, smooth=True, fill=THUMB, outline="", tags="thumb", state="hidden")
        c.tag_bind("thumb", "<ButtonPress-1>", self.thumb_press)
        c.tag_bind("thumb", "<B1-Motion>", self.thumb_drag)
        c.tag_bind("copy", "<Enter>", lambda _e: self.hover_copy(True))
        c.tag_bind("copy", "<Leave>", lambda _e: self.hover_copy(False))
        c.tag_bind("copy", "<Button-1>", lambda _e: self.copy_current())

    def build_input(self) -> None:
        px = self.px
        self.input_canvas = c = GlowCanvas(self.main, self, on_resize=self.redraw_input, height=px(100))
        c.pack(side="bottom", fill="x")
        self.line_height = self.metric("body")[2]
        self.entry = tk.Text(c, height=1, width=1, wrap="word", bg=INPUT_BG, fg="#ffffff", insertbackground="#ffffff",
                             insertwidth=2, relief="flat", borderwidth=0, highlightthickness=0,
                             font=self.fonts["body"], padx=0, pady=0, selectbackground=ACCENT,
                             selectforeground="#ffffff")
        self.entry_item = c.create_window(0, 0, window=self.entry, anchor="nw")
        self.placeholder = tk.Label(c, text=f"Ask {APP_NAME} anything...", bg=INPUT_BG, fg=MUTED,
                                    font=self.fonts["body"], padx=0, pady=0, cursor="xterm")
        self.placeholder_item = c.create_window(0, 0, window=self.placeholder, anchor="nw")
        self.placeholder.bind("<Button-1>", lambda _e: self.entry.focus_set())
        self.entry.bind("<Return>", self.on_enter)
        self.entry.bind("<KeyRelease>", lambda _e: self.redraw_input())
        self.entry.bind("<FocusIn>", lambda _e: self.set_input_focus(True))
        self.entry.bind("<FocusOut>", lambda _e: self.set_input_focus(False))
        c.tag_bind("send", "<Button-1>", lambda _e: self.on_send_click())
        c.tag_bind("send", "<Enter>", lambda _e: self.hover_send(True))
        c.tag_bind("send", "<Leave>", lambda _e: self.hover_send(False))

    def redraw_input(self) -> None:
        c, px = self.input_canvas, self.px
        w = c.winfo_width()
        if w < px(160):
            return
        col = min(w - px(48), px(820))
        x1 = (w - col) / 2
        x2 = x1 + col
        lines = max(1, min(6, display_lines(self.entry)))
        entry_h = lines * self.line_height
        top = px(6)
        bottom = top + entry_h + px(30)
        height = int(bottom + px(32))
        if int(c.cget("height")) != height:
            c.configure(height=height)
        c.delete("capsule", "send", "foot")
        if self.input_focus:
            rounded_rect(c, x1 - px(3), top - px(3), x2 + px(3), bottom + px(3), px(27), fill="",
                         outline=mix(ACCENT, BG, 0.45), width=px(2), tags="capsule")
        rounded_rect(c, x1, top, x2, bottom, px(24), fill=INPUT_BG,
                     outline=ACCENT if self.input_focus else INPUT_BORDER, tags="capsule")
        tx, ty = x1 + px(24), top + px(15)
        c.coords(self.entry_item, tx, ty)
        c.itemconfigure(self.entry_item, width=x2 - px(64) - tx, height=entry_h)
        c.coords(self.placeholder_item, tx, ty)
        c.itemconfigure(self.placeholder_item, state="hidden" if self.get_input() else "normal")

        r = px(18)
        cx = x2 - px(30)
        cy = (top + bottom) / 2 if lines == 1 else bottom - px(12) - r
        self.send_geom = (cx, cy, r)
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="", tags=("send", "send_circle"))
        c.create_text(cx, cy, text="\u25a0" if self.generating else "\u2191", font=(self.family, 13, "bold"),
                      tags=("send", "send_icon"))
        self.style_send()
        c.create_text(w / 2, bottom + px(16), text=f"{APP_NAME} can make mistakes. Check important info.",
                      fill=self.muted_bg, font=(self.family, 9), tags="foot")

    def style_send(self) -> None:
        c = self.input_canvas
        cx, cy, r = self.send_geom
        if self.send_hover:
            r *= 1.1
        c.coords("send_circle", cx - r, cy - r, cx + r, cy + r)
        c.itemconfigure("send_circle", fill=ACCENT if self.send_hover else SEND_BG)
        c.itemconfigure("send_icon", fill="#ffffff" if self.send_hover else ACCENT)

    # ------------------------------------------------------------ rich text

    def draw_runs(self, canvas: Canvas, runs: list[tuple[str, str]], x0: float, y: float, width: float,
                  tags: tuple[str, ...], base: str = "body", tone: str | None = None) -> float:
        """Word-wrap styled text runs onto the canvas; returns the y below the last line."""
        px = self.px
        heading = base.startswith("h")
        segs: list[list] = []  # [x, text, font key] on the current line
        x = x0

        def color(key: str) -> str:
            if tone:
                return tone
            if key == "code":
                return CODE_FG
            return GLOW if heading else CREAM

        def flush() -> None:
            nonlocal y, segs
            if not segs:
                return
            asc = max(self.metric(k)[0] for _, _, k in segs)
            desc = max(self.metric(k)[1] for _, _, k in segs)
            for sx, text, key in segs:
                ty = y + asc - self.metric(key)[0]
                if key == "code":
                    canvas.create_rectangle(sx - px(3), ty, sx + self.measure(key, text.rstrip()) + px(3),
                                            ty + self.metric(key)[2], fill=CODE_BG, outline=BORDER, tags=tags)
                canvas.create_text(sx, ty, text=text, anchor="nw", font=self.fonts[key], fill=color(key), tags=tags)
            y += asc + desc + px(5)
            segs = []

        def add(text: str, key: str) -> None:
            nonlocal x
            if segs and segs[-1][2] == key:
                segs[-1][1] += text
            else:
                segs.append([x, text, key])
            x += self.measure(key, text)

        for text, style in runs:
            key = base if heading and style != "code" else style
            for tok in re.findall(r"\S+|\s+", text):
                if tok.isspace():
                    if x > x0:
                        add(" ", key)
                    continue
                w = self.measure(key, tok)
                if x > x0 and x + w > x0 + width:
                    flush()
                    x = x0
                while w > width and len(tok) > 1:
                    n = max(1, int(len(tok) * width / w))
                    while n > 1 and self.measure(key, tok[:n]) > width:
                        n -= 1
                    add(tok[:n], key)
                    flush()
                    x = x0
                    tok = tok[n:]
                    w = self.measure(key, tok)
                add(tok, key)
        flush()
        return y

    def draw_code(self, canvas: Canvas, code: str, x: float, y: float, width: float,
                  tags: tuple[str, ...]) -> float:
        pad = self.px(12)
        per_line = max(8, int((width - 2 * pad) / max(1, self.measure("code", "0"))))
        lines = []
        for line in code.replace("\t", "    ").split("\n"):
            while len(line) > per_line:
                lines.append(line[:per_line])
                line = line[per_line:]
            lines.append(line)
        h = len(lines) * self.metric("code")[2] + 2 * pad
        rounded_rect(canvas, x, y, x + width, y + h, self.px(8), fill=CODE_BG, outline=BORDER, tags=tags)
        canvas.create_text(x + pad, y + pad, text="\n".join(lines), anchor="nw", font=self.fonts["code"],
                           fill=CODE_FG, tags=tags)
        return y + h

    def draw_markdown(self, canvas: Canvas, text: str, x: float, y: float, width: float,
                      tags: tuple[str, ...]) -> float:
        px = self.px
        first = True
        for block in parse_markdown(text):
            if block.kind == "gap":
                if not first:
                    y += px(6)
                continue
            if block.kind == "code":
                y = self.draw_code(canvas, block.text, x, y + (0 if first else px(4)), width, tags) + px(8)
            elif block.kind in ("h1", "h2", "h3"):
                y = self.draw_runs(canvas, inline_runs(block.text), x, y + (0 if first else px(6)), width, tags,
                                   base=block.kind) + px(2)
            elif block.kind == "bullet":
                indent = px(4) + block.level * px(20)
                canvas.create_text(x + indent, y, text=block.marker, anchor="nw", font=self.fonts["body"],
                                   fill=GLOW, tags=tags)
                mw = max(px(18), self.measure("body", block.marker) + px(8))
                y = self.draw_runs(canvas, inline_runs(block.text), x + indent + mw, y, width - indent - mw, tags)
            elif block.kind == "quote":
                top = y
                y = self.draw_runs(canvas, inline_runs(block.text), x + px(14), y, width - px(14), tags, tone=MUTED)
                canvas.create_rectangle(x, top, x + px(3), y - px(5), fill=BORDER, outline="", tags=tags)
            else:
                y = self.draw_runs(canvas, inline_runs(block.text), x, y, width, tags)
            first = False
        return y

    # ------------------------------------------------------------ events

    def on_main_resize(self, event: Event) -> None:
        if event.widget is not self.main:
            return
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        self._glow_job = self.root.after(80, self.regen_glow)

    def regen_glow(self) -> None:
        self._glow_job = None
        if not self.alive:
            return
        w, h = self.main.winfo_width(), self.main.winfo_height()
        if w < 20 or h < 20 or (w, h) == self._glow_size:
            return
        self._glow_size = (w, h)
        self.glow_img = glow_image(self.root, w, h, max(3, self.px(4)))
        self.repaint_glow()
        self.draw_welcome()

    def repaint_glow(self) -> None:
        if not self.alive:
            return
        for canvas in self.glow_canvases:
            canvas.paint_glow()

    def on_messages_resize(self) -> None:
        at_bottom = self.msg_canvas.yview()[1] > 0.97
        self.relayout(0)
        if at_bottom:
            self.scroll_to_bottom()

    def relayout(self, start: int = 0) -> None:
        c, px = self.msg_canvas, self.px
        w = c.winfo_width()
        if w < px(100):
            return
        col_w = min(w - px(48), px(820))
        col_x = (w - col_w) / 2
        if start > 0:
            prev = self.messages[start - 1]
            y = prev.y + prev.height + px(15)
        else:
            y = px(20)
        for msg in self.messages[start:]:
            y += msg.layout(y, col_x, col_w) + px(15)
        c.configure(scrollregion=(0, 0, w, max(y + px(10), c.winfo_height())))
        c.tag_raise("thumb")

    def refresh_message(self, msg: ChatMessage) -> None:
        if msg not in self.messages:
            return
        at_bottom = self.msg_canvas.yview()[1] > 0.97
        self.relayout(self.messages.index(msg))
        if at_bottom:
            self.scroll_to_bottom()

    def on_yscroll(self, first: str, last: str) -> None:
        c, px = self.msg_canvas, self.px
        c.paint_glow()
        lo, hi = float(first), float(last)
        if hi - lo >= 0.999:
            c.itemconfigure("thumb", state="hidden")
            return
        h, top = c.winfo_height(), c.canvasy(0)
        x2 = c.canvasx(0) + c.winfo_width() - px(4)
        c.coords("thumb", *rounded_points(x2 - px(8), top + lo * h, x2, top + hi * h, px(4)))
        c.itemconfigure("thumb", state="normal")

    def thumb_press(self, event: Event) -> None:
        self._drag = (event.y, self.msg_canvas.yview()[0])

    def thumb_drag(self, event: Event) -> None:
        y0, first = self._drag
        self.msg_canvas.yview_moveto(first + (event.y - y0) / max(1, self.msg_canvas.winfo_height()))

    def hover_copy(self, on: bool) -> None:
        self.msg_canvas.itemconfigure("current", fill=CREAM if on else MUTED)
        self.msg_canvas.configure(cursor="hand2" if on else "")

    def copy_current(self) -> None:
        c = self.msg_canvas
        tags = c.gettags("current")
        msg = next((m for m in self.messages if m.tag in tags), None)
        if msg is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(msg.text)
        item = c.find_withtag("current")[0]
        c.itemconfigure(item, text="Copied!")
        self.root.after(1500, lambda: self.alive and c.find_withtag(item) and c.itemconfigure(item, text="Copy"))

    def hover_send(self, on: bool) -> None:
        self.send_hover = on
        self.input_canvas.configure(cursor="hand2" if on else "")
        self.style_send()

    def set_input_focus(self, focused: bool) -> None:
        self.input_focus = focused
        self.redraw_input()

    def on_wheel(self, event: Event) -> None:
        if event.num == 4:
            delta = -2
        elif event.num == 5:
            delta = 2
        elif sys.platform == "darwin":
            delta = -event.delta
        else:
            delta = -2 * int(event.delta / 120)
        if not delta:
            return
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is self.sidebar:
            self.sidebar_offset += 1 if delta > 0 else -1
            self.draw_sidebar()
        elif widget is self.msg_canvas and self.msg_canvas.winfo_ismapped():
            self.msg_canvas.yview_scroll(delta, "units")

    def on_enter(self, event: Event) -> str | None:
        if event.state & 0x0001:
            return None
        self.on_send_click()
        return "break"

    def on_send_click(self) -> None:
        if self.generating:
            self.stop()
        else:
            self.send()

    def get_input(self) -> str:
        return self.entry.get("1.0", "end").strip()

    # ------------------------------------------------------------ conversations

    def show_messages(self, visible: bool) -> None:
        if visible:
            self.welcome.pack_forget()
            self.msg_canvas.pack(fill="both", expand=True)
        else:
            self.msg_canvas.pack_forget()
            self.welcome.pack(fill="both", expand=True)
        self.root.after_idle(self.repaint_glow)

    def clear_messages(self) -> None:
        for msg in self.messages:
            self.msg_canvas.delete(msg.tag)
        self.messages = []
        self.msg_canvas.yview_moveto(0)

    def open_conversation(self, conv_id: str | None) -> None:
        if self.generating:
            self.stop()
        conv = self.store.get(conv_id)
        self.current_id = conv["id"] if conv else None
        self.clear_messages()
        if conv and conv["messages"]:
            self.messages = [ChatMessage(self, m["role"], m["content"]) for m in conv["messages"]]
            self.show_messages(True)
            self.relayout(0)
            self.scroll_to_bottom()
        else:
            self.show_messages(False)
        self.draw_sidebar()

    def new_chat(self) -> None:
        self.open_conversation(None)
        self.entry.focus_set()

    def rename_chat(self, conv: dict) -> None:
        title = simpledialog.askstring("Rename chat", "New name:", initialvalue=conv["title"], parent=self.root)
        if title and title.strip():
            self.store.rename(conv, title.strip()[:80])
            self.draw_sidebar()

    def delete_chat(self, conv: dict) -> None:
        if messagebox.askyesno("Delete chat", f"Delete \"{conv['title']}\"?", parent=self.root):
            self.store.delete(conv)
            self.open_conversation(None if conv["id"] == self.current_id else self.current_id)

    def toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible
        self.store.set_setting("sidebar", "1" if self.sidebar_visible else "0")
        if self.sidebar_visible:
            self.sidebar.pack(side="left", fill="y", before=self.main)
        else:
            self.sidebar.pack_forget()

    def scroll_to_bottom(self) -> None:
        self.msg_canvas.yview_moveto(1.0)

    def open_settings(self) -> None:
        if self.settings_win and self.settings_win.top.winfo_exists():
            self.settings_win.top.lift()
            self.settings_win.top.focus_force()
        else:
            self.settings_win = SettingsDialog(self)

    def set_brain(self, brain: LocalBrain | OpenAIBrain) -> None:
        self.brain = brain
        self.draw_header()
        self.draw_sidebar()

    # ------------------------------------------------------------ generation

    def send(self, text: str | None = None) -> None:
        text = (text or self.get_input()).strip()
        if not text or self.generating:
            return
        self.entry.delete("1.0", "end")
        conv = self.store.get(self.current_id)
        if conv is None:
            conv = self.store.new_conversation(text[:40] + ("\u2026" if len(text) > 40 else ""))
            self.current_id = conv["id"]
        self.store.add_message(conv, "user", text)
        self.show_messages(True)

        self.reply_msg = ChatMessage(self, "assistant", thinking=True)
        self.messages += [ChatMessage(self, "user", text), self.reply_msg]
        self.relayout(len(self.messages) - 2)
        self.scroll_to_bottom()
        self.root.after(350, self.reply_msg.tick)

        self.generating = True
        self.reply_text = ""
        self.reply_conv = conv
        self.stop_event = threading.Event()
        self.redraw_input()
        self.draw_sidebar()
        history = [dict(m) for m in conv["messages"]]
        threading.Thread(target=self.generate, args=(self.brain, history, self.stop_event), daemon=True).start()

    def generate(self, brain: LocalBrain | OpenAIBrain, history: list[dict], stop_event: threading.Event) -> None:
        try:
            for token in brain.stream(history):
                if stop_event.is_set():
                    break
                self.tokens.put((stop_event, "token", token))
        except Exception as exc:  # noqa: BLE001 - show API/network errors in the chat instead of crashing
            self.tokens.put((stop_event, "error", friendly_error(exc)))
        self.tokens.put((stop_event, "done", ""))

    def stop(self) -> None:
        if self.generating:
            self.stop_event.set()
            self.finish_reply(stopped=True)

    def finish_reply(self, error: str | None = None, stopped: bool = False) -> None:
        self.generating = False
        text = self.reply_text.strip()
        if text:
            self.store.add_message(self.reply_conv, "assistant", text)
        msg = self.reply_msg
        if msg:
            msg.thinking, msg.finished, msg.text = False, True, text
            msg.error = error or ("Response stopped." if stopped and not text else None)
            self.refresh_message(msg)
        self.redraw_input()
        self.draw_sidebar()

    def poll_tokens(self) -> None:
        if not self.alive:
            return
        changed = False
        try:
            while True:
                event, kind, value = self.tokens.get_nowait()
                if event is not self.stop_event or not self.generating:
                    continue
                if kind == "token":
                    self.reply_text += value
                    changed = True
                elif kind == "error":
                    self.finish_reply(error=value)
                else:
                    self.finish_reply()
        except queue.Empty:
            pass
        if changed and self.generating and self.reply_msg:
            self.reply_msg.thinking = False
            self.reply_msg.text = self.reply_text
            self.refresh_message(self.reply_msg)
        self.root.after(30, self.poll_tokens)


# =========================================================== terminal mode

def cli_login(users: UserStore) -> str | None:
    """Log in or sign up in the terminal. Returns the user key, or None if the user gave up."""
    print(f"{APP_NAME} - {WELCOME_TEXT}\n")
    try:
        while True:
            name = input("Username (new names create an account): ").strip()
            if not name:
                continue
            if users.get(name):
                for _ in range(3):
                    if users.verify(name, getpass.getpass("Password: ")):
                        return users.key(name)
                    print("Wrong password.")
                return None
            if input(f"No account '{name}'. Create it? (y/n): ").strip().lower() not in {"y", "yes"}:
                continue
            try:
                key = users.create(name, getpass.getpass("Choose a password: "), getpass.getpass("Repeat it: "))
            except ValueError as exc:
                print(exc)
                continue
            print(f"Account created. Welcome, {name}!")
            return key
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def run_cli(users: UserStore) -> None:
    key = cli_login(users)
    if key is None:
        return
    users.signed_in(key, remember=False)
    store = users.chat_store(key)
    brain = build_brain(store)
    print(f"\nSigned in as {users.display_name(key)} ({brain.name}). Type 'quit' to exit.\n")
    conv = None
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if conv is None:
            conv = store.new_conversation(text[:40])
        store.add_message(conv, "user", text)
        print(f"{APP_NAME}: ", end="", flush=True)
        parts = []
        try:
            for token in brain.stream(conv["messages"]):
                parts.append(token)
                print(token, end="", flush=True)
        except Exception as exc:  # noqa: BLE001 - show API/network errors instead of crashing
            print(f"[{friendly_error(exc)}]", end="")
        print("\n")
        if parts:
            store.add_message(conv, "assistant", "".join(parts).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} - AI chatbot")
    parser.add_argument("--cli", action="store_true", help="chat in the terminal instead of the desktop app")
    args = parser.parse_args()

    users = UserStore(DATA_DIR)
    if args.cli or tk is None:
        if tk is None and not args.cli:
            print("Tkinter is not installed - starting terminal mode.\n")
        run_cli(users)
        return

    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    scale = ui_scale(root)
    root.title(APP_NAME)
    root.configure(bg=BG)
    root.geometry(f"{int(1180 * scale)}x{int(780 * scale)}")
    root.minsize(int(760 * scale), int(560 * scale))

    def show_login() -> None:
        LoginScreen(root, users, open_app)

    def open_app(user_key: str) -> None:
        HaniaAI(root, users, user_key, on_logout=show_login)

    remembered = users.remembered_user()
    if remembered:
        open_app(remembered)
    else:
        show_login()
    root.mainloop()


if __name__ == "__main__":
    main()
