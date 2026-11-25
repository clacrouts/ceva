"""The chat service the dashboard talks to, and the web app that serves it with the MCP server.

``POST /chat`` takes the conversation so far, the site's language and what the map shows; a language model
answers, calling the dataset's tools (in this process) as it needs, and the answer comes back as text with
links to the map. The model runs on free tiers: Groq first, Mistral when Groq refuses or fails. Their keys
come from the environment (``GROQ_API_KEY``, ``MISTRAL_API_KEY``), never from a request. Each visitor may ask
a few questions a minute, each question may run a few rounds of tools, and nothing is stored.

``/mcp`` is the MCP server itself (streamable HTTP), ``/health`` says whether the data are loaded.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ceva.assistant.data import Dataset
from ceva.assistant.server import build

log = logging.getLogger(__name__)

# provider: (OpenAI-compatible address, key variable, model variable, default model). Groq's free tier allows
# about 8,000 tokens a minute for its larger models: a question with one round of tools takes most of them,
# so the tools are described tersely and Mistral takes over when Groq refuses.
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest"),
}
ORIGINS = ["https://clacrouts.github.io"]
MAX_ROUNDS = 6                 # rounds of tool calls for one question
MAX_CALLS = 4                  # tool calls answered in one round
MAX_MESSAGES = 12
MAX_CHARS = 2000               # of one message
MAX_TOOL_CHARS = 12000         # of one tool result, as the model sees it
MAX_ANSWER_TOKENS = 1200       # of one reply of the model, its reasoning included
PER_VISITOR = (6, 60.0)        # questions per period (s)
ALL_VISITORS = (60, 60.0)
LANGUAGES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
             "pt": "Portuguese", "ro": "Romanian", "ar": "Arabic", "zh": "Chinese", "hi": "Hindi"}

SYSTEM = """You are the assistant of the CEVA dashboard, a map of satellite-derived shorelines of the sand and \
gravel coasts of Europe and the Mediterranean since 1984, with a coastal vulnerability index.

Rules:
- Answer only from the results of your tools. If they cannot answer, say so; never guess numbers.
- Rates are in m/yr, positive seaward (accretion, advance), negative landward (erosion, retreat). Give the 95 % \
range or say whether the trend is significant, and the years it covers.
- For a place named by the visitor, call transects_near with its longitude and latitude. If you do not know \
where it is, ask.
- Link every transect or stretch you mention with its map link, as a Markdown link.
- Mention flags (trend not significant, low precision, few years) when they apply.
- The vulnerability classes compare coasts with each other: they are not a hazard or risk assessment.
- No forecasts, and no advice on planning, safety, property, insurance or law: this is not an official \
erosion map, national and regional observatories are the reference.
- Be brief: a few sentences or a short list. Answer in {language}.
{context}"""


def compact(schema):
    """A tool's argument schema without what the model does not need (titles; "or null" of optional arguments)."""
    if isinstance(schema, list):
        return [compact(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: compact(v) for k, v in schema.items() if k != "title"}
    options = out.get("anyOf")
    if isinstance(options, list) and len(options) == 2 and {"type": "null"} in options and out.get("default") is None:
        out.pop("anyOf")
        out.pop("default", None)
        out.update(next(o for o in options if o != {"type": "null"}))
    return out


def model_options(model: str) -> dict:
    """Request options a model needs: gpt-oss models reason before answering, briefly is enough here."""
    return {"reasoning_effort": "low"} if "gpt-oss" in model else {}


class Limiter:
    """At most ``n`` events per ``period`` seconds, for each key and for all keys together."""

    def __init__(self, each: tuple[int, float] = PER_VISITOR, total: tuple[int, float] = ALL_VISITORS):
        self.each, self.total = each, total
        self.events: dict[str, deque] = defaultdict(deque)
        self.all: deque = deque()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        mine = self.events[key]
        for queue, (n, period) in ((mine, self.each), (self.all, self.total)):
            while queue and now - queue[0] > period:
                queue.popleft()
            if len(queue) >= n:
                return False
        mine.append(now)
        self.all.append(now)
        return True


def context_text(view: dict | None, selected: dict | None) -> str:
    """What the visitor sees, so that "here" and "this beach" mean something."""
    lines = []
    if isinstance(view, dict) and all(isinstance(view.get(k), int | float) for k in ("lon", "lat")):
        lines.append(f"The map is centred on longitude {view['lon']:.4f}, latitude {view['lat']:.4f}"
                     + (f", zoom {view['zoom']:.0f}" if isinstance(view.get("zoom"), int | float) else "") + ".")
    if isinstance(selected, dict) and isinstance(selected.get("id"), str) and len(selected["id"]) < 40:
        lines.append(f"The visitor has opened transect {selected['id']}: \"this beach\" or \"here\" means it.")
    return ("\nContext: " + " ".join(lines)) if lines else ""


def clean_messages(messages) -> list[dict] | None:
    """The conversation, checked: user and assistant turns of text only, the last one the user's."""
    if not isinstance(messages, list) or not messages:
        return None
    out = []
    for m in messages[-MAX_MESSAGES:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return None
        if not isinstance(m.get("content"), str):
            return None
        out.append({"role": m["role"], "content": m["content"][:MAX_CHARS]})
    return out if out[-1]["role"] == "user" and out[-1]["content"].strip() else None


class Chat:
    def __init__(self, server, providers: list[str] | None = None):
        self.server = server
        names = providers or [p.strip() for p in os.environ.get("CEVA_PROVIDERS", "groq,mistral").split(",")]
        self.providers = [p for p in names if p in PROVIDERS and os.environ.get(PROVIDERS[p][1])]
        self.limiter = Limiter()
        self._specs = None

    @staticmethod
    def model(provider: str) -> str:
        _, _, variable, default = PROVIDERS[provider]
        return os.environ.get(variable, default)

    async def specs(self) -> list[dict]:
        if self._specs is None:
            self._specs = [{"type": "function", "function": {
                "name": t.name, "description": " ".join((t.description or "").split()),
                "parameters": compact(t.input_schema)}} for t in await self.server.list_tools()]
        return self._specs

    async def call_tool(self, name: str, arguments: str | None) -> str:
        if name not in {s["function"]["name"] for s in await self.specs()}:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments are not valid JSON"})
        try:
            result = await self.server.call_tool(name, args if isinstance(args, dict) else {})
        except Exception as error:                       # a wrong argument: the model may try again
            return json.dumps({"error": str(error)[:300]})
        payload = getattr(result, "structured_content", None)
        if payload is not None:
            text = json.dumps(payload.get("result", payload) if isinstance(payload, dict) and len(payload) == 1
                              else payload, ensure_ascii=False, default=str)
        else:
            text = "\n".join(getattr(c, "text", "") for c in getattr(result, "content", []) or [])
        return text if len(text) <= MAX_TOOL_CHARS else text[:MAX_TOOL_CHARS] + " ... (cut)"

    async def run(self, provider: str, messages: list[dict]) -> str:
        """The model's answer, after the tool rounds it asks for."""
        from openai import AsyncOpenAI

        url, key, _, _ = PROVIDERS[provider]
        url = os.environ.get(f"{provider.upper()}_BASE_URL", url)          # another compatible endpoint
        client = AsyncOpenAI(base_url=url, api_key=os.environ[key], timeout=60,
                             max_retries=0)                                 # a refusal: the next provider
        model = self.model(provider)
        messages = list(messages)
        for round_ in range(MAX_ROUNDS + 1):
            extra = {"tools": await self.specs(), "tool_choice": "auto"} if round_ < MAX_ROUNDS else {}
            response = await client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                                            max_tokens=MAX_ANSWER_TOKENS, **model_options(model),
                                                            **extra)
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or ""
            calls = message.tool_calls
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                              "arguments": c.function.arguments}} for c in calls]})
            for k, c in enumerate(calls):
                result = (await self.call_tool(c.function.name, c.function.arguments) if k < MAX_CALLS
                          else json.dumps({"error": "too many tool calls at once: ask again with fewer"}))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        return ""

    async def handle(self, request: Request) -> JSONResponse:
        visitor = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?"))
        if not self.limiter.allow(visitor.split(",")[0].strip()):
            return JSONResponse({"error": "busy"}, status_code=429)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "bad request"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "bad request"}, status_code=400)
        messages = clean_messages(body.get("messages"))
        if messages is None:
            return JSONResponse({"error": "bad request"}, status_code=400)
        language = LANGUAGES.get(body.get("lang"), "English")
        system = SYSTEM.format(language=language, context=context_text(body.get("view"), body.get("selected")))
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        for provider in self.providers:
            try:
                answer = await self.run(provider, [{"role": "system", "content": system}] + messages)
            except (RateLimitError, APIConnectionError, APITimeoutError) as error:
                log.warning("%s: %s", provider, type(error).__name__)
                continue
            except APIStatusError as error:
                log.warning("%s: status %s", provider, error.status_code)
                continue
            if answer.strip():
                return JSONResponse({"answer": answer, "provider": provider})
        return JSONResponse({"error": "unavailable"}, status_code=503)


def create_app(ds: Dataset, origins: list[str] | None = None, local: bool = False) -> Starlette:
    """The web app: ``/chat``, ``/health`` and the MCP server at ``/mcp``. ``local`` keeps the MCP server's
    protection against DNS rebinding (a server on one's own machine); a public one answers any host."""
    from mcp.server.transport_security import TransportSecuritySettings

    server = build(ds)
    chat = Chat(server)

    @server.custom_route("/chat", methods=["POST"])
    async def _chat(request: Request) -> JSONResponse:
        return await chat.handle(request)

    @server.custom_route("/health", methods=["GET"])
    async def _health(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "transects": int(len(ds.rates)),
                             "models": {p: chat.model(p) for p in chat.providers},
                             "version": ds.manifest.get("ceva_version")})

    security = TransportSecuritySettings(enable_dns_rebinding_protection=local,
                                         allowed_hosts=["127.0.0.1:*", "localhost:*"] if local else [])
    app = server.streamable_http_app(stateless_http=True, json_response=True, transport_security=security)
    app.add_middleware(CORSMiddleware, allow_origins=origins or ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["content-type", "mcp-protocol-version"], max_age=3600)
    return app
