---
title: CEVA chat
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
short_description: Questions to the CEVA shoreline dataset, and its MCP server
---

# CEVA chat

The service behind the "Ask the data" panel of the [CEVA dashboard](https://clacrouts.github.io/ceva-dashboard/),
and an MCP server any MCP client can use to question the dataset
([doi:10.5281/zenodo.24493619](https://doi.org/10.5281/zenodo.24493619)) directly. Code:
[github.com/clacrouts/ceva](https://github.com/clacrouts/ceva) (`ceva.assistant`).

| Address | What |
|---|---|
| `/mcp` | the MCP server (streamable HTTP): ten read-only tools over the dataset |
| `/chat` | the dashboard's chat (POST, from the dashboard only) |
| `/health` | whether the data are loaded, and which language models are set |

## Setting it up

1. Create a Space (Docker, blank, free CPU) and add this folder's `Dockerfile` and `README.md` to it, with
   the dataset's two files (`ceva export --products rates,shorelines`) in a `data/` folder:
   `data/ceva-shorelines_rates.parquet` and `data/ceva-shorelines_shorelines.parquet`.
2. In the Space's settings:
   - **Variables** (optional): `CEVA_RATES` and `CEVA_SHORELINES`, to read the two files from elsewhere
     instead of `data/`, e.g. Zenodo
     (`https://zenodo.org/api/records/<record>/files/ceva-shorelines_rates.parquet/content`, and the same for
     `ceva-shorelines_shorelines.parquet`; downloaded at each start); `GROQ_MODEL`, `MISTRAL_MODEL` and
     `CEVA_PROVIDERS` (default `groq,mistral`).
   - **Secrets:** `GROQ_API_KEY` and `MISTRAL_API_KEY` (free tiers). Without them, `/chat` answers
     "unavailable" and `/mcp` still works.
   - `CEVA_REF` (build argument, default `main`): the CEVA version to install, e.g. a release tag. A new
     version of CEVA on the same branch needs a "Factory rebuild" (Settings) to be installed.
3. When `/health` answers, add to the dashboard's `index.html`:
   `<meta name="ceva-chat" content="https://<owner>-<space>.hf.space">`.

The models are `openai/gpt-oss-120b` on Groq and `mistral-small-latest` on Mistral unless the variables say
otherwise. Groq's free tier allows about 8,000 tokens a minute, roughly one question a minute; Mistral answers
when Groq refuses. The Space sleeps after two days without visitors; the first question after that waits for
it to start and load the data (about a minute).

## Privacy

Questions go to this service and to the language model's provider (Groq, then Mistral if Groq is busy);
the service stores nothing and logs no question. Each visitor may ask a few questions a minute.
