# The CEVA chat and MCP server on a Hugging Face Space (Docker, free CPU).
# The dataset's two files are the Space's own data/ folder, or the addresses in the Space's variables
# CEVA_RATES and CEVA_SHORELINES (downloaded when the Space starts); the language models' keys are the
# Space's secrets (GROQ_API_KEY, MISTRAL_API_KEY).
FROM python:3.12-slim

RUN pip install --no-cache-dir "numpy>=2" "pandas>=2.2" pyarrow geopandas "shapely>=2" scipy httpx \
        "mcp>=2.3,<3" "openai>=1.40" "uvicorn>=0.30" starlette

# CEVA itself, without its processing dependencies (the assistant needs none of them)
ARG CEVA_REF=main
RUN pip install --no-cache-dir --no-deps "https://github.com/clacrouts/ceva/archive/${CEVA_REF}.tar.gz"

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user CEVA_CACHE=/home/user/cache PORT=7860 \
    CEVA_RATES=/home/user/space/data/ceva-shorelines_rates.parquet \
    CEVA_SHORELINES=/home/user/space/data/ceva-shorelines_shorelines.parquet
WORKDIR /home/user
COPY --chown=user . /home/user/space

EXPOSE 7860
CMD ["python", "-m", "ceva.assistant", "--host", "0.0.0.0", "--port", "7860"]
