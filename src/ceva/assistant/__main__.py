"""Serve the chat and the MCP server: ``python -m ceva.assistant``.

The dataset's two files and the dashboard's address come from the options or from the environment
(``CEVA_RATES``, ``CEVA_SHORELINES``, ``CEVA_SITE``, ``CEVA_CACHE``); the language models' keys from the
environment only (see ``ceva.assistant.chat``).
"""
import argparse
import logging
import os

from ceva.assistant.data import SITE_URL, Dataset


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="python -m ceva.assistant", description=__doc__.splitlines()[0])
    p.add_argument("--rates", default=env("CEVA_RATES"), help="the dataset's rates file (path or address)")
    p.add_argument("--shorelines", default=env("CEVA_SHORELINES"), help="its yearly shorelines file")
    p.add_argument("--site", default=env("CEVA_SITE", SITE_URL), help="the dashboard (folder or address)")
    p.add_argument("--cache", default=env("CEVA_CACHE", "ceva-cache"), help="where downloaded files are kept")
    p.add_argument("--origin", action="append", help="a site allowed to call /chat (default: the dashboard's)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(env("PORT", "7860")))
    args = p.parse_args()
    if not args.rates or not args.shorelines:
        p.error("the rates and shorelines files are needed (--rates, --shorelines or CEVA_RATES, CEVA_SHORELINES)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from ceva.assistant.chat import create_app

    ds = Dataset.load(args.rates, args.shorelines, args.site, args.cache)
    logging.getLogger("ceva").info("%d transects loaded", len(ds.rates))
    local = args.host in ("127.0.0.1", "localhost")
    uvicorn.run(create_app(ds, args.origin, local=local), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
