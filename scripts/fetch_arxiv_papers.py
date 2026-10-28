#!/usr/bin/env python
"""Build a small paper corpus from arXiv HTML renderings.

    python scripts/fetch_arxiv_papers.py --out data/papers-arxiv --n 5

The paper uses S2ORC CS papers; this is a stand-in for smoke and gate runs.
It lists the newest submissions in the given categories, fetches each
paper's HTML page, converts it (autodata.data_module.arxiv_html), and keeps
papers within the length window. Papers without an HTML version are skipped.
Each kept paper is logged to ``_manifest.jsonl`` with its arXiv version and a
hash of the markdown.

arXiv asks API clients to wait 3 seconds between requests; this script does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import httpx

from autodata.data_module.arxiv_html import NoFullTextError, html_to_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"
HTML = "https://arxiv.org/html/{}"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
PAGE_SIZE = 50


def _list_entries(client: httpx.Client, query: str, start: int) -> list[dict[str, str]]:
    params = {"search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
              "start": start, "max_results": PAGE_SIZE}
    r = client.get(API, params=params)
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ATOM):
        abs_url = e.findtext("a:id", "", ATOM)            # http://arxiv.org/abs/2610.34589v1
        title = " ".join(e.findtext("a:title", "", ATOM).split())
        out.append({"versioned_id": abs_url.rsplit("/abs/", 1)[-1], "title": title})
    return out


def _fetch_one(client: httpx.Client, versioned_id: str, title: str) -> str | None:
    r = client.get(HTML.format(versioned_id))
    if r.status_code != 200:
        logger.info("skip %s: HTML status %d", versioned_id, r.status_code)
        return None
    try:
        return html_to_markdown(r.text, title=title)
    except NoFullTextError:
        logger.info("skip %s: no LaTeXML article", versioned_id)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5, help="papers to keep")
    ap.add_argument("--categories", default="cs.CL,cs.LG,cs.AI,cs.CV")
    ap.add_argument("--start", type=int, default=0, help="offset into the newest-first listing")
    ap.add_argument("--min-chars", type=int, default=20_000)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--max-scan", type=int, default=200, help="give up after this many entries")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between arXiv requests")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    query = " OR ".join(f"cat:{c.strip()}" for c in args.categories.split(","))
    kept = scanned = 0
    with httpx.Client(timeout=60, follow_redirects=True,
                      headers={"User-Agent": "autodata-corpus/0.1"}) as client, \
            (args.out / "_manifest.jsonl").open("a", encoding="utf-8") as manifest:
        start = args.start
        while kept < args.n and scanned < args.max_scan:
            entries = _list_entries(client, query, start)
            time.sleep(args.delay)
            if not entries:
                break
            start += len(entries)
            for e in entries:
                if kept >= args.n or scanned >= args.max_scan:
                    break
                scanned += 1
                paper_id = e["versioned_id"].rsplit("v", 1)[0].replace("/", "_")
                path = args.out / f"{paper_id}.md"
                if path.exists():
                    continue
                md = _fetch_one(client, e["versioned_id"], e["title"])
                time.sleep(args.delay)
                if md is None:
                    continue
                if not args.min_chars <= len(md) <= args.max_chars:
                    logger.info("skip %s: %d chars outside window", paper_id, len(md))
                    continue
                path.write_text(md, encoding="utf-8")
                manifest.write(json.dumps({
                    "paper_id": paper_id, "arxiv_version": e["versioned_id"], "title": e["title"],
                    "chars": len(md), "sha256_12": hashlib.sha256(md.encode()).hexdigest()[:12],
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }) + "\n")
                kept += 1
                logger.info("kept %s (%d chars): %s", paper_id, len(md), e["title"][:80])
    logger.info("kept %d of %d scanned into %s", kept, scanned, args.out)


if __name__ == "__main__":
    main()
