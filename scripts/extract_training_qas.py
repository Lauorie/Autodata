"""Extract training QAs for GRPO from a run.

Default behaviour: emit every ACCEPTED QA pair from ``accepted_qa.jsonl``.

Fallback (``--include-best-rejected N``): if ``accepted_qa.jsonl`` has fewer
than ``--min-accepted`` rows, we top up the training set with the highest-gap
*rejected* rounds from ``rounds.jsonl``. This is a pragmatic compromise — a
QA pair with gap=0.18 (just under threshold) still carries meaningful
weak/strong signal for GRPO. The output JSONL is shape-compatible with the
GRPO trainer's expected format.

Usage:
    python scripts/extract_training_qas.py \\
        --run-dir /root/autodl-fs/autodata/outputs/run20 \\
        --out /root/autodl-fs/autodata/outputs/run20/grpo_input.jsonl \\
        --min-accepted 10 --include-best-rejected 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_accepted(run_dir: Path) -> list[dict]:
    p = run_dir / "accepted_qa.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def best_rejected(run_dir: Path, n: int) -> list[dict]:
    """Pull the top-``n`` rejected rounds (by gap) from rounds.jsonl.

    Returns QA dicts shaped like accepted_qa.jsonl entries.
    """
    p = run_dir / "rounds.jsonl"
    if not p.exists():
        return []
    candidates = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("accepted"):
            continue                       # already in the accepted set
        qa = row.get("qa")
        if not qa:                         # error/QV-rejected rounds have no qa
            continue
        acc = row.get("acceptance") or {}
        gap = float(acc.get("gap", 0.0))
        candidates.append((gap, qa))
    candidates.sort(key=lambda t: -t[0])
    return [qa for _, qa in candidates[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-accepted", type=int, default=10,
                    help="If accepted_qa.jsonl has fewer than this many rows, top up with best-rejected.")
    ap.add_argument("--include-best-rejected", type=int, default=0,
                    help="Max number of rejected rounds to add (0 = never, even if accepted is empty).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    accepted = load_accepted(run_dir)
    logger.info("accepted: %d", len(accepted))

    chosen = list(accepted)
    if len(accepted) < args.min_accepted and args.include_best_rejected > 0:
        topup_n = max(0, args.include_best_rejected - len(accepted))
        topup = best_rejected(run_dir, topup_n)
        logger.info("topping up with %d best-rejected rounds (gap-sorted)", len(topup))
        chosen.extend(topup)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for qa in chosen:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows -> %s", len(chosen), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
