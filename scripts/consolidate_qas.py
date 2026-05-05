"""Walk every outputs/* dir, gather every QA written to rounds.jsonl,
dedupe by question text, sort by gap, and write to a single training file.

Usage: python scripts/consolidate_qas.py /root/autodl-fs/autodata/outputs
"""

import json
import sys
from pathlib import Path


def main(root_str: str) -> int:
    root = Path(root_str)
    out: list[tuple[float, float, float, dict, str]] = []
    seen: set[str] = set()
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rounds = run_dir / "rounds.jsonl"
        if not rounds.exists():
            continue
        for line in rounds.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            qa = d.get("qa")
            if not qa or not qa.get("question"):
                continue
            key = qa["question"][:200]
            if key in seen:
                continue
            seen.add(key)
            acc = d.get("acceptance") or {}
            gap = float(acc.get("gap") or 0.0)
            strong = float(acc.get("strong_avg") or 0.0)
            weak = float(acc.get("weak_avg") or 0.0)
            out.append((gap, strong, weak, qa, run_dir.name))
    out.sort(key=lambda t: -t[0])
    print(f"unique QAs across all runs: {len(out)}")
    print("top 5 by gap:")
    for g, s, w, qa, run in out[:5]:
        q = (qa.get("question") or "")[:80]
        print(f"  gap={g:.2f} strong={s:.2f} weak={w:.2f} run={run} q={q!r}")
    dest = root / "consolidated" / "training_qas.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for _, _, _, qa, _ in out:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs"))
