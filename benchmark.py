"""
Next-action prediction benchmark for Rocketium canvas editor sessions.

Usage:
  python benchmark.py [--window N] [--samples K] [--session FILE.jsonl]

Defaults: window=10, samples=100, session=DSNsZv85yqoI-1158.jsonl
"""

import json
import os
import random
import sys
import argparse
from pathlib import Path

from openai import OpenAI

DATA_DIR = Path(__file__).parent

# --- Clean up action types ---
ACTION_PREFIX = "@@canvas-editor/"

ALL_ACTION_TYPES = [
    "UPDATE_ELEMENT_SAGA",
    "SET_LOCAL_STATE",
    "ALIGN_ELEMENTS",
    "REPLACE_MEDIA",
    "SET_AI_SUGGESTED_LAYER_NAMES",
    "APPLY_STYLE",
    "IMPORT_SPREADSHEET_SUCCESS",
    "DELETE_VARIANT",
]


def short_action(action_type: str) -> str:
    return action_type.replace(ACTION_PREFIX, "")


# --- Codex-improved describe_action + burst compressor ---

def _prop_name(path: str) -> str:
    return path.split(".")[-1] if path else "unknown"


def _obj_id(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else path


_NUMERIC_PROPS = {"left", "top", "width", "height", "fontSize", "lineHeight",
                  "imageScale", "imageTop", "imageLeft"}
_FLAG_PROPS = {"visible", "isDeleted"}
_LAYOUT_PROPS = {"objectPosition", "objectFit"}
_SIZE_PROPS = {"imageWidth", "imageHeight"}

_BURST_PROPS = _NUMERIC_PROPS  # properties eligible for burst compression


def describe_action(row: dict) -> str:
    """Rich single-row description encoding what changed, not just action name."""
    action = short_action(row["action_type"])
    try:
        delta = json.loads(row["delta_json"])
    except Exception:
        return action

    key = delta.get("key", "")
    key_parts = [p for p in key.split(".") if not (len(p) > 20 and "-" in p)]
    key_short = ".".join(key_parts[-2:]) if key_parts else key

    diffs = delta.get("diff", [])
    objects: set = set()
    props = []

    for d in diffs[:4]:
        p = d.get("path", [])
        path_str = ".".join(str(x) for x in p)
        prop = _prop_name(path_str)
        obj = _obj_id(path_str)
        objects.add(obj.split(".")[-1][:8])  # short object hint
        lhs = d.get("lhs", "")
        rhs = d.get("rhs", "")

        if prop in _NUMERIC_PROPS:
            try:
                delta_val = float(rhs) - float(lhs)
                direction = "↑" if delta_val > 0 else "↓"
                props.append(f"{prop}:{lhs}→{rhs}({direction}{abs(delta_val):.2g})")
            except Exception:
                props.append(f"{prop}:{lhs}→{rhs}")
        elif prop in _FLAG_PROPS | _LAYOUT_PROPS:
            props.append(f"{prop}:{lhs}→{rhs}")
        elif prop == "src":
            props.append("src:changed")
        elif prop in _SIZE_PROPS:
            props.append(f"{prop}:{lhs}→{rhs}")
        else:
            props.append(f"{prop}:{str(lhs)[:12]}→{str(rhs)[:12]}")

    if len(diffs) > 4:
        props.append(f"+{len(diffs)-4}more")

    obj_hint = ",".join(sorted(objects)[:2])
    prop_hint = "; ".join(props)
    return f"{action} [{key_short}] obj=[{obj_hint}] {prop_hint}"


def describe_actions_compressed(rows: list[dict]) -> list[str]:
    """
    Burst compressor: collapse 3+ consecutive same-action/same-property nudges
    (e.g. dragging left) into a single BURST summary line.
    """
    out = []
    i = 0
    while i < len(rows):
        row = rows[i]
        try:
            delta = json.loads(row["delta_json"])
        except Exception:
            out.append(describe_action(row))
            i += 1
            continue

        diffs = delta.get("diff", [])
        if not diffs:
            out.append(describe_action(row))
            i += 1
            continue

        action = short_action(row["action_type"])
        key = delta.get("key", "")
        first_path = ".".join(str(x) for x in diffs[0].get("path", []))
        prop = _prop_name(first_path)
        obj = _obj_id(first_path)

        if prop not in _BURST_PROPS:
            out.append(describe_action(row))
            i += 1
            continue

        # Scan forward for burst
        j = i + 1
        first_lhs = diffs[0].get("lhs")
        last_rhs = diffs[0].get("rhs")

        while j < len(rows):
            try:
                d2 = json.loads(rows[j]["delta_json"])
            except Exception:
                break
            d2diffs = d2.get("diff", [])
            if not d2diffs:
                break
            p2 = ".".join(str(x) for x in d2diffs[0].get("path", []))
            same = (
                short_action(rows[j]["action_type"]) == action
                and d2.get("key") == key
                and _obj_id(p2) == obj
                and _prop_name(p2) == prop
            )
            if not same:
                break
            last_rhs = d2diffs[0].get("rhs")
            j += 1

        burst_len = j - i
        if burst_len >= 3:
            key_parts = [p for p in key.split(".") if not (len(p) > 20 and "-" in p)]
            key_short = ".".join(key_parts[-2:]) if key_parts else key
            try:
                total = float(last_rhs) - float(first_lhs)
                direction = "↑" if total > 0 else "↓"
                out.append(f"BURST×{burst_len}: {action} [{key_short}] {prop} {first_lhs}→{last_rhs} ({direction}{abs(total):.3g})")
            except Exception:
                out.append(f"BURST×{burst_len}: {action} [{key_short}] {prop} {first_lhs}→{last_rhs}")
        else:
            for k in range(i, j):
                out.append(describe_action(rows[k]))

        i = j

    return out


def build_prompt(history: list[dict], label_set: list[str]) -> str:
    compressed = describe_actions_compressed(history)
    lines = ["You are observing a user editing a design canvas (Rocketium editor)."]
    lines.append(f"Here are their recent actions (bursts compressed):\n")
    for i, line in enumerate(compressed, 1):
        lines.append(f"{i:2d}. {line}")
    lines.append("")
    lines.append("What is the most likely NEXT action? Choose ONE from the list below:")
    for t in label_set:
        lines.append(f"  - {t}")
    lines.append("")
    lines.append("Reply with ONLY the action name, nothing else.")
    return "\n".join(lines)


def get_label_set(sessions: list[list[dict]]) -> list[str]:
    """Collect all action types present in data, return sorted."""
    seen = set()
    for session in sessions:
        for row in session:
            seen.add(short_action(row["action_type"]))
    return sorted(seen)


def run_benchmark(
    sessions: list[list[dict]],
    window: int = 10,
    n_samples: int = 100,
    seed: int = 42,
    model: str = "gpt-4o-mini",
) -> dict:
    client = OpenAI()
    label_set = get_label_set(sessions)
    print(f"Model: {model}")
    print(f"Label set ({len(label_set)}): {label_set}\n")

    # Build evaluation pairs: (history, true_next_action)
    examples = []
    for session in sessions:
        for i in range(window, len(session)):
            history = session[i - window : i]
            true_next = short_action(session[i]["action_type"])
            examples.append((history, true_next))

    random.seed(seed)
    random.shuffle(examples)
    examples = examples[:n_samples]

    correct = 0
    results = []

    for idx, (history, true_next) in enumerate(examples):
        prompt = build_prompt(history, label_set)
        print(f"[{idx+1}/{len(examples)}] true={true_next}", end=" ... ", flush=True)

        response = client.chat.completions.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        # Normalize: strip prefix if model included it
        predicted = raw.replace(ACTION_PREFIX, "").strip().upper()

        # Fuzzy match against label_set (case-insensitive)
        matched = None
        for lbl in label_set:
            if lbl.upper() == predicted:
                matched = lbl
                break
        if matched is None:
            # Try partial match
            for lbl in label_set:
                if lbl.upper() in predicted or predicted in lbl.upper():
                    matched = lbl
                    break

        hit = matched == true_next
        if hit:
            correct += 1
        print(f"pred={matched or raw!r}  {'✓' if hit else '✗'}")

        results.append(
            {
                "true": true_next,
                "predicted_raw": raw,
                "predicted_matched": matched,
                "correct": hit,
                "history_types": [short_action(r["action_type"]) for r in history],
            }
        )

    accuracy = correct / len(examples) if examples else 0.0
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{len(examples)} = {accuracy:.1%}")

    # Per-class breakdown
    from collections import defaultdict, Counter

    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        t = r["true"]
        class_stats[t]["total"] += 1
        if r["correct"]:
            class_stats[t]["correct"] += 1

    print("\nPer-class accuracy:")
    for t in sorted(class_stats):
        s = class_stats[t]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {t:<40s}  {s['correct']}/{s['total']}  {acc:.1%}")

    return {"model": model, "accuracy": accuracy, "n_samples": len(examples), "results": results}


def load_session(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Next-action prediction benchmark")
    parser.add_argument("--window", type=int, default=10, help="Context window size")
    parser.add_argument("--samples", type=int, default=100, help="Number of eval samples")
    parser.add_argument("--session", type=str, default=None, help="Specific session file")
    parser.add_argument("--all", action="store_true", help="Use all downloaded sessions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    if args.session:
        paths = [DATA_DIR / args.session]
    elif args.all:
        paths = sorted(DATA_DIR.glob("*.jsonl"))
    else:
        # Default: largest session
        paths = [DATA_DIR / "DSNsZv85yqoI-1158.jsonl"]

    sessions = []
    for p in paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
        s = load_session(p)
        print(f"Loaded {p.name}: {len(s)} actions")
        sessions.append(s)

    print()
    result = run_benchmark(sessions, window=args.window, n_samples=args.samples, seed=args.seed, model=args.model)

    out_path = DATA_DIR / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
