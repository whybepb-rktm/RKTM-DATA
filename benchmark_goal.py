"""
Goal-conditioned next-action prediction benchmark.

Same as benchmark.py but each prompt includes:
  - Initial canvas state summary (what the project looked like at the start)
  - Final canvas state summary  (the "goal" — where the session ended)
  - Recent action history (window)

Usage:
  python benchmark_goal.py [--window N] [--samples K] [--model MODEL]
"""

import json
import os
import random
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter

from openai import OpenAI

DATA_DIR = Path(__file__).parent
STATES_DIR = DATA_DIR / "states"

ACTION_PREFIX = "@@canvas-editor/"

_NUMERIC_PROPS = {"left", "top", "width", "height", "fontSize", "lineHeight",
                  "imageScale", "imageTop", "imageLeft"}
_FLAG_PROPS = {"visible", "isDeleted"}
_LAYOUT_PROPS = {"objectPosition", "objectFit"}
_SIZE_PROPS = {"imageWidth", "imageHeight"}
_BURST_PROPS = _NUMERIC_PROPS


def short_action(action_type: str) -> str:
    return action_type.replace(ACTION_PREFIX, "")


def _prop_name(path: str) -> str:
    return path.split(".")[-1] if path else "unknown"


def _obj_id(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else path


def describe_action(row: dict) -> str:
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
        objects.add(obj.split(".")[-1][:8])
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


# --- Canvas state summarizer ---

def _extract_objects(state_json: dict) -> list[dict]:
    """Pull all canvas objects out of a version snapshot."""
    try:
        caps = state_json.get("capsules", [])
        if not caps:
            return []
        v = caps[0]["canvasData"]["variant"]
        objs = v.get("objects", {})
        if isinstance(objs, dict):
            return list(objs.values())
        return objs
    except Exception:
        return []


def _canvas_size(state_json: dict) -> str:
    try:
        caps = state_json.get("capsules", [])
        v = caps[0]["canvasData"]["variant"]
        sizes = v.get("sizes", {})
        if sizes:
            sz = next(iter(sizes.values()))
            return f"{sz['width']}×{sz['height']}"
    except Exception:
        pass
    return "unknown"


def summarize_state(state_json: dict, label: str = "") -> str:
    """Return a compact text summary of a canvas state for prompt injection."""
    size = _canvas_size(state_json)
    objs = _extract_objects(state_json)

    type_counts = Counter(o.get("type", "?") for o in objs)
    total = len(objs)
    visible = sum(1 for o in objs if o.get("visible", True) and not o.get("isDeleted", False))

    # Collect text content
    texts = []
    for o in objs:
        t = o.get("serializedText") or o.get("displayText") or o.get("text")
        if t and isinstance(t, str) and t.strip():
            texts.append(t.strip()[:50])

    # Build type summary line
    type_parts = []
    for t, n in type_counts.most_common():
        short = t.replace("-container", "").replace("rounded-", "r")
        type_parts.append(f"{n}×{short}")

    lines = [f"[{label}] Canvas {size} | {total} objects ({visible} visible): {', '.join(type_parts)}"]
    if texts:
        text_sample = " | ".join(f'"{t}"' for t in texts[:4])
        lines.append(f"  Text layers: {text_sample}")

    return "\n".join(lines)


def state_diff_summary(initial: dict, final: dict) -> str:
    """Highlight what changed between initial and final state."""
    init_objs = {o["id"]: o for o in _extract_objects(initial) if "id" in o}
    final_objs = {o["id"]: o for o in _extract_objects(final) if "id" in o}

    added = [o for oid, o in final_objs.items() if oid not in init_objs]
    removed = [o for oid, o in init_objs.items() if oid not in final_objs]
    modified = []

    for oid in set(init_objs) & set(final_objs):
        io, fo = init_objs[oid], final_objs[oid]
        changes = []
        for k in ["left", "top", "width", "height", "visible", "serializedText", "src"]:
            iv, fv = io.get(k), fo.get(k)
            if iv != fv and iv is not None and fv is not None:
                if k in _NUMERIC_PROPS:
                    try:
                        delta = float(fv) - float(iv)
                        changes.append(f"{k}:{iv}→{fv}({'↑' if delta>0 else '↓'}{abs(delta):.3g})")
                    except Exception:
                        changes.append(f"{k} changed")
                elif k == "serializedText":
                    changes.append(f'text:"{str(iv)[:20]}"→"{str(fv)[:20]}"')
                else:
                    changes.append(f"{k}:{iv}→{fv}")
        if changes:
            t = fo.get("type", "?").replace("-container", "")
            modified.append(f"{t}({', '.join(changes[:3])})")

    parts = []
    if added:
        atypes = Counter(o.get("type","?").replace("-container","") for o in added)
        parts.append(f"+{len(added)} objs added ({', '.join(f'{n}×{t}' for t,n in atypes.items())})")
    if removed:
        parts.append(f"-{len(removed)} objs removed")
    if modified:
        parts.append(f"{len(modified)} objs modified: " + "; ".join(modified[:4]))
        if len(modified) > 4:
            parts[-1] += f" (+{len(modified)-4} more)"

    if not parts:
        return "  (no structural changes detected)"
    return "  Session changes: " + " | ".join(parts)


# --- Prompt builder ---

def build_prompt(history: list[dict], label_set: list[str],
                 initial_state: dict | None = None,
                 final_state: dict | None = None) -> str:
    compressed = describe_actions_compressed(history)
    lines = ["You are observing a user editing a design canvas (Rocketium editor)."]

    if initial_state is not None and final_state is not None:
        lines.append("\n=== PROJECT CONTEXT ===")
        lines.append(summarize_state(initial_state, "START"))
        lines.append(summarize_state(final_state, "GOAL"))
        lines.append(state_diff_summary(initial_state, final_state))
        lines.append("=======================\n")

    lines.append("Here are the user's recent actions (bursts compressed):\n")
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
    seen = set()
    for session in sessions:
        for row in session:
            seen.add(short_action(row["action_type"]))
    return sorted(seen)


def load_state(project_id: str, kind: str) -> dict | None:
    """Load initial or final state JSON for a project_id."""
    path = STATES_DIR / f"{project_id}_{kind}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_benchmark(
    sessions: list[list[dict]],
    project_ids: list[str],
    window: int = 10,
    n_samples: int = 100,
    seed: int = 42,
    model: str = "gpt-4o-mini",
    use_goal: bool = True,
) -> dict:
    client = OpenAI()
    label_set = get_label_set(sessions)
    print(f"Model: {model}  |  goal-conditioned: {use_goal}")
    print(f"Label set ({len(label_set)}): {label_set}\n")

    # Pre-load states
    states: dict[str, dict] = {}
    for pid in project_ids:
        init = load_state(pid, "initial")
        final = load_state(pid, "final")
        states[pid] = {"initial": init, "final": final}
        status = "ok" if init and final else ("no_initial" if not init else "no_final")
        print(f"  {pid}: states {status}")
    print()

    examples = []
    for session, pid in zip(sessions, project_ids):
        for i in range(window, len(session)):
            history = session[i - window: i]
            true_next = short_action(session[i]["action_type"])
            examples.append((history, true_next, pid))

    random.seed(seed)
    random.shuffle(examples)
    examples = examples[:n_samples]

    correct = 0
    results = []

    for idx, (history, true_next, pid) in enumerate(examples):
        init_state = states[pid]["initial"] if use_goal else None
        final_state = states[pid]["final"] if use_goal else None

        prompt = build_prompt(history, label_set, init_state, final_state)
        print(f"[{idx+1}/{len(examples)}] pid={pid[-4:]} true={true_next}", end=" ... ", flush=True)

        response = client.chat.completions.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        predicted = raw.replace(ACTION_PREFIX, "").strip().upper()

        matched = None
        for lbl in label_set:
            if lbl.upper() == predicted:
                matched = lbl
                break
        if matched is None:
            for lbl in label_set:
                if lbl.upper() in predicted or predicted in lbl.upper():
                    matched = lbl
                    break

        hit = matched == true_next
        if hit:
            correct += 1
        print(f"pred={matched or raw!r}  {'✓' if hit else '✗'}")

        results.append({
            "true": true_next,
            "predicted_raw": raw,
            "predicted_matched": matched,
            "correct": hit,
            "project_id": pid,
            "history_types": [short_action(r["action_type"]) for r in history],
        })

    accuracy = correct / len(examples) if examples else 0.0
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{len(examples)} = {accuracy:.1%}")

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

    return {
        "model": model,
        "goal_conditioned": use_goal,
        "accuracy": accuracy,
        "n_samples": len(examples),
        "results": results,
    }


def load_session(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Goal-conditioned next-action benchmark")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--no-goal", action="store_true", help="Disable goal context (baseline comparison)")
    args = parser.parse_args()

    paths = sorted(DATA_DIR.glob("*.jsonl"))
    if not paths:
        print("No .jsonl files found in", DATA_DIR, file=sys.stderr)
        sys.exit(1)

    sessions = []
    project_ids = []
    for p in paths:
        s = load_session(p)
        print(f"Loaded {p.name}: {len(s)} actions")
        sessions.append(s)
        project_ids.append(p.stem)

    print()
    result = run_benchmark(
        sessions,
        project_ids,
        window=args.window,
        n_samples=args.samples,
        seed=args.seed,
        model=args.model,
        use_goal=not args.no_goal,
    )

    suffix = "goal" if not args.no_goal else "baseline"
    out_path = DATA_DIR / f"benchmark_results_{args.model}_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
