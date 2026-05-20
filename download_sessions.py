"""
Download editor sessions from ClickHouse + states from S3 for multiple users.
Runs in batches to avoid timeouts and rate limits.

Usage:
  python download_sessions.py [--dry-run]
"""

import json
import time
import subprocess
import argparse
from pathlib import Path

import requests

# --- Config ---
CH_HOST = "https://b6gd2i2opm.eu-west-1.aws.clickhouse.cloud:8443"
CH_USER = "default"
WORKSPACE_ID = "623a177218b0b745935d7a49"
S3_BUCKET = "s3://versions.prod.rocketium.com"

DATA_DIR = Path(__file__).parent
STATES_DIR = DATA_DIR / "states"
STATES_DIR.mkdir(exist_ok=True)

# Users to download: (user_id, label, max_projects)
# Top 4 new users from Nykaa Fashion workspace
USERS = [
    ("6465b5ef12dd262b452cdd18", "u1", 12),  # 68K actions
    ("698c18884abb907eb9543a53", "u2", 12),  # 63K actions
    ("69dca868686daf55cc6e2c73", "u3", 12),  # 5.7K actions
    ("69803f66fd9a27cda5862d21", "u4", 12),  # 5.5K actions
]

# Already downloaded projects — skip these
EXISTING_PROJECTS = {p.stem for p in DATA_DIR.glob("*.jsonl")}

# Min actions per project to be worth including
MIN_ACTIONS = 200


def get_ch_password() -> str:
    result = subprocess.run(
        ["aws", "secretsmanager", "get-secret-value",
         "--secret-id", "rocketium/PROD",
         "--query", "SecretString", "--output", "text"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)["clickhousePassword"]


def ch_query(password: str, sql: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(
                CH_HOST,
                params={"query": sql},
                auth=(CH_USER, password),
                timeout=60,
            )
            resp.raise_for_status()
            lines = [l for l in resp.text.strip().split("\n") if l]
            return [json.loads(l) for l in lines]
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {retries} attempts: {e}")
                return []


def get_top_projects(password: str, user_id: str, limit: int) -> list[dict]:
    sql = f"""
SELECT project_id, count() as cnt
FROM prod.editor_deltas
WHERE workspace_id = '{WORKSPACE_ID}'
  AND user_id = '{user_id}'
  AND action_type LIKE '@@canvas-editor/%'
GROUP BY project_id
HAVING cnt >= {MIN_ACTIONS}
ORDER BY cnt DESC
LIMIT {limit}
FORMAT JSONEachRow
"""
    return ch_query(password, sql)


def download_project_actions(password: str, user_id: str, project_id: str,
                              label: str, dry_run: bool = False) -> bool:
    out_path = DATA_DIR / f"{project_id}.jsonl"
    if out_path.exists():
        print(f"    {project_id} already exists, skipping")
        return True

    if dry_run:
        print(f"    [DRY RUN] would download {project_id}")
        return True

    sql = f"""
SELECT
    project_id,
    action_type,
    delta_json,
    created_at
FROM prod.editor_deltas
WHERE workspace_id = '{WORKSPACE_ID}'
  AND user_id = '{user_id}'
  AND project_id = '{project_id}'
  AND action_type LIKE '@@canvas-editor/%'
ORDER BY created_at ASC
FORMAT JSONEachRow
"""
    rows = ch_query(password, sql)
    if not rows:
        print(f"    {project_id}: no rows returned")
        return False

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"    {project_id}: {len(rows)} actions saved")
    return True


def download_states(project_id: str, dry_run: bool = False) -> bool:
    initial_path = STATES_DIR / f"{project_id}_initial.json"
    final_path = STATES_DIR / f"{project_id}_final.json"

    if initial_path.exists() and final_path.exists():
        print(f"    {project_id} states already exist, skipping")
        return True

    if dry_run:
        print(f"    [DRY RUN] would download states for {project_id}")
        return True

    # Find initial key (earliest non-cached file)
    result = subprocess.run(
        ["aws", "s3", "ls", f"{S3_BUCKET}/{project_id}/"],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    non_cached = [l for l in lines if "_cached" not in l and l]

    if not non_cached:
        print(f"    {project_id}: no versions found in S3")
        return False

    # Earliest = first after sort
    initial_filename = sorted(non_cached)[0].split()[-1]

    ok = True
    if not initial_path.exists():
        r = subprocess.run(
            ["aws", "s3", "cp",
             f"{S3_BUCKET}/{project_id}/{initial_filename}", str(initial_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"    {project_id}: failed to download initial state")
            ok = False

    if not final_path.exists():
        r = subprocess.run(
            ["aws", "s3", "cp",
             f"{S3_BUCKET}/{project_id}/{project_id}_cached.json", str(final_path)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"    {project_id}: failed to download final state")
            ok = False

    if ok:
        print(f"    {project_id}: states downloaded")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Projects to process per batch before pausing")
    args = parser.parse_args()

    print("Fetching ClickHouse password...")
    password = get_ch_password()
    print("OK\n")

    all_projects: list[tuple[str, str]] = []  # (project_id, user_label)

    # Step 1: collect all projects across users
    for user_id, label, max_proj in USERS:
        print(f"User {label} ({user_id[-6:]}) — fetching top projects...")
        projects = get_top_projects(password, user_id, max_proj)
        new = [(p["project_id"], label) for p in projects
               if p["project_id"] not in EXISTING_PROJECTS]
        print(f"  Found {len(projects)} qualifying projects, {len(new)} new\n")
        all_projects.extend(new)

    if not all_projects:
        print("Nothing new to download.")
        return

    print(f"Total new projects to download: {len(all_projects)}\n")

    # Step 2: download in batches
    batch_size = args.batch_size
    success, failed = 0, 0

    for i in range(0, len(all_projects), batch_size):
        batch = all_projects[i:i + batch_size]
        print(f"--- Batch {i//batch_size + 1} ({i+1}–{min(i+batch_size, len(all_projects))} of {len(all_projects)}) ---")

        for project_id, label in batch:
            print(f"  [{label}] {project_id}")
            ok_actions = download_project_actions(password,
                # look up user_id from label
                next(u for u, l, _ in USERS if l == label),
                project_id, label, args.dry_run)
            ok_states = download_states(project_id, args.dry_run)
            if ok_actions and ok_states:
                success += 1
            else:
                failed += 1

        if i + batch_size < len(all_projects):
            print(f"  Batch done. Pausing 2s...\n")
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Done. Success: {success}, Failed: {failed}")
    print(f"Total .jsonl files now: {len(list(DATA_DIR.glob('*.jsonl')))}")
    print(f"Total state files now: {len(list(STATES_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
