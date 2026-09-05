"""
push.py — Run both ABL scripts and push updated dashboards to GitHub.

Usage:
  python push.py          # run both scripts and push
  python push.py --dry    # run scripts but don't push (test mode)

Requirements:
  - git installed and configured
  - This script lives in the root of your git repo
  - GitHub remote already set up (git remote add origin ...)
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime, timezone

DRY_RUN = '--dry' in sys.argv

SCRIPTS = [
    ('gold_standings.py', 'Gold Standings'),
    ('abl_analytics.py',  'ABL Analytics'),
]

OUTPUT_MAP = {
    'gold_dashboard.html': 'docs/gold.html',
    'dashboard.html':      'docs/analytics.html',
    'analytics_data.json': 'docs/analytics_data.json',
    'gold_state.json':     'docs/gold_state.json',
}

def run(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=False)
    if check and result.returncode != 0:
        print(f"  !! Command failed with code {result.returncode}")
    return result.returncode == 0

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    print("=" * 55)
    print("ABL Dashboard Push")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    print("=" * 55)

    # 1. Run each script
    any_success = False
    for script, name in SCRIPTS:
        if not os.path.exists(script):
            print(f"\n[SKIP] {name} — {script} not found")
            continue
        print(f"\n[RUN] {name}")
        ok = run([sys.executable, script])
        if ok:
            any_success = True
            print(f"  {name} completed OK")
        else:
            print(f"  {name} exited with errors — continuing anyway")

    if not any_success:
        print("\nAll scripts failed — nothing to push.")
        return

    # 2. Copy outputs to docs/
    print("\n[COPY] Output files -> docs/")
    os.makedirs('docs', exist_ok=True)
    copied = []
    for src_file, dst_file in OUTPUT_MAP.items():
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"  {src_file} -> {dst_file}")
            copied.append(dst_file)
        else:
            print(f"  {src_file} not found — skipping")

    if not copied:
        print("  Nothing to copy.")
        return

    if DRY_RUN:
        print("\n[DRY RUN] Skipping git push.")
        return

    # 3. Git add, commit, push
    print("\n[GIT] Staging changes...")
    run(['git', 'add', 'docs/'])

    # Check if there's anything to commit
    result = subprocess.run(['git', 'diff', '--staged', '--quiet'])
    if result.returncode == 0:
        print("  No changes to commit — dashboards already up to date.")
        return

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    commit_msg = f"Auto-update dashboards {timestamp}"
    print(f"\n[GIT] Committing: {commit_msg}")
    run(['git', 'commit', '-m', commit_msg])

    print("\n[GIT] Pushing to origin...")
    ok = run(['git', 'push', 'origin', 'main'])
    if ok:
        print("\n  Done — dashboards live on GitHub Pages.")
    else:
        print("\n  Push failed — check your git credentials and remote config.")
        print("  Try: git remote -v  to verify your remote is set up.")

if __name__ == '__main__':
    main()
