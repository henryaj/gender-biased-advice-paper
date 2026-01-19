#!/usr/bin/env python3
"""
Simple terminal dashboard for monitoring classification progress.
"""

import sqlite3
import time
import os
import subprocess
import sys
from datetime import datetime

DB_PATH = "data/research.db"

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    stats = {}

    # Get sources
    sources = conn.execute("SELECT * FROM sources").fetchall()

    for source in sources:
        source_id = source['source_id']
        name = source['name']

        posts = conn.execute(
            "SELECT COUNT(*) as n FROM posts WHERE source_id = ?",
            (source_id,)
        ).fetchone()['n']

        if posts == 0:
            continue

        comments = conn.execute("""
            SELECT COUNT(*) as n FROM comments c
            JOIN posts p ON c.post_id = p.post_id
            WHERE p.source_id = ?
        """, (source_id,)).fetchone()['n']

        classified_comments = conn.execute("""
            SELECT COUNT(*) as n FROM comment_classifications cc
            JOIN comments c ON cc.comment_id = c.comment_id
            JOIN posts p ON c.post_id = p.post_id
            WHERE p.source_id = ?
        """, (source_id,)).fetchone()['n']

        classified_posts = conn.execute("""
            SELECT COUNT(*) as n FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.source_id = ?
        """, (source_id,)).fetchone()['n']

        posts_with_details = conn.execute("""
            SELECT COUNT(*) as n FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.source_id = ? AND pc.situation_severity IS NOT NULL
        """, (source_id,)).fetchone()['n']

        # Gender distribution
        gender_dist = conn.execute("""
            SELECT pc.poster_gender, COUNT(*) as n
            FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.source_id = ?
            GROUP BY pc.poster_gender
        """, (source_id,)).fetchall()

        # Advice direction distribution
        direction_dist = conn.execute("""
            SELECT cc.advice_direction, COUNT(*) as n
            FROM comment_classifications cc
            JOIN comments c ON cc.comment_id = c.comment_id
            JOIN posts p ON c.post_id = p.post_id
            WHERE p.source_id = ? AND cc.is_advice = 1
            GROUP BY cc.advice_direction
        """, (source_id,)).fetchall()

        # Posts with classified comments
        posts_with_classified = conn.execute("""
            SELECT COUNT(DISTINCT p.post_id) as n
            FROM posts p
            JOIN comments c ON c.post_id = p.post_id
            JOIN comment_classifications cc ON cc.comment_id = c.comment_id
            WHERE p.source_id = ?
        """, (source_id,)).fetchone()['n']

        stats[name] = {
            'posts': posts,
            'classified_posts': classified_posts,
            'posts_with_details': posts_with_details,
            'comments': comments,
            'classified_comments': classified_comments,
            'posts_with_classified_comments': posts_with_classified,
            'gender_distribution': {r['poster_gender']: r['n'] for r in gender_dist},
            'direction_distribution': {r['advice_direction']: r['n'] for r in direction_dist}
        }

    conn.close()
    return stats

def get_running_processes():
    """Check for running classification processes."""
    processes = []
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'classify_reddit.py'],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    cmd_result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'args='],
                        capture_output=True, text=True
                    )
                    cmd = cmd_result.stdout.strip()
                    if cmd and 'classify_reddit.py' in cmd:
                        processes.append({'pid': pid, 'cmd': cmd})
                except:
                    pass
    except:
        pass
    return processes

def format_bar(current, total, width=30):
    """Create a progress bar."""
    if total == 0:
        return "[" + " " * width + "]"
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct*100:.1f}%"

def render_dashboard(stats, processes):
    clear_screen()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"  CLASSIFICATION DASHBOARD                          {now}")
    print("=" * 70)
    print()

    # Running processes
    print("RUNNING PROCESSES")
    print("-" * 70)
    if processes:
        for p in processes:
            print(f"  PID {p['pid']}: {p['cmd'][:60]}...")
    else:
        print("  No classification processes running")
    print()

    # Stats for each source
    for name, s in stats.items():
        display_name = name.upper().replace('_', ' ')
        print(f"{display_name}")
        print("-" * 70)

        # Posts
        print(f"  Posts:     {s['posts']:,} total | {s['classified_posts']:,} classified | {s['posts_with_details']:,} with details")

        # Gender distribution
        genders = s['gender_distribution']
        male = genders.get('male', 0)
        female = genders.get('female', 0)
        unknown = genders.get('unknown', 0)
        print(f"  Genders:   {female:,} female | {male:,} male | {unknown:,} unknown")

        # Comments
        pct = (s['classified_comments'] / s['comments'] * 100) if s['comments'] > 0 else 0
        print(f"  Comments:  {s['comments']:,} total | {s['classified_comments']:,} classified ({pct:.1f}%)")
        print(f"             {format_bar(s['classified_comments'], s['comments'])}")
        print(f"  Coverage:  {s['posts_with_classified_comments']:,} / {s['posts']:,} posts have classified comments")

        # Direction distribution
        dirs = s['direction_distribution']
        supportive = dirs.get('supportive_of_op', 0)
        critical = dirs.get('critical_of_op', 0)
        neutral = dirs.get('neutral', 0)
        mixed = dirs.get('mixed', 0)
        total_advice = supportive + critical + neutral + mixed
        if total_advice > 0:
            print(f"  Advice:    {total_advice:,} comments with advice")
            print(f"             supportive: {supportive:,} ({supportive/total_advice*100:.1f}%) | critical: {critical:,} ({critical/total_advice*100:.1f}%)")
            print(f"             neutral: {neutral:,} ({neutral/total_advice*100:.1f}%) | mixed: {mixed:,} ({mixed/total_advice*100:.1f}%)")

        print()

    print("-" * 70)
    print("  Press Ctrl+C to exit | Refreshes every 10 seconds")
    print()

def main():
    refresh_interval = 10
    if len(sys.argv) > 1:
        try:
            refresh_interval = int(sys.argv[1])
        except:
            pass

    try:
        while True:
            stats = get_stats()
            processes = get_running_processes()
            render_dashboard(stats, processes)
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\nExiting dashboard...")

if __name__ == "__main__":
    main()
