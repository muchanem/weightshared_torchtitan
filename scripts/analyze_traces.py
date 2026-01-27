#!/usr/bin/env python3
"""Analyze and compare PyTorch profiler traces."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(path: Path) -> dict:
    """Load a trace JSON file, handling potential encoding issues."""
    with open(path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error at position {e.pos}, attempting fix...")
        # Try to fix common issues - find the traceEvents array
        import re
        match = re.search(r'"traceEvents"\s*:\s*\[', content)
        if match:
            # Find the closing bracket for traceEvents
            start = match.end()
            bracket_count = 1
            i = start
            while i < len(content) and bracket_count > 0:
                if content[i] == '[':
                    bracket_count += 1
                elif content[i] == ']':
                    bracket_count -= 1
                i += 1
            # Extract just traceEvents and parse line by line
            events_str = content[start:i-1]
            events = []
            for line in events_str.split('\n'):
                line = line.strip().rstrip(',')
                if line.startswith('{') and line.endswith('}'):
                    try:
                        events.append(json.loads(line))
                    except:
                        pass
            return {"traceEvents": events}
        raise


def analyze_trace(trace: dict) -> dict:
    """Analyze trace events and compute statistics."""
    events = trace.get("traceEvents", [])

    # Group events by category and name
    stats = defaultdict(lambda: {"count": 0, "total_dur": 0, "events": []})

    for event in events:
        if not isinstance(event, dict):
            continue

        # Only look at duration events (ph="X")
        if event.get("ph") != "X":
            continue

        cat = event.get("cat", "unknown")
        name = event.get("name", "unknown")
        dur = event.get("dur", 0)  # duration in microseconds

        key = f"{cat}::{name}"
        stats[key]["count"] += 1
        stats[key]["total_dur"] += dur
        stats[key]["cat"] = cat
        stats[key]["name"] = name

    return dict(stats)


def print_top_operations(stats: dict, n: int = 30, by: str = "total_dur"):
    """Print top N operations by duration or count."""
    sorted_ops = sorted(stats.items(), key=lambda x: x[1][by], reverse=True)

    print(f"\nTop {n} operations by {by}:")
    print("-" * 80)
    print(f"{'Operation':<50} {'Count':>10} {'Total (ms)':>12} {'Avg (us)':>10}")
    print("-" * 80)

    for key, data in sorted_ops[:n]:
        total_ms = data["total_dur"] / 1000
        avg_us = data["total_dur"] / data["count"] if data["count"] > 0 else 0
        print(f"{key[:50]:<50} {data['count']:>10} {total_ms:>12.2f} {avg_us:>10.2f}")


def compare_traces(stats1: dict, stats2: dict, label1: str, label2: str):
    """Compare two trace statistics."""
    all_keys = set(stats1.keys()) | set(stats2.keys())

    comparisons = []
    for key in all_keys:
        dur1 = stats1.get(key, {}).get("total_dur", 0)
        dur2 = stats2.get(key, {}).get("total_dur", 0)
        count1 = stats1.get(key, {}).get("count", 0)
        count2 = stats2.get(key, {}).get("count", 0)

        if dur1 > 0 or dur2 > 0:
            diff = dur2 - dur1
            pct = ((dur2 / dur1) - 1) * 100 if dur1 > 0 else float('inf')
            comparisons.append((key, dur1, dur2, diff, pct, count1, count2))

    # Sort by absolute difference
    comparisons.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\n{'='*100}")
    print(f"Comparison: {label1} vs {label2}")
    print(f"{'='*100}")
    print(f"{'Operation':<40} {label1+' (ms)':>12} {label2+' (ms)':>12} {'Diff (ms)':>12} {'% Change':>10}")
    print("-" * 100)

    for key, dur1, dur2, diff, pct, count1, count2 in comparisons[:30]:
        dur1_ms = dur1 / 1000
        dur2_ms = dur2 / 1000
        diff_ms = diff / 1000
        pct_str = f"{pct:+.1f}%" if pct != float('inf') else "new"
        print(f"{key[:40]:<40} {dur1_ms:>12.2f} {dur2_ms:>12.2f} {diff_ms:>+12.2f} {pct_str:>10}")


def get_total_time(stats: dict) -> float:
    """Get total time in milliseconds."""
    return sum(s["total_dur"] for s in stats.values()) / 1000


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_traces.py <trace1.json> <trace2.json>")
        print("       python analyze_traces.py <trace.json>")
        sys.exit(1)

    trace1_path = Path(sys.argv[1])

    print(f"Loading {trace1_path}...")
    trace1 = load_trace(trace1_path)
    stats1 = analyze_trace(trace1)

    print(f"\n=== Analysis of {trace1_path.name} ===")
    print(f"Total tracked time: {get_total_time(stats1):.2f} ms")
    print_top_operations(stats1)

    if len(sys.argv) >= 3:
        trace2_path = Path(sys.argv[2])
        print(f"\nLoading {trace2_path}...")
        trace2 = load_trace(trace2_path)
        stats2 = analyze_trace(trace2)

        print(f"\n=== Analysis of {trace2_path.name} ===")
        print(f"Total tracked time: {get_total_time(stats2):.2f} ms")
        print_top_operations(stats2)

        compare_traces(stats1, stats2, trace1_path.parent.name, trace2_path.parent.name)


if __name__ == "__main__":
    main()
