#!/usr/bin/env python3
"""Compare profiler traces using HolisticTraceAnalysis."""

import sys
sys.path.insert(0, "/local/scratch/muchane_680186/HolisticTraceAnalysis")

from hta.trace_analysis import TraceAnalysis
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Updated paths for new scratch directory
UNSHARED_TRACE = "/local/scratch/muchane_680186/weightshared_torchtitan/outputs/traces/250m_unshared/iteration_10"
SHARED_TRACE = "/local/scratch/muchane_680186/weightshared_torchtitan/outputs/traces/250m_shared/iteration_10"

def analyze_model(name, trace_dir):
    """Analyze a single model's trace."""
    print("=" * 80)
    print(f"{name} MODEL")
    print("=" * 80)

    analyzer = TraceAnalysis(trace_dir=trace_dir)

    print("\n--- Temporal Breakdown ---")
    temporal = analyzer.get_temporal_breakdown(visualize=False)
    print(temporal.to_string())

    print("\n--- Kernel Type Breakdown ---")
    kernel_type, kernel = analyzer.get_gpu_kernel_breakdown(visualize=False, num_kernels=20)
    print(kernel_type.to_string())

    print("\n--- Top 20 Kernels ---")
    print(kernel.head(20).to_string())

    return temporal, kernel_type, kernel, analyzer


def main():
    print("HTA Trace Analysis")
    print("=" * 80)

    temporal_u, kernel_type_u, kernel_u, analyzer_u = analyze_model("UNSHARED", UNSHARED_TRACE)
    print("\n")
    temporal_s, kernel_type_s, kernel_s, analyzer_s = analyze_model("SHARED", SHARED_TRACE)

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (rank 0)")
    print("=" * 80)

    u0 = temporal_u[temporal_u['rank'] == 0].iloc[0]
    s0 = temporal_s[temporal_s['rank'] == 0].iloc[0]

    print(f"\n{'Metric':<30} {'Unshared':>15} {'Shared':>15} {'Change':>12}")
    print("-" * 75)
    for col in ['idle_time(us)', 'compute_time(us)', 'non_compute_time(us)', 'kernel_time(us)']:
        if col in temporal_u.columns:
            u_val = float(u0[col])
            s_val = float(s0[col])
            diff = ((s_val / u_val) - 1) * 100 if u_val > 0 else 0
            print(f"{col:<30} {u_val:>15.0f} {s_val:>15.0f} {diff:>+11.1f}%")

    # Kernel comparison
    print("\n" + "=" * 80)
    print("TOP KERNEL TIME DIFFERENCES (Shared - Unshared)")
    print("=" * 80)

    # Merge kernel data for comparison
    kernel_u_agg = kernel_u.groupby('name').agg({'sum': 'sum', 'count': 'sum'}).reset_index()
    kernel_s_agg = kernel_s.groupby('name').agg({'sum': 'sum', 'count': 'sum'}).reset_index()

    merged = kernel_u_agg.merge(kernel_s_agg, on='name', how='outer', suffixes=('_unshared', '_shared'))
    merged = merged.fillna(0)
    merged['diff_us'] = merged['sum_shared'] - merged['sum_unshared']
    merged['diff_pct'] = merged.apply(lambda r: ((r['sum_shared']/r['sum_unshared'])-1)*100 if r['sum_unshared'] > 0 else float('inf'), axis=1)
    merged = merged.sort_values('diff_us', ascending=False)

    print(f"\n{'Kernel':<50} {'Unshared(us)':>12} {'Shared(us)':>12} {'Diff(us)':>12} {'%Change':>10}")
    print("-" * 100)
    for _, row in merged.head(20).iterrows():
        pct_str = f"{row['diff_pct']:+.1f}%" if row['diff_pct'] != float('inf') else "NEW"
        print(f"{row['name'][:50]:<50} {row['sum_unshared']:>12.0f} {row['sum_shared']:>12.0f} {row['diff_us']:>+12.0f} {pct_str:>10}")


if __name__ == "__main__":
    main()
