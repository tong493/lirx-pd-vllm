#!/usr/bin/env python
"""Plot whole-prefill SM load from an nsys sqlite export.

Two panels, restricted to the prefill NVTX range when present:
  top    - SM Busy % over time (hardware-sampled GPU metrics, ~10 kHz)
  bottom - one rectangle per kernel launch (start->end, colored by name)

Usage on the A100 box:
  1. profile (nvtx ranges come from run_vllm_baseline.py's "prefill" range):
       nsys profile -t cuda,nvtx -o prefill --force-overwrite true \
           --gpu-metrics-devices=cuda-visible \
           python run_vllm_baseline.py --model <dir> --bench-tokens 1024 --bench-reps 1
  2. nsys export --type sqlite -o prefill.sqlite prefill.nsys-rep
  3. python plot_prefill_load.py prefill.sqlite [--out prefill_load.png]

Without a matching NVTX range the whole timeline is plotted instead.
The nsys sqlite schema differs slightly across versions, so table/column
names are probed rather than assumed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NS_PER_MS = 1e6


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def find_col(cols: list[str], pattern: str) -> str | None:
    import re

    for c in cols:
        if re.search(pattern, c, re.IGNORECASE):
            return c
    return None


def nvtx_prefill_span(con: sqlite3.Connection, text: str = "prefill"):
    """(start_ns, end_ns) of the NVTX range named `text`, or None."""
    cols = columns(con, "NVTX_EVENTS")
    if not cols or "text" not in cols:
        return None
    rows = con.execute(
        f"SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL",
        (text,),
    ).fetchall()
    if not rows:
        return None
    return min(r[0] for r in rows), max(r[1] for r in rows)


def sm_busy_series(con: sqlite3.Connection, metric: str, span=None):
    """[(t_ns, value)] for one GPU metric.

    GPU_METRICS is long-format (every sampled timestamp x every metric, 10^7+
    rows on a real profile), so the metric filter MUST happen inside SQL —
    joining/summing all metrics in Python would run for tens of minutes.
    """
    mcols = columns(con, "GPU_METRICS")
    ccols = columns(con, "GPU_METRICS_CONFIG")
    if not mcols or not ccols:
        return None
    mid_col = find_col(mcols, r"metricId")
    ts_col = find_col(mcols, r"^timestamp")
    val_col = find_col(mcols, r"^value")
    cid_col = find_col(ccols, r"^id$")
    name_col = find_col(ccols, r"name")
    if not all([mid_col, ts_col, val_col, cid_col, name_col]):
        return None
    names = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT {name_col} FROM GPU_METRICS_CONFIG WHERE "
            f"{name_col} IS NOT NULL"
        )
    ]
    if metric not in names:
        print(f"available GPU metrics: {sorted(names)}", file=sys.stderr)
        return None
    where = (f"WHERE m.{mid_col} IN "
             f"(SELECT {cid_col} FROM GPU_METRICS_CONFIG WHERE {name_col} = ?)")
    params: list = [metric]
    if span is not None:
        where += f" AND m.{ts_col} BETWEEN ? AND ?"
        params += [span[0], span[1]]
    rows = con.execute(
        f"SELECT m.{ts_col}, m.{val_col} FROM GPU_METRICS m {where} "
        f"ORDER BY m.{ts_col}",
        params,
    ).fetchall()
    # Same-timestamp rows across devices are averaged.
    series: dict[int, float] = {}
    counts: dict[int, int] = {}
    for t, v in rows:
        series[t] = series.get(t, 0.0) + float(v)
        counts[t] = counts.get(t, 0) + 1
    return [(t, series[t] / counts[t]) for t in sorted(series)]


def kernel_intervals(con: sqlite3.Connection):
    cols = columns(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    if not cols:
        print(
            "no CUPTI_ACTIVITY_KIND_KERNEL table — zero kernel records were "
            "captured. When profiling vLLM v1, all kernels run in the "
            "EngineCore child process; retry with "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs inside the "
            "profiled process.",
            file=sys.stderr,
        )
        return []
    name_ref = find_col(cols, r"shortName|demangledName")
    rows = con.execute(
        f"SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        f"JOIN StringIds s ON k.{name_ref} = s.id ORDER BY k.start"
    ).fetchall()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="nsys export --type sqlite output")
    ap.add_argument("--out", default="prefill_load.png")
    ap.add_argument("--metric", default="SM Active",
                    help="GPU metric name; run once to list available names")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    span = nvtx_prefill_span(con)
    busy = sm_busy_series(con, args.metric, span)
    kernels = kernel_intervals(con)
    if span is not None:
        t0, t1 = span
        kernels = [k for k in kernels if k[0] >= t0 and k[1] <= t1]
    else:
        print("no NVTX 'prefill' range found - plotting whole timeline",
              file=sys.stderr)
        pts = [t for t, _ in busy] + [k[0] for k in kernels]
        if not pts:
            print("nothing to plot", file=sys.stderr)
            return 1
        t0, t1 = min(pts), max(pts)

    if not kernels and not busy:
        print("nothing to plot", file=sys.stderr)
        return 1
    rel_ms = lambda t: (t - t0) / NS_PER_MS
    dur = rel_ms(t1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08},
    )

    if busy:
        ts, vs = zip(*busy)
        ax1.plot([rel_ms(t) for t in ts], vs, lw=0.8, color="#2563eb")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel(f"{args.metric} (%)")
        mean = sum(v for _, v in busy) / len(busy)
        ax1.axhline(mean, ls="--", lw=0.8, color="gray")
        ax1.text(dur, mean + 2, f"mean {mean:.0f}%", ha="right", fontsize=8,
                 color="gray")
    ax1.set_title(f"Prefill SM load ({dur:.1f} ms, {len(kernels)} kernel launches)")

    counts: dict[str, int] = {}
    for k in kernels:
        counts[k[2] or "?"] = counts.get(k[2] or "?", 0) + 1
    palette = plt.cm.tab20.colors
    top_names = [n for n, _ in sorted(counts.items(), key=lambda x: -x[1])[:19]]
    color = {n: palette[i % 20] for i, n in enumerate(top_names)}
    for i, (ks, ke, name) in enumerate(kernels):
        c = color.get(name or "?", "#94a3b8")
        ax2.broken_barh(
            [(rel_ms(ks), max((ke - ks) / NS_PER_MS, dur * 2e-4))], (i, 1),
            facecolors=c,
        )
    ax2.set_ylabel("kernel launch #")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylim(0, max(len(kernels), 1))
    handles = [Line2D([0], [0], color=color[n], lw=6, label=f"{n} ×{counts[n]}")
               for n in top_names]
    if len(counts) > len(top_names):
        handles.append(Line2D([0], [0], color="#94a3b8", lw=6, label="other"))
    ax2.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
