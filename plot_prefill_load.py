#!/usr/bin/env python
"""Plot prefill SM load from an nsys sqlite export.

Default: ONE panel — <metric> % over time for a single "prefill" NVTX range
(default: the last one; --range K selects). In practice the hardware-sampled
curve and the kernel-derived busy curve track each other, so the kernel view
(per-launch gantt when a range holds few launches, bucketed kernel-busy-%
when many) is behind --kernels; --gantt forces the gantt inside that panel.

Per-SM (0..107) lanes are NOT buildable from this data: the sqlite export
carries device-wide GPU metrics and per-kernel records only, no per-SM
timeline (that would need ncu, which is per-kernel replay, not nsys).

Usage on the A100 box:
  1. profile (nvtx ranges come from run_vllm_baseline.py's "prefill" range):
       VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       nsys profile -t cuda,nvtx -o prefill --force-overwrite true \
           --gpu-metrics-devices=cuda-visible \
           python run_vllm_baseline.py --model <dir> --bench-tokens 1024 ...
  2. nsys export --type sqlite -o prefill.sqlite prefill.nsys-rep
  3. python plot_prefill_load.py prefill.sqlite [--out prefill_load.png]
       [--range K]      K-th (0-based) "prefill" NVTX range, default: last
       [--metric NAME]  GPU metric; default: first available of a candidate
                        list (full list printed when nothing matches)
       [--kernels]      add the kernel-activity panel
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
GANTT_MAX_LAUNCHES = 2000  # above this the per-launch gantt is unreadable
BUSY_BUCKETS = 1000
# Metric names differ across nsys generations; try these in order.
DEFAULT_METRICS = (
    "SM Active",
    "SMs Active [Throughput %]",
    "SM Active [Throughput %]",
    "GR Active [Throughput %]",
)


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def find_col(cols: list[str], pattern: str) -> str | None:
    import re

    for c in cols:
        if re.search(pattern, c, re.IGNORECASE):
            return c
    return None


def nvtx_prefill_ranges(con: sqlite3.Connection, text: str = "prefill"):
    """[(start_ns, end_ns), ...] of the NVTX ranges named `text`."""
    cols = columns(con, "NVTX_EVENTS")
    if not cols or "text" not in cols:
        return []
    rows = con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL",
        (text,),
    ).fetchall()
    return sorted(rows)


def metric_meta(con: sqlite3.Connection):
    """(metadata table, name column) for GPU metrics, across nsys versions:
    older generations use GPU_METRICS_CONFIG(id, name); newer ones use
    TARGET_INFO_GPU_METRICS(metricId, typeId, metricName)."""
    if columns(con, "GPU_METRICS_CONFIG"):
        return "GPU_METRICS_CONFIG", "name"
    if columns(con, "TARGET_INFO_GPU_METRICS"):
        return "TARGET_INFO_GPU_METRICS", "metricName"
    return None


def metric_names(con: sqlite3.Connection) -> list[str]:
    meta = metric_meta(con)
    if meta is None:
        return []
    table, name_col = meta
    return [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT {name_col} FROM {table} WHERE {name_col} "
            f"IS NOT NULL"
        )
    ]


def sm_busy_series(con: sqlite3.Connection, metric: str, span=None):
    """[(t_ns, value)] for one GPU metric.

    GPU_METRICS is long-format (every sampled timestamp x every metric, 10^7+
    rows on a real profile), so the metric filter MUST happen inside SQL —
    joining/summing all metrics in Python would run for tens of minutes.
    """
    mcols = columns(con, "GPU_METRICS")
    ts_col = find_col(mcols, r"^timestamp")
    val_col = find_col(mcols, r"^value")
    mid_col = find_col(mcols, r"metricId")
    meta = metric_meta(con)
    if not (ts_col and val_col and mid_col and meta):
        return None
    table, name_col = meta
    if metric not in metric_names(con):
        print(f"available GPU metrics: {sorted(metric_names(con))}",
              file=sys.stderr)
        return None
    if table == "TARGET_INFO_GPU_METRICS":
        # newer schema keys on (metricId, typeId) — typeId is per device
        tcols = columns(con, table)
        tid = find_col(tcols, r"^metricId")
        ttype = find_col(tcols, r"^typeId")
        mtype = find_col(mcols, r"^typeId")
        if not (tid and ttype and mtype):
            return None
        join = f"c.{tid} = m.{mid_col} AND c.{ttype} = m.{mtype}"
    else:
        cid_col = find_col(columns(con, table), r"^id$")
        if not cid_col:
            return None
        join = f"c.{cid_col} = m.{mid_col}"
    where = f"WHERE c.{name_col} = ?"
    params: list = [metric]
    if span is not None:
        where += f" AND m.{ts_col} BETWEEN ? AND ?"
        params += [span[0], span[1]]
    rows = con.execute(
        f"SELECT m.{ts_col}, m.{val_col} FROM GPU_METRICS m "
        f"JOIN {table} c ON {join} {where} ORDER BY m.{ts_col}",
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


def short_kernel_name(name: str, limit: int = 40) -> str:
    """Strip template args / 'void ' so legend entries stay readable."""
    base = name.split("<", 1)[0].strip()
    if base.startswith("void "):
        base = base[5:]
    if not base:
        base = name
    if len(base) > limit:
        base = base[: limit - 1] + "…"
    return base


def busy_fraction_series(kernels, t0: int, t1: int, buckets: int = BUSY_BUCKETS):
    """[(bucket_center_ns, busy_pct)] — fraction of wall time covered by at
    least one kernel, per bucket. Overlaps (concurrent streams) cap at 100%
    because only coverage matters."""
    width = max((t1 - t0) / buckets, 1)
    covered = [0.0] * buckets
    for ks, ke, _ in kernels:
        if ke <= ks:
            continue
        a = max(int((ks - t0) / width), 0)
        b = min(int((ke - t0) / width), buckets - 1)
        for i in range(a, b + 1):
            lo = max(ks, t0 + int(i * width))
            hi = min(ke, t0 + int((i + 1) * width))
            if hi > lo:
                covered[i] += hi - lo
    return [((i + 0.5) * width, min(c / width, 1.0) * 100.0)
            for i, c in enumerate(covered)]


def draw_kernel_panel(ax, kernels, t0: int, t1: int, dur: float,
                      force_gantt: bool):
    if not force_gantt and len(kernels) > GANTT_MAX_LAUNCHES:
        series = busy_fraction_series(kernels, t0, t1)
        xs = [(t0 + x) / NS_PER_MS for x, _ in series]
        ys = [y for _, y in series]
        ax.fill_between(xs, ys, step="mid", color="#2563eb", alpha=0.7)
        ax.set_ylim(0, 105)
        ax.set_ylabel("kernel busy (%)")
        kmean = sum(ys) / len(ys)
        ax.axhline(kmean, ls="--", lw=0.8, color="gray")
        ax.text(dur, kmean + 2, f"mean {kmean:.0f}%", ha="right", fontsize=8,
                color="gray")
        ax.text(0.01, 0.97,
                f"{len(kernels)} launches — too many for a gantt; showing "
                f"bucketed busy% (--gantt to force)",
                transform=ax.transAxes, fontsize=8, va="top", color="gray")
        return

    # Collapse to truncated names first so identical kernels share a color.
    counts: dict[str, int] = {}
    for k in kernels:
        counts[short_kernel_name(k[2] or "?")] = (
            counts.get(short_kernel_name(k[2] or "?"), 0) + 1
        )
    palette = plt.cm.tab20.colors
    top_names = [n for n, _ in sorted(counts.items(), key=lambda x: -x[1])[:19]]
    color = {n: palette[i % 20] for i, n in enumerate(top_names)}
    for i, (ks, ke, name) in enumerate(kernels):
        c = color.get(short_kernel_name(name or "?"), "#94a3b8")
        ax.broken_barh(
            [((ks - t0) / NS_PER_MS,
              max((ke - ks) / NS_PER_MS, dur * 2e-4))], (i, 1),
            facecolors=c,
        )
    ax.set_ylabel("kernel launch #")
    ax.set_ylim(0, max(len(kernels), 1))
    handles = [Line2D([0], [0], color=color[n], lw=6, label=f"{n} ×{counts[n]}")
               for n in top_names]
    if len(counts) > len(top_names):
        handles.append(Line2D([0], [0], color="#94a3b8", lw=6, label="other"))
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="lower right")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="nsys export --type sqlite output")
    ap.add_argument("--out", default="prefill_load.png")
    ap.add_argument("--metric", default=None,
                    help="GPU metric name (default: first available of "
                         f"{DEFAULT_METRICS}; run once to list all names)")
    ap.add_argument("--range", type=int, default=-1,
                    help="0-based index of the 'prefill' NVTX range to plot "
                         "(default: the last one)")
    ap.add_argument("--kernels", action="store_true",
                    help="add the kernel-activity panel (default: metric "
                         "curve only)")
    ap.add_argument("--gantt", action="store_true",
                    help="force the per-launch gantt in the kernel panel")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    ranges = nvtx_prefill_ranges(con)
    if ranges:
        if not -len(ranges) <= args.range < len(ranges):
            print(f"--range {args.range} out of bounds: "
                  f"{len(ranges)} 'prefill' ranges found", file=sys.stderr)
            return 1
        t0, t1 = ranges[args.range]
    else:
        print("no NVTX 'prefill' range found - plotting whole timeline",
              file=sys.stderr)
        t0 = t1 = None

    metric = args.metric
    if metric is None:
        names = metric_names(con)
        metric = next((m for m in DEFAULT_METRICS if m in names), None)
        if metric is None:
            print(f"no known default metric; available: {sorted(names)}",
                  file=sys.stderr)
            return 1
        print(f"using metric: {metric}", file=sys.stderr)

    busy = sm_busy_series(con, metric, (t0, t1) if t0 is not None else None)
    kernels = kernel_intervals(con) if args.kernels else []
    if t0 is None:
        pts = [t for t, _ in (busy or [])] + [k[0] for k in kernels]
        if not pts:
            print("nothing to plot", file=sys.stderr)
            return 1
        t0, t1 = min(pts), max(pts)
    if args.kernels:
        kernels = [k for k in kernels if k[0] >= t0 and k[1] <= t1]
    if not kernels and not busy:
        print("nothing to plot inside the selected range", file=sys.stderr)
        return 1

    rel_ms = lambda t: (t - t0) / NS_PER_MS
    dur = rel_ms(t1)

    if args.kernels:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(16, 7), sharex=True,
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08},
        )
    else:
        fig, ax1 = plt.subplots(figsize=(16, 4))
        ax2 = None

    if busy:
        ts, vs = zip(*busy)
        ax1.plot([rel_ms(t) for t in ts], vs, lw=0.8, color="#2563eb")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel(f"{metric} (%)")
        mean = sum(v for _, v in busy) / len(busy)
        ax1.axhline(mean, ls="--", lw=0.8, color="gray")
        ax1.text(dur, mean + 2, f"mean {mean:.0f}%", ha="right", fontsize=8,
                 color="gray")
    n_k = f", {len(kernels)} kernel launches" if args.kernels else ""
    ax1.set_title(f"Prefill SM load ({dur:.1f} ms, range {args.range}{n_k})")

    if ax2 is not None:
        draw_kernel_panel(ax2, kernels, t0, t1, dur, args.gantt)
        ax2.set_xlabel("time (ms)")
    else:
        ax1.set_xlabel("time (ms)")

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
