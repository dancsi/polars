"""
TPC-DS SQL benchmark: Polars `SQLContext` vs DuckDB, on the same parquet data.

Data comes from DuckDB's `dsdgen` and the 99 queries from its `tpcds_queries()`, both
via the DuckDB `tpcds` extension (downloaded on first use). Each (engine, query) runs
in a fresh subprocess, so a crash or timeout only fails that query. Results are
compared positionally: numerics as Float64 within a relative tolerance, the rest as
strings, retried order-insensitively for ties in ORDER BY.

Requires `duckdb`, `pyarrow` and Polars built from main (`make build-release`);
released versions lack SQL features the queries need.

Usage::

    python tpcds.py run --sf 10 --data-dir /tmp/tpcds --out results-m4.json
    python tpcds.py report results-m4.json results-graviton4.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

import polars as pl

ENGINES = ("polars", "duckdb")
NUMERIC_RTOL = 1e-6


def generate(sf: str, data_dir: Path) -> Path:
    """Write each TPC-DS table at `sf` to `<data_dir>/sf<sf>/<table>.parquet`."""
    out = data_dir / f"sf{sf}"
    if out.exists():
        return out
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".sf{sf}-", dir=data_dir))
    try:
        # A file-backed database lets dsdgen spill instead of holding SF10+ in RAM.
        con = duckdb.connect(tmp / "gen.duckdb")
        con.sql("INSTALL tpcds; LOAD tpcds")
        con.sql(f"CALL dsdgen(sf={sf})")
        for (t,) in con.sql("SHOW TABLES").fetchall():
            con.sql(
                f"COPY {t} TO '{tmp / t}.parquet' (FORMAT parquet, COMPRESSION zstd)"
            )
        con.close()
        (tmp / "gen.duckdb").unlink()
        tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def fetch_queries(query_dir: Path) -> list[str]:
    query_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("INSTALL tpcds; LOAD tpcds")
    names = []
    for nr, sql in con.sql("SELECT query_nr, query FROM tpcds_queries()").fetchall():
        (query_dir / f"q{nr}.sql").write_text(sql)
        names.append(f"q{nr}")
    return names


def worker(
    engine: str, sql_path: Path, data: Path, out: Path, repeat: int, polars_engine: str
) -> None:
    sql = sql_path.read_text().strip().rstrip(";")
    tables = {p.stem: p for p in sorted(data.glob("*.parquet"))}
    meta: dict[str, Any] = {"times_s": []}
    try:
        if engine == "duckdb":
            con = duckdb.connect()
            for t, p in tables.items():
                con.sql(f"CREATE VIEW {t} AS FROM read_parquet('{p}')")
            for _ in range(repeat):
                t0 = time.perf_counter()
                df = con.sql(sql).pl()
                meta["times_s"].append(time.perf_counter() - t0)
        else:
            ctx = pl.SQLContext({t: pl.scan_parquet(p) for t, p in tables.items()})
            for _ in range(repeat):
                t0 = time.perf_counter()
                df = ctx.execute(sql, eager=False).collect(engine=polars_engine)  # type: ignore[call-overload]
                meta["times_s"].append(time.perf_counter() - t0)
        meta["schema"] = {k: str(v) for k, v in df.schema.items()}
        df.write_parquet(Path(f"{out}.parquet"))
        meta["ok"] = True
    except BaseException as e:  # PanicException is a BaseException
        meta["ok"] = False
        meta["error"] = f"{type(e).__name__}: {e}"[:2000]
        meta["traceback"] = traceback.format_exc()[-4000:]
    Path(f"{out}.json").write_text(json.dumps(meta))


def run_engine(
    engine: str, q: str, args: argparse.Namespace, data: Path, work: Path
) -> dict[str, Any]:
    out = work / f"{q}.{engine}"
    cmd = [
        sys.executable, __file__, "_worker", engine, str(work / "queries" / f"{q}.sql"),
        str(data), str(out), str(args.repeat), args.polars_engine,
    ]  # fmt: skip
    try:
        proc = subprocess.run(
            cmd, timeout=args.timeout, check=False, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {args.timeout}s", "times_s": []}
    meta = Path(f"{out}.json")
    if not meta.exists():
        err = f"worker died with exit code {proc.returncode}: {proc.stderr[-500:]}"
        return {"ok": False, "error": err, "times_s": []}
    result: dict[str, Any] = json.loads(meta.read_text())
    return result


def normalise(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col(c).cast(pl.Float64)
        if dt.is_numeric() or isinstance(dt, pl.Decimal)
        else pl.col(c).cast(pl.String)
        for c, dt in df.schema.items()
    )


def first_mismatch(a: pl.DataFrame, b: pl.DataFrame) -> str | None:
    for ca, cb in zip(a.columns, b.columns, strict=True):
        x, y = a[ca], b[cb]
        both_null = x.is_null() & y.is_null()
        if x.dtype == pl.Float64:
            close = (x - y).abs() <= NUMERIC_RTOL * y.abs().clip(lower_bound=1)
            ok = close.fill_null(False) | (x == y).fill_null(False) | both_null
        else:
            ok = (x == y).fill_null(False) | both_null
        if not ok.all():
            return ca
    return None


def sort_all(df: pl.DataFrame) -> pl.DataFrame:
    keys = [
        pl.col(c).round(4) if dt == pl.Float64 else pl.col(c)
        for c, dt in df.schema.items()
    ]
    return df.sort(keys, nulls_last=True)


def compare(p: pl.DataFrame, d: pl.DataFrame) -> tuple[str, str]:
    if p.width != d.width:
        return "width", f"polars {p.width} columns vs duckdb {d.width}"
    if p.height != d.height:
        return "rows", f"polars {p.height} rows vs duckdb {d.height}"
    p_norm, d_norm = normalise(p), normalise(d)
    p_norm.columns = d_norm.columns
    if first_mismatch(p_norm, d_norm) is None:
        return "match", ""
    if (col := first_mismatch(sort_all(p_norm), sort_all(d_norm))) is None:
        return "match_unordered", ""
    return "values", f"first differing column: {col}"


def cpu_name() -> str:
    if sys.platform == "darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()  # fmt: skip
    try:
        lscpu = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        lscpu = ""
    for line in lscpu.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def polars_is_local_build() -> bool:
    import polars._plr

    return "site-packages" not in Path(polars._plr.__file__).parts


def machine_info() -> dict[str, Any]:
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "system": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "memory_gb": round(mem / 2**30, 1),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "polars_local_build": polars_is_local_build(),
        "polars_threads": pl.thread_pool_size(),
        "duckdb": duckdb.__version__,
        "duckdb_threads": duckdb.connect()
        .sql("SELECT current_setting('threads')")
        .fetchone()[0],  # type: ignore[index]
    }


def run(args: argparse.Namespace) -> None:
    data = generate(args.sf, args.data_dir)
    work = Path(tempfile.mkdtemp(prefix="tpcds-run-"))
    names = fetch_queries(work / "queries")
    if args.queries:
        names = [q if q.startswith("q") else f"q{q}" for q in args.queries.split(",")]

    record: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "config": {
            "sf": args.sf,
            "repeat": args.repeat,
            "timeout_s": args.timeout,
            "polars_engine": args.polars_engine,
        },
        "queries": [],
    }
    print(json.dumps(record["machine"], indent=2), flush=True)
    if record["machine"]["polars_local_build"]:
        print("WARNING: local Polars build; timings are only valid for a release build")
    for q in names:
        res = {e: run_engine(e, q, args, data, work) for e in ENGINES}
        if all(res[e]["ok"] for e in ENGINES):
            try:
                status, detail = compare(
                    pl.read_parquet(work / f"{q}.polars.parquet"),
                    pl.read_parquet(work / f"{q}.duckdb.parquet"),
                )
            except Exception as e:
                status, detail = "compare_error", f"{type(e).__name__}: {e}"[:300]
        else:
            failed = next(e for e in ENGINES if not res[e]["ok"])
            status, detail = f"{failed}_error", res[failed]["error"][:300]
        record["queries"].append(
            {"query": q, "status": status, "detail": detail}
            | {
                e: {k: v for k, v in res[e].items() if k != "traceback"}
                for e in ENGINES
            }
        )
        t = {e: min(res[e]["times_s"], default=math.nan) for e in ENGINES}
        print(
            f"{q:>4} {status:<16} polars {t['polars']:8.3f}s  duckdb {t['duckdb']:8.3f}s"
            f"  {detail[:120]}",
            flush=True,
        )
    args.out.write_text(json.dumps(record, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {args.out}")
    report([args.out])


def best_times(rec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        r["query"]: (min(r["polars"]["times_s"]), min(r["duckdb"]["times_s"]))
        for r in rec["queries"]
        if r["polars"]["times_s"] and r["duckdb"]["times_s"]
    }


def report(paths: list[Path]) -> None:
    recs = [json.loads(p.read_text()) for p in paths]
    for p, rec in zip(paths, recs, strict=True):
        m, c = rec["machine"], rec["config"]
        times = best_times(rec)
        tp = sum(t[0] for t in times.values())
        td = sum(t[1] for t in times.values())
        geo = math.exp(sum(math.log(a / b) for a, b in times.values()) / len(times))
        statuses: dict[str, int] = {}
        for r in rec["queries"]:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        print(f"\n{p.name}: {m['cpu']} ({m['arch']}, {m['logical_cpus']} cpus)")
        print(f"  sf{c['sf']}, polars {m['polars']}, duckdb {m['duckdb']}")
        print(f"  {statuses}")
        for r in rec["queries"]:
            if r["status"] not in ("match", "match_unordered"):
                print(f"    {r['query']:>4} {r['status']}: {r['detail'][:120]}")
        print(f"  total polars {tp:.2f}s  duckdb {td:.2f}s  ({tp / td:.2f}x)")
        faster = sum(a < b for a, b in times.values())
        print(
            f"  geomean polars/duckdb {geo:.2f}x, polars faster on {faster}/{len(times)}"
        )

    if len(recs) > 1:
        all_times = [best_times(r) for r in recs]
        common = sorted(
            set.intersection(*(set(t) for t in all_times)), key=lambda q: int(q[1:])
        )
        print("\npolars/duckdb time ratio per query")
        print("query " + " ".join(f"{p.stem[:14]:>14}" for p in paths))
        for q in common:
            print(
                f"{q:>5} " + " ".join(f"{t[q][0] / t[q][1]:14.2f}" for t in all_times)
            )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        engine, sql, data, out, repeat, polars_engine = sys.argv[2:8]
        worker(engine, Path(sql), Path(data), Path(out), int(repeat), polars_engine)
        return

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate", help="only generate the parquet data")
    run_p = sub.add_parser("run", help="generate data if missing, then benchmark")
    for p in (gen, run_p):
        p.add_argument("--sf", default="1", help="scale factor (default 1)")
        p.add_argument("--data-dir", type=Path, required=True)
    run_p.add_argument("--out", type=Path, required=True, help="results JSON")
    run_p.add_argument(
        "--repeat", type=int, default=3, help="runs per query; min reported"
    )
    run_p.add_argument("--timeout", type=int, default=900, help="seconds per query")
    run_p.add_argument("--polars-engine", default="auto")
    run_p.add_argument("--queries", help="comma-separated subset, e.g. 1,49,q83")
    rep = sub.add_parser("report", help="summarise and compare results JSON files")
    rep.add_argument("files", type=Path, nargs="+")

    args = parser.parse_args()
    if args.cmd == "generate":
        print(generate(args.sf, args.data_dir))
    elif args.cmd == "run":
        run(args)
    else:
        report(args.files)


if __name__ == "__main__":
    main()
