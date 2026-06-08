"""triton-inspector: Capture IR at every pipeline stage with optional diff.

Usage:
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir --diff
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir --out ./ir_dumps
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir --passes
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir --stats
    python .github\\skills\\triton-director\\scripts\\inspector.py test.ttir --json

Captures IR at each Vulkan pipeline stage boundary and optionally shows
diffs between consecutive stages. Also supports per-pass IR capture within
each stage (--passes) and dialect operation statistics (--stats).
"""
import argparse, difflib, json, os, re, sys, time
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from _common import load_ttir, get_ir_text, count_ops, get_vulkan_backend, get_stage_defs
from triton._C.libtriton import ir, passes


def capture_stages(ttir_path, per_pass=False):
    """Run the Vulkan pipeline and capture IR at each stage boundary.

    Returns list of dicts: [{"name": str, "ir": str, "time_ms": float, "passes": [...]}]
    """
    m, c = load_ttir(ttir_path)

    stages = []

    # Capture initial TTIR
    stages.append({
        "name": "input",
        "ir": get_ir_text(m),
        "time_ms": 0,
        "passes": [],
    })

    stage_defs = get_stage_defs()

    for stage_name, pass_list in stage_defs:
        pass_snapshots = []
        stage_error = None

        if per_pass:
            # Run each pass individually and capture IR after each
            for pass_name, add_pass_fn in pass_list:
                pm = ir.pass_manager(m.context)
                add_pass_fn(pm)
                t0 = time.perf_counter()
                try:
                    pm.run(m, f'{stage_name}.{pass_name}')
                except RuntimeError as e:
                    pass_snapshots.append({
                        "name": pass_name,
                        "ir": get_ir_text(m),
                        "time_ms": (time.perf_counter() - t0) * 1000,
                        "error": str(e),
                    })
                    stage_error = str(e)
                    break
                elapsed = (time.perf_counter() - t0) * 1000
                pass_snapshots.append({
                    "name": pass_name,
                    "ir": get_ir_text(m),
                    "time_ms": elapsed,
                })

            stage_time = sum(p["time_ms"] for p in pass_snapshots)
        else:
            # Run all passes in one PassManager (normal mode)
            pm = ir.pass_manager(m.context)
            for _, add_pass_fn in pass_list:
                add_pass_fn(pm)
            t0 = time.perf_counter()
            try:
                pm.run(m, stage_name)
            except RuntimeError as e:
                stage_error = str(e)
            stage_time = (time.perf_counter() - t0) * 1000

        entry = {
            "name": stage_name,
            "ir": get_ir_text(m),
            "time_ms": stage_time,
            "passes": pass_snapshots,
        }
        if stage_error:
            entry["error"] = stage_error
        stages.append(entry)

        if stage_error:
            break

    return stages


def print_diff(ir_before, ir_after, name_before, name_after, context=3):
    """Print unified diff between two IR texts."""
    a = ir_before.splitlines(keepends=True)
    b = ir_after.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a, b, fromfile=name_before, tofile=name_after, n=context))
    if not diff:
        print(f"  (no changes between {name_before} and {name_after})")
    else:
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                sys.stdout.write(f"\033[32m{line}\033[0m")
            elif line.startswith('-') and not line.startswith('---'):
                sys.stdout.write(f"\033[31m{line}\033[0m")
            else:
                sys.stdout.write(line)


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(
        description="Inspect Triton Vulkan pipeline — capture IR at each stage")
    parser.add_argument("ttir", help="Path to .ttir input file")
    parser.add_argument("--diff", action="store_true",
                        help="Show diffs between consecutive stages")
    parser.add_argument("--passes", action="store_true",
                        help="Capture IR after each individual pass (slower)")
    parser.add_argument("--stats", action="store_true",
                        help="Show operation statistics per stage")
    parser.add_argument("--out", metavar="DIR",
                        help="Write IR files to directory")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args()

    stages = capture_stages(args.ttir, per_pass=args.passes)

    if args.json:
        # JSON output (omit large IR text, include stats)
        output = []
        for s in stages:
            entry = {
                "name": s["name"],
                "time_ms": round(s["time_ms"], 2),
                "ir_lines": len(s["ir"].splitlines()),
                "ir_bytes": len(s["ir"]),
                "ops": count_ops(s["ir"]),
            }
            if s.get("error"):
                entry["error"] = s["error"]
            if s["passes"]:
                entry["passes"] = [{
                    "name": p["name"],
                    "time_ms": round(p["time_ms"], 2),
                    "ir_lines": len(p["ir"].splitlines()),
                } for p in s["passes"]]
            output.append(entry)
        print(json.dumps(output, indent=2))
        return

    # Table header
    print(f"\n{'Stage':<18s} {'Time':>8s} {'Lines':>7s} {'Bytes':>8s}", end="")
    if args.stats:
        print(f"  {'Top Dialects':<40s}", end="")
    print()
    print("-" * (82 if args.stats else 42))

    for s in stages:
        lines = len(s["ir"].splitlines())
        nbytes = len(s["ir"])
        status = " ERROR" if s.get("error") else ""
        print(f"{s['name']:<18s} {s['time_ms']:>6.1f}ms {lines:>7d} {nbytes:>8d}{status}", end="")
        if args.stats:
            ops = count_ops(s["ir"])
            top = ", ".join(f"{k}:{v}" for k, v in list(ops.items())[:4])
            print(f"  {top:<40s}", end="")
        print()

        # Per-pass detail
        if args.passes and s["passes"]:
            for p in s["passes"]:
                plines = len(p["ir"].splitlines())
                perr = " ERROR" if p.get("error") else ""
                print(f"  └─ {p['name']:<14s} {p['time_ms']:>6.1f}ms {plines:>7d}{perr}")

    # Total
    total_ms = sum(s["time_ms"] for s in stages)
    print("-" * (82 if args.stats else 42))
    print(f"{'Total':<18s} {total_ms:>6.1f}ms")

    # Diffs
    if args.diff:
        print("\n" + "=" * 60)
        print("Stage-to-stage diffs:")
        print("=" * 60)
        for i in range(1, len(stages)):
            if stages[i].get("error"):
                print(f"\n--- {stages[i]['name']} FAILED: {stages[i]['error']}")
                break
            print(f"\n--- {stages[i-1]['name']} → {stages[i]['name']} ---")
            print_diff(stages[i-1]["ir"], stages[i]["ir"],
                      stages[i-1]["name"], stages[i]["name"])

    # Write files
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for i, s in enumerate(stages):
            fname = f"{i:02d}_{s['name']}.mlir"
            with open(os.path.join(args.out, fname), "w") as f:
                f.write(s["ir"])
        # Write manifest
        manifest = [{
            "index": i, "name": s["name"],
            "file": f"{i:02d}_{s['name']}.mlir",
            "time_ms": round(s["time_ms"], 2),
            "lines": len(s["ir"].splitlines()),
        } for i, s in enumerate(stages)]
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nIR files written to {args.out}/")


if __name__ == "__main__":
    main()
