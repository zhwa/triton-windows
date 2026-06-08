"""triton-timer: Per-pass timing for Triton Vulkan pipeline.

Usage:
    python .github\\skills\\triton-director\\scripts\\timer.py test.ttir
    python .github\\skills\\triton-director\\scripts\\timer.py test.ttir --format csv
    python .github\\skills\\triton-director\\scripts\\timer.py test.ttir --format json
    python .github\\skills\\triton-director\\scripts\\timer.py test.ttir --runs 5
    python .github\\skills\\triton-director\\scripts\\timer.py test_*.ttir

Reports wall time per pass within each pipeline stage. With --runs,
averages over multiple compilations for more stable numbers.

Note: Each pass runs in its own PassManager, so per-pass times include
PM creation overhead and won't sum exactly to batched execution time.
Use inspector.py (without --passes) for representative total stage times.
"""
import argparse, glob, json, os, sys, time
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from _common import load_ttir, get_ir_text, get_vulkan_backend, get_stage_defs
from triton._C.libtriton import ir, passes


def time_pipeline(ttir_path):
    """Run the Vulkan pipeline with per-pass timing.

    Returns list of (stage, pass_name, time_ms) tuples.
    """
    m, c = load_ttir(ttir_path)

    results = []
    stage_defs = get_stage_defs()

    for stage_name, pass_list in stage_defs:
        for pass_name, add_pass_fn in pass_list:
            pm = ir.pass_manager(m.context)
            add_pass_fn(pm)
            t0 = time.perf_counter()
            try:
                pm.run(m, f'{stage_name}.{pass_name}')
            except RuntimeError as e:
                results.append((stage_name, pass_name, -1, str(e)))
                return results
            elapsed = (time.perf_counter() - t0) * 1000
            results.append((stage_name, pass_name, elapsed, None))

    return results


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(
        description="Per-pass timing for Triton Vulkan pipeline")
    parser.add_argument("ttir", nargs="+", help="Path(s) to .ttir input file(s)")
    parser.add_argument("--format", choices=["table", "csv", "json"],
                        default="table", help="Output format")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs to average (default: 1)")
    args = parser.parse_args()

    # Expand globs on Windows
    paths = []
    for p in args.ttir:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])

    all_results = {}
    for path in paths:
        kernel = os.path.splitext(os.path.basename(path))[0]
        run_results = []
        for _ in range(args.runs):
            run_results.append(time_pipeline(path))
        # Average across runs
        avg = []
        for i in range(len(run_results[0])):
            stage, name, _, err = run_results[0][i]
            times = [r[i][2] for r in run_results if i < len(r)]
            valid = [t for t in times if t >= 0]
            avg_ms = sum(valid) / len(valid) if valid else -1
            avg.append((stage, name, avg_ms, err))
        all_results[kernel] = avg

    if args.format == "json":
        output = {}
        for kernel, results in all_results.items():
            output[kernel] = [{
                "stage": s, "pass": p, "time_ms": round(t, 3),
                **({"error": e} if e else {}),
            } for s, p, t, e in results]
        print(json.dumps(output, indent=2))

    elif args.format == "csv":
        print("kernel,stage,pass,time_ms")
        for kernel, results in all_results.items():
            for stage, name, ms, _ in results:
                print(f"{kernel},{stage},{name},{ms:.3f}")

    else:  # table
        for kernel, results in all_results.items():
            total = sum(t for _, _, t, _ in results if t >= 0)
            print(f"\n{kernel} (total: {total:.1f}ms, {args.runs} run{'s' if args.runs > 1 else ''})")
            print(f"  {'Stage':<15s} {'Pass':<30s} {'Time':>8s} {'%':>6s}")
            print(f"  {'-'*15} {'-'*30} {'-'*8} {'-'*6}")

            cur_stage = None
            stage_total = 0
            for stage, name, ms, err in results:
                if stage != cur_stage:
                    if cur_stage is not None:
                        print(f"  {'':15s} {'STAGE TOTAL':<30s} {stage_total:>6.1f}ms {stage_total/total*100:>5.1f}%")
                        print()
                    cur_stage = stage
                    stage_total = 0

                if err:
                    print(f"  {stage:<15s} {name:<30s} {'ERROR':>8s}")
                else:
                    pct = ms / total * 100 if total > 0 else 0
                    bar = "█" * int(pct / 2)
                    print(f"  {stage:<15s} {name:<30s} {ms:>6.1f}ms {pct:>5.1f}% {bar}")
                    stage_total += ms

            if cur_stage is not None:
                print(f"  {'':15s} {'STAGE TOTAL':<30s} {stage_total:>6.1f}ms {stage_total/total*100:>5.1f}%")
            print(f"  {'-'*62}")
            print(f"  {'':15s} {'TOTAL':<30s} {total:>6.1f}ms")


if __name__ == "__main__":
    main()
