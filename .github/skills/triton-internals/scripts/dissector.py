"""triton-dissect: surgical dissection of the Triton lowering pipeline.

Locate the exact hook points (symbol, file, Python binding, signature) at every
lowering stage / pass / function so you can set breakpoints, bind to Python
tests, redirect API calls, or manipulate arguments for perf experiments.

Cross-platform (pure Python, no OS-specific APIs) and version-independent:
function/class SYMBOLS are the stable key; line numbers are resolved on demand
from the actual source tree (--source <root>, $TRITON_SRC, or auto-detected by
walking up from the current directory). Baked snapshot lines are advisory only.

Self-contained — reads data/pipeline.json (regenerate with generate_pipeline.py
when triton-ppd updates). Works WITHOUT triton-ppd present.

Usage:
    python dissector.py [--source ROOT] [--backend NAME] <command> ...

Commands:
    stages                         # the 5 lowering stages + pass counts
    stage <name>                   # one stage's ordered passes + bindings
    pass <name>                    # full hook sheet for a pass
    func <name>                    # one function's hook detail
    funcs [--core]                 # all (or prime) breakpoint targets
    trace <concept>                # a concept's lifecycle across passes
    hooks                          # every core intervention point
    snippet <pass|stage|func> <name> [--action ...]   # ready-to-run hook code
    search <query>                 # search passes + functions + concepts

Examples:
    python dissector.py stages
    python dissector.py --source ../triton trace convert_layout
    python dissector.py pass remove_layout_conversions
    python dissector.py snippet stage make_ttgir --action wrap
    python dissector.py snippet pass remove_layout_conversions --action run
"""
import argparse, json, os, re, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.normpath(os.path.join(_HERE, "..", "data", "pipeline.json"))

# resolved at runtime in main(); module-global so command fns can use it
SOURCE_ROOT = None
BACKEND = {"vendor": "nvidia", "cls": "CUDABackend"}


# ── portable color (works on Linux/macOS/Windows terminals) ────────────────
def _supports_color():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
def _c(t, code):
    return f"\033[{code}m{t}\033[0m" if _supports_color() else t
def bold(t):  return _c(t, "1")
def green(t): return _c(t, "32")
def cyan(t):  return _c(t, "36")
def yellow(t):return _c(t, "33")
def dim(t):   return _c(t, "2")
def red(t):   return _c(t, "31")
def mag(t):   return _c(t, "35")


def load():
    if not os.path.isfile(_PIPE):
        print(red(f"Pipeline index not found: {_PIPE}"), file=sys.stderr)
        print("Run generate_pipeline.py to create it from triton-ppd.", file=sys.stderr)
        sys.exit(1)
    with open(_PIPE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── source-root detection + on-demand line resolution (version independence) ─
def find_source_root(explicit):
    """Find a Triton source checkout to resolve current line numbers from.

    Order: --source, $TRITON_SRC, then auto-detect by walking up from CWD and
    from this script looking for a Triton-shaped tree. Returns None if none.
    Fully cross-platform (os.path only).
    """
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("TRITON_SRC"):
        cands.append(os.environ["TRITON_SRC"])
    for start in (os.getcwd(), _HERE):
        d = start
        for _ in range(10):
            if (os.path.isdir(os.path.join(d, "include", "triton"))
                    or os.path.isdir(os.path.join(d, "lib", "Dialect", "TritonGPU"))):
                cands.append(d)
                break
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def _symbol_base(symbol):
    return symbol.split("::")[-1].split("(")[0].strip()


def resolve_line(rel_file, symbol):
    """Find the CURRENT line of a symbol in the live source tree (or None).

    Symbol-based => robust to version drift. Cross-platform file IO.
    """
    if not SOURCE_ROOT or not rel_file:
        return None
    path = os.path.join(SOURCE_ROOT, *rel_file.split("/"))
    if not os.path.isfile(path):
        return None
    base = _symbol_base(symbol)
    if not base:
        return None
    defpat = re.compile(r"\b" + re.escape(base) + r"\s*\(")
    first = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if base in line:
                    if first is None:
                        first = i
                    if defpat.search(line):
                        return i
    except OSError:
        return None
    return first


def loc(rel_file, symbol, snapshot_line=None):
    """Render a location string: live `file:line` if resolvable, else advisory."""
    if not rel_file:
        return symbol or ""
    live = resolve_line(rel_file, symbol or "")
    base = os.path.basename(rel_file)
    if live:
        return f"{base}:{live}"
    if snapshot_line:
        return f"{base}:~{snapshot_line}" + dim(" (snapshot)")
    return base


# ── command: stages ────────────────────────────────────────────────────────
def cmd_stages(idx, args):
    print(bold("\n  Triton Lowering Pipeline — 5 Stages\n"))
    print(dim("  Each stage is a backend method you can wrap or override in Python.\n"))
    for i, s in enumerate(idx["stages"], 1):
        method_line = resolve_line(s["source"], s["backend_method"].split(".")[-1])
        loc_s = f"  {s['source']}"
        if method_line:
            loc_s += f":{method_line}"
        elif s.get("snapshot_lines"):
            loc_s += f":~{s['snapshot_lines'][0]}"
        print(f"  {cyan(str(i))}. {bold(green(s['name']))}  "
              + dim(f"({s['pass_count']} passes)"))
        print(f"      method:  {s['backend_method']}{dim(loc_s)}")
    print(dim("\n  Next: 'stage <name>' for the pass list, 'pass <name>' for hooks.\n"))


# ── command: stage ─────────────────────────────────────────────────────────
def cmd_stage(idx, args):
    q = args.name.strip()
    s = next((s for s in idx["stages"] if s["name"] == q
              or s["name"] == "make_" + q.replace("make_", "")), None)
    if not s:
        print(yellow(f"No stage '{args.name}'. Try: stages"))
        return
    print()
    print(bold(green(s["name"])) + dim(f"   {s['pass_count']} passes   "
          f"{s['backend_method']}  {s['source']}"))
    print()
    pass_by_name = {p["name"]: p for p in idx["passes"] if p["stage"] == s["name"]}
    seen = {}
    for i, pname in enumerate(s["passes"], 1):
        seen[pname] = seen.get(pname, 0) + 1
        p = pass_by_name.get(pname, {})
        bind = p.get("python_binding") or dim("(plugin/backend pass)")
        total = s["passes"].count(pname)
        runtag = dim(f"  [run {seen[pname]}/{total}]") if total > 1 else ""
        star = cyan(" *") if p.get("cpp_file") else "  "
        print(f"  {i:>2}.{star} {green(pname):<46}{runtag}")
        print(f"        {dim(p.get('purpose',''))}")
        extra = dim(f"   {p.get('sm','')}  {p.get('chapter','')}") if p.get("sm") else ""
        print(f"        bind: {bind}{extra}")
    print(dim("\n  * = has C++ key-function hooks. Use 'pass <name>' to see them.\n"))


# ── command: pass ──────────────────────────────────────────────────────────
def _norm_pass(q):
    q = q.strip()
    return q if q.startswith("add_") else "add_" + q


def _resolve_keyfns(idx, p):
    cpp_bn = os.path.basename(p["cpp_file"]) if p.get("cpp_file") else None
    out = [fn for fn in idx["functions"]
           if cpp_bn and os.path.basename(fn.get("file", "")) == cpp_bn
           and fn["name"] in p.get("key_functions", [])]
    order = {n: i for i, n in enumerate(p.get("key_functions", []))}
    out.sort(key=lambda f: order.get(f["name"], 999))
    return out


def cmd_pass(idx, args):
    q = _norm_pass(args.name)
    cands = [p for p in idx["passes"] if p["name"] == q]
    if not cands:
        cands = [p for p in idx["passes"] if args.name.lower() in p["name"].lower()]
    if not cands:
        print(yellow(f"No pass '{args.name}'. Try: stage make_ttgir"))
        return
    for p in cands:
        _print_pass(idx, p)


def _print_pass(idx, p):
    print()
    runs = f"   runs {p['runs']}x" if p.get("runs", 1) > 1 else ""
    print(bold(green(p["name"])) + dim(f"   [{p['stage']}]{runs}"))
    print(f"  purpose:  {p.get('purpose','')}")
    if p.get("sm") or p.get("chapter"):
        print(f"  applies:  {dim(p.get('sm',''))}    study: {dim(p.get('chapter',''))}")
    print()
    print(bold("  Hooks:"))
    if p.get("python_binding"):
        print(f"    python:   {cyan(p['python_binding'])}(pm)   "
              + dim("# call/redirect from a PassManager"))
    else:
        print(f"    python:   {dim('(backend/plugin pass — not in core passes.* bindings)')}")
    if p.get("cpp_file"):
        print(f"    C++ file: {cyan(p['cpp_file'])}   " + dim("# set breakpoints here"))
    if p.get("runs", 1) > 1:
        print(f"    runs:     {yellow(str(p['runs']) + 'x')} in {p['stage']}  "
              + dim("# same pass re-applied as a fixed-point"))
    keyfns = _resolve_keyfns(idx, p)
    if keyfns:
        print()
        print(bold("  Breakpoint targets (key functions):"))
        for f in keyfns:
            star = mag(" *") if f.get("core") else "  "
            print(f"   {star} {green(f['name']):<44} "
                  f"{cyan(loc(f.get('file'), f['name'], f.get('snapshot_line')))}")
            if f.get("purpose"):
                print(f"        {dim(f['purpose'][:74])}")
    print(dim("\n  Tip: 'snippet pass " + p["name"].replace("add_", "")
              + " --action run' for ready-to-run code.\n"))


# ── command: func / funcs ──────────────────────────────────────────────────
def cmd_func(idx, args):
    q = args.name.strip().lower()
    hits = [f for f in idx["functions"] if f["name"].lower() == q]
    if not hits:
        hits = [f for f in idx["functions"] if q in f["name"].lower()]
    if not hits:
        print(yellow(f"No function matching '{args.name}'."))
        return
    for f in hits:
        _print_func(idx, f)


def _print_func(idx, f):
    print()
    star = mag(" *core") if f.get("core") else ""
    print(bold(green(f["name"])) + star)
    print(f"  location:  {cyan(loc(f.get('file'), f['name'], f.get('snapshot_line')))}"
          + dim("   # breakpoint here (symbol-resolved)"))
    if f.get("signature"):
        print(f"  signature: {dim(f['signature'])}")
    if f.get("purpose"):
        print(f"  purpose:   {f['purpose']}")
    owners = [p for p in idx["passes"]
              if p.get("cpp_file")
              and os.path.basename(p["cpp_file"]) == os.path.basename(f.get("file", ""))]
    if owners:
        print("  in pass:   " + ", ".join(green(p["name"]) for p in owners)
              + dim(f"   [{owners[0]['stage']}]"))
    print()


def cmd_funcs(idx, args):
    fns = [f for f in idx["functions"] if f.get("core")] if args.core else idx["functions"]
    print(bold(f"\n  {'Core ' if args.core else ''}Functions ({len(fns)})\n"))
    cur = None
    for f in sorted(fns, key=lambda x: (x.get("file", ""), x.get("snapshot_line") or 0)):
        if f.get("file") != cur:
            cur = f.get("file")
            print(bold(f"\n  {cur}"))
        star = mag("*") if f.get("core") else " "
        print(f"   {star} {green(f['name']):<44} "
              f"{dim(loc(f.get('file'), f['name'], f.get('snapshot_line')))}")
    print()


# ── command: trace ─────────────────────────────────────────────────────────
def cmd_trace(idx, args):
    q = args.concept.strip().lower().replace("ttg.", "").replace("tt.", "")
    concepts = idx.get("concepts", {})
    key = next((k for k in concepts if k == q or q in k or k in q), None)
    if not key:
        print(yellow(f"No concept trace for '{args.concept}'. Available: "
                     + ", ".join(concepts)))
        return
    c = concepts[key]
    print()
    print(bold(green(key)) + dim(f"   {c.get('op','')}"))
    print(f"  {c.get('what','')}")

    def section(title, items, color):
        if not items:
            return
        print()
        print(bold(color(f"  {title}:")))
        for it in items:
            run = dim(f"  (runs {it['runs']}x)") if it.get("runs") else ""
            print(f"    {green(it['site'])}{run}")
            if it.get("file"):
                print(f"      {cyan(loc(it['file'], it['site']))}")
            if it.get("how"):
                print(f"      {dim(it['how'])}")

    section("Produced by", c.get("produced_by"), yellow)
    section("Eliminated by", c.get("eliminated_by"), green)
    section("Lowered by (to LLVM/PTX)", c.get("lowered_by"), cyan)
    if c.get("notes"):
        print()
        print(dim(f"  Note: {c['notes']}"))
    print()


# ── command: hooks ─────────────────────────────────────────────────────────
def cmd_hooks(idx, args):
    print(bold("\n  Prime Intervention Points (core functions across the pipeline)\n"))
    core = [f for f in idx["functions"] if f.get("core")]
    cur = None
    for f in sorted(core, key=lambda x: (x.get("file", ""), x.get("snapshot_line") or 0)):
        if f.get("file") != cur:
            cur = f.get("file")
            print(bold(f"\n  {cur}"))
        print(f"   {mag('*')} {green(f['name']):<42} "
              f"{dim(loc(f.get('file'), f['name'], f.get('snapshot_line'))):<14} "
              f"{dim(f.get('purpose','')[:44])}")
    print(dim("\n  Symbol-based locations resolve against your source tree (--source).\n"))


# ── command: snippet ───────────────────────────────────────────────────────
def cmd_snippet(idx, args):
    kind = args.kind
    if kind == "pass":
        _snippet_pass(idx, args.name, args.action)
    elif kind == "stage":
        _snippet_stage(idx, args.name, args.action)
    elif kind == "func":
        _snippet_func(idx, args.name, args.action)
    else:
        print(yellow("kind must be one of: pass | stage | func"))


def _snippet_pass(idx, name, action):
    q = _norm_pass(name)
    p = next((p for p in idx["passes"] if p["name"] == q), None)
    if not p:
        p = next((p for p in idx["passes"] if name.lower() in p["name"].lower()), None)
    if not p:
        print(yellow(f"No pass '{name}'."))
        return
    action = action or "run"
    print(bold(f"\n# snippet: pass {p['name']} ({action}) — copy/paste\n"))
    if action == "run":
        if not p.get("python_binding"):
            print(dim("# This pass has no core passes.* binding (backend/plugin pass)."))
            print(dim("# Add it via its backend module, or run the whole stage instead."))
            return
        print("from triton._C.libtriton import ir, passes")
        print("# mod = <your module at the right stage> (e.g. a parsed .ttgir)")
        print("pm = ir.pass_manager(mod.context)")
        print(f"{p['python_binding']}(pm)")
        print("pm.run(mod)")
        print("print(mod)  # inspect the result")
    elif action == "break":
        keyfns = _resolve_keyfns(idx, p)
        sym = next((f["name"] for f in keyfns if f.get("core")),
                   keyfns[0]["name"] if keyfns else p["name"])
        base = _symbol_base(sym)
        print(dim("# Debugger breakpoint (build Triton with debug info)."))
        print(dim("# Symbol-based = version independent (no line numbers).\n"))
        print("# gdb:")
        print(f"break {base}")
        print("# lldb:")
        print(f"breakpoint set --name {base}")
    else:
        print(yellow("pass actions: run | break"))


def _snippet_stage(idx, name, action):
    s = next((s for s in idx["stages"]
              if s["name"] == name or s["name"] == "make_" + name.replace("make_", "")), None)
    if not s:
        print(yellow(f"No stage '{name}'."))
        return
    action = action or "wrap"
    method = s["name"]
    cls = BACKEND["cls"]
    vendor = BACKEND["vendor"]
    print(bold(f"\n# snippet: stage {method} ({action}) — copy/paste\n"))
    print(dim(f"# Adjust the import for your backend (vendor='{vendor}', class='{cls}')."))
    print(f"from triton.backends.{vendor}.compiler import {cls}")
    print(f"_orig = {cls}.{method}")
    if action == "wrap":
        print("def _wrap(mod, *a, **k):")
        print(f'    print(">>> before {method}\\n", mod)')
        print("    out = _orig(mod, *a, **k)")
        print(f'    print(">>> after  {method}\\n", out)')
        print("    return out")
    elif action == "time":
        print("import time")
        print("def _wrap(mod, *a, **k):")
        print("    t = time.perf_counter()")
        print("    out = _orig(mod, *a, **k)")
        print(f'    print(f">>> {method} took {{(time.perf_counter()-t)*1e3:.1f}} ms")')
        print("    return out")
    else:
        print(yellow("stage actions: wrap | time"))
        return
    print(f"{cls}.{method} = staticmethod(_wrap)")
    print(dim("\n# Now compile a kernel as usual; the wrapper runs automatically."))


def _snippet_func(idx, name, action):
    q = name.strip().lower()
    f = next((f for f in idx["functions"] if f["name"].lower() == q), None)
    if not f:
        f = next((f for f in idx["functions"] if q in f["name"].lower()), None)
    if not f:
        print(yellow(f"No function '{name}'."))
        return
    base = _symbol_base(f["name"])
    print(bold(f"\n# snippet: func {f['name']} (break) — copy/paste\n"))
    print(dim(f"# Location: {loc(f.get('file'), f['name'], f.get('snapshot_line'))}"))
    print(dim("# Symbol-based breakpoint = version independent.\n"))
    print("# gdb:")
    print(f"break {base}")
    print("# lldb:")
    print(f"breakpoint set --name {base}")
    if f.get("file", "").endswith(".py"):
        print("# Python-side: add at the call site ->  import pdb; pdb.set_trace()")


# ── command: search ────────────────────────────────────────────────────────
def cmd_search(idx, args):
    q = args.query.lower()
    passes = [p for p in idx["passes"] if q in json.dumps(p, ensure_ascii=False).lower()]
    funcs = [f for f in idx["functions"] if q in json.dumps(f, ensure_ascii=False).lower()]
    concepts = [k for k, v in idx.get("concepts", {}).items()
                if q in (k + json.dumps(v, ensure_ascii=False)).lower()]
    if not (passes or funcs or concepts):
        print(yellow(f"Nothing matches '{args.query}'."))
        return
    if passes:
        print(bold(f"\n  Passes ({len(passes)}):"))
        for p in passes:
            print(f"    {green(p['name']):<46} {dim('['+p['stage']+']')}")
    if funcs:
        print(bold(f"\n  Functions ({len(funcs)}):"))
        for f in funcs[:40]:
            print(f"    {green(f['name']):<42} "
                  f"{dim(loc(f.get('file'), f['name'], f.get('snapshot_line')))}")
        if len(funcs) > 40:
            print(dim(f"    ... and {len(funcs)-40} more"))
    if concepts:
        print(bold(f"\n  Concept traces ({len(concepts)}):"))
        for k in concepts:
            print(f"    {green(k)}  " + dim("(use: trace " + k + ")"))
    print()


def main():
    # Portable UTF-8 stdout (works on every platform; no win32-specific check).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="triton-dissect",
        description="Dissect the Triton lowering pipeline into hook points",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", metavar="ROOT",
                        help="Triton source checkout for live line resolution "
                             "(or $TRITON_SRC; auto-detected if omitted)")
    parser.add_argument("--backend", default="nvidia",
                        help="Backend vendor for snippets (default: nvidia)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stages", help="List the 5 lowering stages + pass counts")
    p_stage = sub.add_parser("stage", help="One stage's ordered passes + bindings")
    p_stage.add_argument("name")
    p_pass = sub.add_parser("pass", help="Full hook sheet for a pass")
    p_pass.add_argument("name")
    p_func = sub.add_parser("func", help="One function's hook detail")
    p_func.add_argument("name")
    p_funcs = sub.add_parser("funcs", help="List functions (--core for prime hooks)")
    p_funcs.add_argument("--core", action="store_true")
    p_trace = sub.add_parser("trace", help="A concept's lifecycle across passes")
    p_trace.add_argument("concept")
    sub.add_parser("hooks", help="All core intervention points")
    p_snip = sub.add_parser("snippet", help="Emit ready-to-run hook code")
    p_snip.add_argument("kind", choices=["pass", "stage", "func"])
    p_snip.add_argument("name")
    p_snip.add_argument("--action", default=None,
                        help="pass: run|break  stage: wrap|time  func: break")
    p_search = sub.add_parser("search", help="Search passes + functions + concepts")
    p_search.add_argument("query")

    args = parser.parse_args()

    global SOURCE_ROOT, BACKEND
    SOURCE_ROOT = find_source_root(getattr(args, "source", None))
    vendor = getattr(args, "backend", "nvidia") or "nvidia"
    cls = {"nvidia": "CUDABackend", "cuda": "CUDABackend",
           "vulkan": "VulkanBackend", "amd": "HIPBackend"}.get(
               vendor.lower(), vendor.capitalize() + "Backend")
    BACKEND = {"vendor": vendor, "cls": cls}

    idx = load()
    dispatch = {
        "stages": cmd_stages, "stage": cmd_stage, "pass": cmd_pass,
        "func": cmd_func, "funcs": cmd_funcs, "trace": cmd_trace,
        "hooks": cmd_hooks, "snippet": cmd_snippet, "search": cmd_search,
    }
    if args.command in dispatch:
        dispatch[args.command](idx, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
