"""triton-internals locator: find and explain Triton core components.

A self-contained navigator over Triton's compiler internals, distilled from
the triton-ppd study project. Works WITHOUT triton-ppd present — all data is
in data/index.json (regenerate with generate_index.py when triton-ppd updates).

Usage:
    python locator.py list                     # all components
    python locator.py list --ops               # all TTIR/TTGIR ops
    python locator.py list --encodings         # all encoding attributes
    python locator.py find <query>             # search components by name/text
    python locator.py show <component>         # full detail for a component
    python locator.py op <tt.load|ttg.xxx>     # look up an op
    python locator.py encoding <BlockedEncodingAttr>
    python locator.py dag                       # chapter dependency graph
    python locator.py search <query>            # search everything

Examples:
    python locator.py show LinearLayout
    python locator.py op tt.dot
    python locator.py find layout
    python locator.py search swizzle
"""
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX = os.path.normpath(os.path.join(_HERE, "..", "data", "index.json"))


# ── colors ──────────────────────────────────────────────────────────────
def _c(t, code):
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t
def bold(t):  return _c(t, "1")
def green(t): return _c(t, "32")
def cyan(t):  return _c(t, "36")
def yellow(t):return _c(t, "33")
def dim(t):   return _c(t, "2")
def red(t):   return _c(t, "31")


def load_index():
    if not os.path.isfile(_INDEX):
        print(red(f"Index not found: {_INDEX}"), file=sys.stderr)
        print("Run generate_index.py to create it from triton-ppd.", file=sys.stderr)
        sys.exit(1)
    with open(_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


# ── rendering ───────────────────────────────────────────────────────────
def show_component(c):
    print()
    print(bold(green(c["name"])) + dim(f"   [{c.get('maturity','?')}]  {c.get('chapter','')}"))
    print(dim("  " + c.get("category", "")))
    if c.get("blurb"):
        print(f"  {c['blurb']}")
    if c.get("motivation"):
        print()
        print(bold("  Why it exists:"))
        for line in _wrap(c["motivation"], 76):
            print(f"    {line}")
    if c.get("reference_source"):
        print()
        print(bold("  Upstream source (real Triton):"))
        for a in c["reference_source"]:
            print(f"    {cyan(a)}")
    print()
    print(bold("  Study references (triton-ppd):"))
    if c.get("spec_file"):
        print(f"    spec:     {c['spec_file']}")
    if c.get("reference_chapter"):
        print(f"    chapter:  {c['reference_chapter']}/")
    if c.get("textbook"):
        print(f"    textbook: {c['textbook']}")
    print()


def show_op(o):
    print()
    print(bold(green(o["op"])) + dim(f"   ({o['dialect'].upper()})"))
    print(f"  class:   {o['class']}")
    print(f"  defined: {cyan(o['td'])}  " + dim("(in real Triton .td)"))
    if o.get("traits"):
        print(f"  traits:  {dim(o['traits'])}")
    print(f"  summary: {o['summary']}")
    print()


def show_encoding(e):
    print()
    star = " ⭐" if e.get("in_slice") else ""
    print(bold(green(e["encoding"])) + star)
    print(f"  defined:    {cyan(e['td'])}")
    print(f"  parameters: {e['parameters']}")
    print(f"  trait:      {e.get('trait','')}")
    print(f"  purpose:    {e['purpose']}")
    print()


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width)


# ── commands ────────────────────────────────────────────────────────────
def cmd_list(idx, args):
    if args.ops:
        print(bold("\n  TTIR / TTGIR Ops\n"))
        cur = None
        for o in sorted(idx["ops"], key=lambda x: (x["dialect"], x["op"])):
            if o["dialect"] != cur:
                cur = o["dialect"]
                print(bold(f"\n  {cur.upper()}:"))
            print(f"    {green(o['op']):<34} {dim(o['summary'][:48])}")
        print()
    elif args.encodings:
        print(bold("\n  TTGIR Encoding Attributes\n"))
        for e in idx["encodings"]:
            star = cyan(" *") if e.get("in_slice") else ""
            print(f"    {green(e['encoding']):<40}{star} {dim(e['td'])}")
        print(dim("\n  * = in the triton-ppd 20-chapter slice (SM 7.5-8.9 core)\n"))
    else:
        print(bold("\n  Triton Core Components\n"))
        cur = None
        for c in idx["components"]:
            if c.get("category") != cur:
                cur = c.get("category")
                print(bold(f"\n  {cur}"))
            print(f"    {green(c['name']):<28} {dim(c.get('maturity','')):<8} {c.get('chapter','')}")
        print(dim("\n  Use 'show <name>' for full detail, 'op <tt.xxx>' for an op.\n"))


def cmd_find(idx, args):
    q = args.query.lower()
    hits = [c for c in idx["components"]
            if q in c["name"].lower() or q in c.get("blurb", "").lower()
            or q in c.get("category", "").lower()]
    if not hits:
        print(yellow(f"No component matches '{args.query}'. Try 'search' for ops/encodings too."))
        return
    for c in hits:
        show_component(c)


def cmd_show(idx, args):
    q = args.component.lower()
    # exact, then prefix, then substring
    exact = [c for c in idx["components"] if c["name"].lower() == q]
    if exact:
        show_component(exact[0]); return
    pref = [c for c in idx["components"] if c["name"].lower().startswith(q)]
    sub = [c for c in idx["components"] if q in c["name"].lower()]
    cands = pref or sub
    if not cands:
        print(yellow(f"No component named '{args.component}'."))
        print(dim("  Try: python locator.py list"))
        return
    if len(cands) > 1:
        print(yellow(f"Multiple matches for '{args.component}':"))
        for c in cands:
            print(f"    {c['name']}")
        return
    show_component(cands[0])


def cmd_op(idx, args):
    q = args.op.lower().lstrip(".")
    # accept "tt.load", "load", "ttg.convert_layout"
    hits = [o for o in idx["ops"]
            if o["op"].lower() == q or o["op"].lower().split(".")[-1] == q.split(".")[-1]
            and (("." not in q) or o["op"].lower() == q)]
    if not hits:
        hits = [o for o in idx["ops"] if q in o["op"].lower()]
    if not hits:
        print(yellow(f"No op matches '{args.op}'."))
        return
    for o in hits:
        show_op(o)


def cmd_encoding(idx, args):
    q = args.name.lower()
    hits = [e for e in idx["encodings"] if q in e["encoding"].lower()]
    if not hits:
        print(yellow(f"No encoding matches '{args.name}'."))
        return
    for e in hits:
        show_encoding(e)


def cmd_dag(idx, args):
    dag = idx.get("chapter_dag", "")
    if not dag:
        print(yellow("No chapter DAG in index."))
        return
    print(bold("\n  Triton Compiler — Component Dependency Graph (from triton-ppd)\n"))
    for line in dag.splitlines():
        print("  " + line)
    print()


def cmd_search(idx, args):
    q = args.query.lower()
    comp = [c for c in idx["components"]
            if q in json.dumps(c, ensure_ascii=False).lower()]
    ops = [o for o in idx["ops"] if q in json.dumps(o, ensure_ascii=False).lower()]
    encs = [e for e in idx["encodings"] if q in json.dumps(e, ensure_ascii=False).lower()]
    if not (comp or ops or encs):
        print(yellow(f"Nothing matches '{args.query}'."))
        return
    if comp:
        print(bold(f"\n  Components ({len(comp)}):"))
        for c in comp:
            print(f"    {green(c['name']):<28} {dim(c.get('chapter',''))}")
    if ops:
        print(bold(f"\n  Ops ({len(ops)}):"))
        for o in ops:
            print(f"    {green(o['op']):<28} {dim(o['summary'][:50])}")
    if encs:
        print(bold(f"\n  Encodings ({len(encs)}):"))
        for e in encs:
            print(f"    {green(e['encoding']):<40} {dim(e['td'])}")
    print()


def main():
    # Portable UTF-8 stdout (works on every platform; no win32-specific check).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="triton-internals",
        description="Locate and explain Triton core components",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List components (or --ops / --encodings)")
    p_list.add_argument("--ops", action="store_true", help="List TTIR/TTGIR ops")
    p_list.add_argument("--encodings", action="store_true", help="List encoding attributes")

    p_find = sub.add_parser("find", help="Search components by name/text")
    p_find.add_argument("query")

    p_show = sub.add_parser("show", help="Full detail for one component")
    p_show.add_argument("component")

    p_op = sub.add_parser("op", help="Look up a TTIR/TTGIR op")
    p_op.add_argument("op")

    p_enc = sub.add_parser("encoding", help="Look up an encoding attribute")
    p_enc.add_argument("name")

    sub.add_parser("dag", help="Print the chapter dependency graph")

    p_search = sub.add_parser("search", help="Search everything (components/ops/encodings)")
    p_search.add_argument("query")

    args = parser.parse_args()
    idx = load_index()

    dispatch = {
        "list": cmd_list, "find": cmd_find, "show": cmd_show,
        "op": cmd_op, "encoding": cmd_encoding, "dag": cmd_dag,
        "search": cmd_search,
    }
    if args.command in dispatch:
        dispatch[args.command](idx, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
