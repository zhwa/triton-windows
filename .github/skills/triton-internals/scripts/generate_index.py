"""Generate a self-contained Triton-internals index from the triton-ppd repo.

This script distills triton-ppd's spec sheets, dialect inventory, and chapter
map into a single committed JSON file (`data/index.json`) so the locator works
WITHOUT triton-ppd present at runtime.

Re-run this only when triton-ppd updates (cross-platform; set $TRITON_PPD_DIR
or pass --ppd):

    python generate_index.py --ppd /path/to/triton-ppd

Sources parsed:
    docs/spec/README.md          -> component -> category, spec file, chapter
    docs/spec/<component>.md      -> maturity, reference source anchors, motivation
    docs/analysis/01-dialect-inventory.md -> TTIR/TTGIR ops + encoding attributes
    docs/PROJECT.md              -> chapter DAG (embedded verbatim)
"""
import argparse, json, os, re, sys
from datetime import datetime, timezone


def default_ppd():
    """Cross-platform default location for the triton-ppd repo.

    Order: $TRITON_PPD_DIR, then common relative guesses, else empty
    (caller must pass --ppd). No hardcoded OS-specific drive paths.
    """
    env = os.environ.get("TRITON_PPD_DIR")
    if env:
        return env
    for guess in ("references/triton-ppd", os.path.join("..", "triton-ppd"),
                  "triton-ppd"):
        if os.path.isdir(guess):
            return os.path.abspath(guess)
    return ""


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_spec_readme(ppd):
    """Return list of {name, category, spec_file, chapter} from spec/README.md."""
    path = os.path.join(ppd, "docs", "spec", "README.md")
    text = _read(path)
    components = []
    category = None
    for line in text.splitlines():
        h = re.match(r"^##\s+(.*)", line)
        if h:
            category = h.group(1).strip()
            continue
        # | [file.md](file.md) | `Component` desc | ch.NN |
        m = re.match(r"^\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*(.+?)\s*\|\s*(ch\.\d+)\s*\|", line)
        if m:
            spec_file, _link, desc, chapter = m.groups()
            # component name: first backticked token in desc, else file stem
            nm = re.search(r"`([^`]+)`", desc)
            name = nm.group(1) if nm else os.path.splitext(spec_file)[0]
            components.append({
                "name": name.strip(),
                "category": category or "",
                "spec_file": f"docs/spec/{spec_file}",
                "spec_basename": spec_file,
                "chapter": chapter,
                "blurb": desc.strip(),
            })
    return components


def parse_spec_file(ppd, spec_basename):
    """Extract maturity, reference_source[], motivation from a spec sheet.

    Spec sheets use two header styles, both handled here:
      Style A (meta blockquote): **Maturity**:, **Reference source**:, ## 1. Motivation
      Style B (Identity table):  # ... L3 Spec ..., ## Identity table, ## Purpose
    """
    path = os.path.join(ppd, "docs", "spec", spec_basename)
    if not os.path.isfile(path):
        return {}
    text = _read(path)
    out = {}

    # --- Maturity ---
    mat = re.search(r"\*\*Maturity\*\*:\s*([^\n(]+)", text)
    if mat:
        out["maturity"] = mat.group(1).strip()
    else:
        title = text.splitlines()[0] if text else ""
        tm = re.search(r"\bL([0-3])\b", title)
        if tm:
            out["maturity"] = f"L{tm.group(1)}"

    # --- Reference source anchors ---
    anchors = []
    ref = (re.search(r"\*\*Reference source\*\*:\s*(.+)", text)
           or re.search(r"\*\*Reference\*\*:\s*(.+)", text))
    if ref:
        anchors = re.findall(r"`([^`]+)`", ref.group(1))
    if not anchors:
        # Identity table "| Reference | ... |" row
        idrow = re.search(r"\|\s*Reference\s*\|\s*(.+?)\s*\|", text)
        if idrow:
            cell = idrow.group(1)
            anchors = re.findall(r"`([^`]+)`", cell)
            if not anchors:
                anchors = re.findall(r"(?:lib|include|python|bin)/[\w/.]+", cell)
    if anchors:
        out["reference_source"] = anchors

    # --- Motivation / Purpose ---
    mo = (re.search(r"^##\s+1\.\s*Motivation\s*$(.*?)^##\s", text,
                    re.MULTILINE | re.DOTALL)
          or re.search(r"^##\s+Motivation\s*$(.*?)^##\s", text,
                       re.MULTILINE | re.DOTALL)
          or re.search(r"^##\s+Purpose\s*$(.*?)^##\s", text,
                       re.MULTILINE | re.DOTALL))
    if mo:
        body = mo.group(1).strip()
        paras = [p.strip().replace("\n", " ") for p in body.split("\n\n") if p.strip()]
        out["motivation"] = " ".join(paras[:2])
    return out


def _split_table_rows(text, header_anchor):
    """Yield raw cell lists for a markdown table that follows header_anchor."""
    lines = text.splitlines()
    rows = []
    started = False
    for i, line in enumerate(lines):
        if not started:
            if header_anchor in line:
                started = True
            continue
        if started:
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                # skip the separator row (---|---)
                if all(set(c) <= set("-: ") for c in cells):
                    continue
                # skip the header row (contains "Op name" / "Encoding name")
                rows.append(cells)
            elif line.startswith("## ") and rows:
                break
    return rows


def parse_ops(ppd):
    """Parse TTIR (section A) and TTGIR (section E) ops + encodings (section G)."""
    path = os.path.join(ppd, "docs", "analysis", "01-dialect-inventory.md")
    text = _read(path)
    ops = []

    def collect(section_header, dialect, prefix):
        # isolate the section block
        m = re.search(rf"^{re.escape(section_header)}\s*$(.*?)^##\s",
                      text, re.MULTILINE | re.DOTALL)
        block = m.group(1) if m else ""
        for cells in _split_table_rows(section_header + "\n" + block, "| Op name"):
            if len(cells) < 10:
                continue
            opname, klass, td = cells[0], cells[1], cells[2]
            if opname.lower().startswith("op name"):
                continue
            ops.append({
                "op": f"{prefix}.{opname}",
                "dialect": dialect,
                "class": klass,
                "td": td,
                "traits": cells[3],
                "summary": cells[9],
            })

    collect("## A. TTIR Ops", "ttir", "tt")
    collect("## E. TTGIR Ops", "ttgir", "ttg")

    # Encodings (section G) — different columns
    encs = []
    m = re.search(r"^## G\. TTGIR Encoding Attributes\s*$(.*?)^##\s",
                  text, re.MULTILINE | re.DOTALL)
    block = m.group(1) if m else ""
    for cells in _split_table_rows("| Encoding name\n" + block, "| Encoding name"):
        if len(cells) < 6:
            continue
        name = cells[0].replace("**", "").replace("⭐", "").strip()
        if name.lower().startswith("encoding name"):
            continue
        encs.append({
            "encoding": name,
            "td": cells[1],
            "parameters": cells[2],
            "trait": cells[3],
            "purpose": cells[5],
            "in_slice": "⭐" in cells[0],
        })
    return ops, encs


def find_chapter_dir(ppd, chapter):
    """Map 'ch.08' -> 'ch.08.LinearLayout' (directory name) if present."""
    try:
        for entry in os.listdir(ppd):
            if entry.startswith(chapter + ".") and os.path.isdir(os.path.join(ppd, entry)):
                return entry
    except OSError:
        pass
    return None


def extract_dag(ppd):
    """Pull the Chapter DAG ascii block from PROJECT.md (verbatim)."""
    path = os.path.join(ppd, "docs", "PROJECT.md")
    if not os.path.isfile(path):
        return ""
    text = _read(path)
    m = re.search(r"## 7\. Chapter DAG\s*```(.*?)```", text, re.DOTALL)
    return m.group(1).strip("\n") if m else ""


def main():
    ap = argparse.ArgumentParser(description="Generate triton-internals index from triton-ppd")
    ap.add_argument("--ppd", default=default_ppd(),
                    help="Path to triton-ppd repo (or set $TRITON_PPD_DIR)")
    ap.add_argument("--out", default=None, help="Output index.json path")
    args = ap.parse_args()

    ppd = args.ppd
    if not ppd or not os.path.isdir(ppd):
        print("ERROR: triton-ppd not found. Pass --ppd <path> or set "
              "$TRITON_PPD_DIR.", file=sys.stderr)
        sys.exit(1)

    components = parse_spec_readme(ppd)
    for c in components:
        c.update(parse_spec_file(ppd, c["spec_basename"]))
        cd = find_chapter_dir(ppd, c["chapter"])
        if cd:
            c["reference_chapter"] = cd
            tb = f"book-zh/{cd}.md"
            if os.path.isfile(os.path.join(ppd, tb)):
                c["textbook"] = tb
        del c["spec_basename"]

    ops, encs = parse_ops(ppd)
    dag = extract_dag(ppd)

    index = {
        "generated_from": os.path.abspath(ppd),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Self-contained index distilled from triton-ppd. Regenerate with generate_index.py.",
        "chapter_dag": dag,
        "components": components,
        "ops": ops,
        "encodings": encs,
    }

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "index.json")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out}")
    print(f"  components: {len(components)}")
    print(f"  ops:        {len(ops)}")
    print(f"  encodings:  {len(encs)}")
    print(f"  chapter_dag: {'yes' if dag else 'no'}")


if __name__ == "__main__":
    main()
