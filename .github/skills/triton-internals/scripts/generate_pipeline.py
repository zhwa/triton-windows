"""Generate the Triton lowering-pipeline dissection index from triton-ppd.

Produces a self-contained `data/pipeline.json` (stages -> passes -> functions
+ concept traces) so the dissector works WITHOUT triton-ppd present.

Re-run when triton-ppd updates (cross-platform; set $TRITON_PPD_DIR or pass --ppd):
    python generate_pipeline.py --ppd /path/to/triton-ppd

Sources parsed:
    docs/spec/golden-pass-list.md          -> stages + ordered passes
    docs/analysis/06-orchestrator-skeleton.md -> backend make_* method ranges (advisory)
    docs/analysis/03/04/05-algorithm-skeleton-*.md -> per-file key functions (symbol + advisory line)

Version independence: function/class SYMBOLS are the stable key; line numbers
are stored as advisory `snapshot_line` only. The dissector resolves the current
line on demand from the actual source tree.

Curated maps (verified against triton-windows python/src/passes.cc and spec sheets)
live in this file: pass -> Python binding submodule, pass -> C++ source file,
and the concept-trace table (convert_layout lifecycle, etc.).
"""
import argparse, json, os, re, sys
from datetime import datetime, timezone

DEFAULT_PPD_ENV = "TRITON_PPD_DIR"


def default_ppd():
    """Cross-platform default location for the triton-ppd repo.

    Order: $TRITON_PPD_DIR, then common relative guesses, else empty
    (caller must pass --ppd). No hardcoded OS-specific drive paths.
    """
    env = os.environ.get(DEFAULT_PPD_ENV)
    if env:
        return env
    for guess in ("references/triton-ppd", os.path.join("..", "triton-ppd"),
                  "triton-ppd"):
        if os.path.isdir(guess):
            return os.path.abspath(guess)
    return ""


# ── Curated: pass name -> passes.<submodule>.add_<name> binding ────────────
# Verified against triton-windows python/src/passes.cc.
PASS_SUBMODULE = {
    "add_sccp": "common", "add_symbol_dce": "common", "add_inliner": "common",
    "add_canonicalizer": "common", "add_cse": "common", "add_licm": "common",
    "add_combine": "ttir", "add_reorder_broadcast": "ttir",
    "add_rewrite_tensor_descriptor_to_pointer": "ttir",
    "add_loop_unroll": "ttir", "add_triton_licm": "ttir",
    "add_loop_aware_cse": "ttir", "add_convert_to_ttgpuir": "ttir",
    "add_coalesce": "ttgpuir", "add_optimize_thread_locality": "ttgpuir",
    "add_schedule_loops": "ttgpuir", "add_prefetch": "ttgpuir",
    "add_accelerate_matmul": "ttgpuir", "add_reorder_instructions": "ttgpuir",
    "add_remove_layout_conversions": "ttgpuir",
    "add_pipeline": "ttgpuir", "add_f32_dot_tc": "ttgpuir",
    "add_optimize_dot_operands": "ttgpuir",
    "add_reduce_data_duplication": "ttgpuir",
    "add_allocate_warp_groups": "ttgpuir",
    "add_allocate_shared_memory": "ttgpuir",
    "add_combine_tensor_select_and_if": "ttgpuir",
    "add_optimize_accumulator_init": "ttgpuir",
    "add_fuse_nested_loops": "ttgpuir", "add_coalesce_async_copy": "ttgpuir",
    "add_optimize_partition_warps": "ttgpuir",
    "add_scf_to_cf": "convert", "add_cf_to_llvmir": "convert",
    "add_index_to_llvmir": "convert", "add_arith_to_llvmir": "convert",
    "add_nvvm_to_llvm": "convert",
    "add_di_scope": "llvmir", "add_di_local_variable": "llvmir",
    "add_resolve_auto_encodings": "llvmir",
    "add_infer_coalesced_encodings": "llvmir",
}

# ── Curated: pass name -> C++ source file (joins to parsed functions) ───────
PASS_CPP = {
    "add_convert_to_ttgpuir": "lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp",
    "add_coalesce": "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp",
    "add_remove_layout_conversions": "lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp",
    "add_accelerate_matmul": "lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp",
    "add_optimize_dot_operands": "lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp",
    "add_to_llvmir": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
}

# ── Curated: concept lifecycle traces ──────────────────────────────────────
# Sites are SYMBOLS (function/class) or PASS names — stable across versions.
# `file` is a path only (no line number); the dissector resolves the current
# line on demand from the actual source tree (--source / $TRITON_SRC).
CONCEPTS = {
    "convert_layout": {
        "op": "ttg.convert_layout",
        "what": "Repartition a tensor from one layout encoding to another; "
                "physically lowered through shared memory.",
        "produced_by": [
            {"site": "add_convert_to_ttgpuir",
             "how": "TTIR->TTGIR bridge assigns a default Blocked encoding; "
                    "converts appear at layout boundaries"},
            {"site": "add_accelerate_matmul",
             "how": "installs MMA encodings on dot ops, inserting converts around them"},
            {"site": "LayoutPropagation::getValueAs",
             "file": "lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp",
             "how": "inserts a convert when a value's encoding differs from the target"},
        ],
        "eliminated_by": [
            {"site": "add_remove_layout_conversions", "runs": 3,
             "how": "3-run fixed-point GC: (1) bridge converts, (2) matmul/dot "
                    "converts, (3) late pipeline/async-copy converts"},
            {"site": "LayoutRematerialization::rewriteSlice",
             "file": "lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp",
             "how": "clones the convert's backward slice in the target encoding"},
            {"site": "LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast",
             "file": "lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp",
             "how": "hoists a convert above broadcast / type-extend ops"},
        ],
        "lowered_by": [
            {"site": "applyLinearLayout",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "matrix-vector product computing the layout transform"},
            {"site": "emitIndices",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "emits per-register LLVM indices for the converted tensor"},
        ],
        "notes": "remove_layout_conversions appears 3x in make_ttgir as a "
                 "fixed-point garbage collector for convert noise "
                 "(golden-pass-list section 7).",
    },
    "coalesce": {
        "op": "vectorized tt.load / tt.store (Blocked layout reorder)",
        "what": "Reorder each memory op's Blocked encoding so adjacent lanes "
                "touch contiguous addresses, enabling wide vector loads "
                "(ld.global.v4) instead of scattered single-element accesses.",
        "produced_by": [
            {"site": "add_convert_to_ttgpuir",
             "how": "bridge assigns a generic Blocked encoding, often NOT "
                    "coalesced for the access pattern"},
        ],
        "eliminated_by": [
            {"site": "add_coalesce",
             "file": "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp",
             "how": "picks a per-op Blocked encoding whose fastest dim matches "
                    "the contiguous memory dim; wraps in convert_layout"},
            {"site": "add_coalesce_async_copy",
             "how": "coalesces async-copy traffic on the Ampere path"},
        ],
        "lowered_by": [
            {"site": "getNumConsecutiveInOut",
             "file": "lib/Tools/LinearLayout.cpp",
             "how": "queries the layout for the max contiguous run -> vector width"},
            {"site": "lowerLdSt",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "emits the vectorized load/store using that width"},
        ],
        "notes": "Coalescing is thesis optimization T3a; vector width comes from "
                 "LinearLayout.getNumConsecutiveInOut() on the lane->addr sublayout.",
    },
    "pipeline": {
        "op": "ttg.async_copy_global_to_local + multi-buffered local_alloc",
        "what": "Software-pipeline loop loads: prefetch future iterations into "
                "multi-buffered shared memory with async copies so compute and "
                "memory overlap (Ampere SM 8.x).",
        "produced_by": [
            {"site": "add_assign_latencies",
             "how": "annotates each staged op with a pipeline latency"},
            {"site": "add_schedule_loops",
             "how": "chooses producer/consumer ordering for the schedule"},
        ],
        "eliminated_by": [
            {"site": "add_pipeline",
             "how": "materializes the async pipeline: inserts async_copy + "
                    "async_wait, multi-buffers the shared allocation"},
        ],
        "lowered_by": [
            {"site": "add_coalesce_async_copy",
             "how": "merges async-copy traffic before LLVM lowering"},
            {"site": "lowerLocalLdSt",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "lowers the staged local loads/stores to LLVM"},
        ],
        "notes": "SM 8.x only (Ampere async-copy). Hopper (SM 9+) uses TMA + warp "
                 "specialization instead — out of the SM 7.5-8.9 slice.",
    },
    "shared_memory": {
        "op": "ttg.local_alloc / ttg.local_load / ttg.local_store",
        "what": "Staging tensors in GPU shared memory for cross-thread exchange "
                "(e.g. layout conversion, matmul operands).",
        "produced_by": [
            {"site": "add_accelerate_matmul",
             "how": "MMA operands are staged in shared memory"},
            {"site": "add_pipeline",
             "how": "software pipelining allocates multi-buffered shared memory"},
        ],
        "eliminated_by": [
            {"site": "add_allocate_shared_memory",
             "how": "assigns concrete shared-memory offsets to local_alloc ops"},
        ],
        "lowered_by": [
            {"site": "lowerLocalLdSt",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "lowers local load/store to address calc + LLVM memory ops"},
            {"site": "SharedMemoryObject",
             "file": "lib/Conversion/TritonGPUToLLVM/Utility.cpp",
             "how": "encapsulates shared-memory base pointer and per-lane offsets"},
        ],
        "notes": "Swizzling (SwizzledSharedEncoding) avoids bank conflicts; "
                 "see encoding spec swizzled-shared-encoding.md.",
    },
    "mma": {
        "op": "tt.dot -> NvidiaMmaEncoding",
        "what": "Tensor-core matrix multiply: BlockedEncoding dot operands are "
                "converted to MMA/DotOperand encodings, then lowered to mma.sync PTX.",
        "produced_by": [
            {"site": "add_accelerate_matmul",
             "file": "lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp",
             "how": "selects the MMA instruction shape and installs MmaEncoding"},
            {"site": "add_optimize_dot_operands",
             "how": "normalizes dot operands into DotOperandEncoding"},
        ],
        "eliminated_by": [
            {"site": "add_lower_mma",
             "how": "lowers abstract MMA ops to target mma.sync form (NVIDIA)"},
        ],
        "lowered_by": [
            {"site": "DotOpToLLVM",
             "file": "lib/Conversion/TritonGPUToLLVM/DotOpToLLVM.cpp",
             "how": "emits mma.sync.* PTX inline asm per warp tile"},
        ],
        "notes": "MMAv2 (m16n8k16 fp16/bf16) is the in-slice target for SM 7.5-8.9.",
    },
}


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── parse golden-pass-list: stages + ordered passes ────────────────────────
def parse_golden(ppd):
    text = _read(os.path.join(ppd, "docs", "spec", "golden-pass-list.md"))
    stages = []  # ordered list of (stage_name, [pass dict])
    cur_stage = None
    cur_passes = None

    stage_hdr = re.compile(r"^##\s+\d+\.\s+`?(make_\w+)`?")
    # also handle "make_ptx / make_cubin" combined header
    combo_hdr = re.compile(r"^##\s+\d+\.\s+(make_ptx)\s*/\s*(make_cubin)")
    pass_line = re.compile(r"^\s*\d+\.\s+`(add_\w+)`\s*[—-]\s*(.+?)\s*$")

    def flush():
        if cur_stage is not None:
            stages.append((cur_stage, cur_passes))

    for line in text.splitlines():
        cm = combo_hdr.match(line)
        if cm:
            flush()
            cur_stage, cur_passes = cm.group(1), []
            stages.append((cur_stage, []))
            cur_stage, cur_passes = cm.group(2), []
            continue
        sm = stage_hdr.match(line)
        if sm:
            flush()
            cur_stage = sm.group(1)
            cur_passes = []
            continue
        if line.startswith("## ") and cur_stage:
            flush()
            cur_stage, cur_passes = None, None
            continue
        pm = pass_line.match(line)
        if pm and cur_passes is not None:
            name = pm.group(1)
            tail = pm.group(2).rstrip(".")
            parts = [p.strip() for p in tail.split(";")]
            purpose = parts[0] if parts else tail
            sm_cond = parts[1] if len(parts) > 1 else ""
            chapter = parts[2] if len(parts) > 2 else ""
            cur_passes.append({
                "name": name,
                "purpose": purpose,
                "sm": sm_cond,
                "chapter": chapter,
            })
    flush()
    # de-dup combined make_ptx/make_cubin empties handled below
    merged = {}
    order = []
    for nm, ps in stages:
        if nm not in merged:
            merged[nm] = ps
            order.append(nm)
        else:
            merged[nm].extend(ps)
    return [(nm, merged[nm]) for nm in order]


# ── parse algorithm skeletons: functions with file:line ────────────────────
FILE_HDR = re.compile(r"^##\s+File:\s+(\S+)")


def _table_blocks(text):
    """Yield (current_file, header_cells, [row_cells]) for each markdown table."""
    cur_file = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        fh = FILE_HDR.match(lines[i])
        if fh:
            cur_file = fh.group(1)
            i += 1
            continue
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|", lines[i + 1]):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            yield cur_file, header, rows
            continue
        i += 1


def parse_functions(ppd):
    funcs = []
    seen = set()
    for fname in ("03-algorithm-skeleton-core.md",
                  "04-algorithm-skeleton-passes.md",
                  "05-algorithm-skeleton-lowering.md"):
        path = os.path.join(ppd, "docs", "analysis", fname)
        if not os.path.isfile(path):
            continue
        text = _read(path)
        for cur_file, header, rows in _table_blocks(text):
            hl = [h.lower() for h in header]
            if "symbol" not in hl or "lines" not in hl:
                continue
            i_sym = hl.index("symbol")
            i_lines = hl.index("lines")
            i_sig = hl.index("signature") if "signature" in hl else None
            i_purpose = len(header) - 1  # purpose is last column
            for cells in rows:
                if len(cells) <= i_lines:
                    continue
                sym = cells[i_sym]
                if not sym or sym.lower() == "symbol":
                    continue
                purpose = cells[i_purpose] if len(cells) > i_purpose else ""
                if "OUT OF SLICE" in purpose.upper():
                    continue
                core = "⭐" in sym
                name = sym.replace("⭐", "").replace("**", "").replace("`", "").strip()
                if not name:
                    continue
                # first integer in the Lines cell
                lm = re.search(r"(\d+)", cells[i_lines])
                line_no = int(lm.group(1)) if lm else None
                sig = ""
                if i_sig is not None and len(cells) > i_sig:
                    sig = cells[i_sig].replace("`", "").strip()
                key = (name, cur_file)
                if key in seen:
                    continue
                seen.add(key)
                funcs.append({
                    "name": name,
                    "file": cur_file,
                    "snapshot_line": line_no,
                    "signature": sig,
                    "core": core,
                    "purpose": purpose.replace("**", "").strip(),
                })
    return funcs


# ── parse backend make_* method line ranges (from orchestrator) ────────────
def parse_make_ranges(ppd):
    text = _read(os.path.join(ppd, "docs", "analysis", "06-orchestrator-skeleton.md"))
    ranges = {}
    for m in re.finditer(r"`(make_\w+)`\s*\(lines\s+(\d+)[–-](\d+)\)", text):
        ranges[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return ranges


def binding_for(pass_name):
    sub = PASS_SUBMODULE.get(pass_name)
    if sub:
        return f"passes.{sub}.{pass_name}"
    return None


def main():
    ap = argparse.ArgumentParser(description="Generate Triton dissection pipeline index")
    ap.add_argument("--ppd", default=default_ppd(),
                    help="Path to triton-ppd repo (or set $TRITON_PPD_DIR)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ppd = args.ppd
    if not ppd or not os.path.isdir(ppd):
        print("ERROR: triton-ppd not found. Pass --ppd <path> or set "
              "$TRITON_PPD_DIR.", file=sys.stderr)
        sys.exit(1)

    golden = parse_golden(ppd)
    funcs = parse_functions(ppd)
    make_ranges = parse_make_ranges(ppd)

    BACKEND_SRC = "third_party/nvidia/backend/compiler.py"

    # index functions by source-file basename for pass joins
    by_basename = {}
    for fn in funcs:
        bn = os.path.basename(fn["file"]) if fn.get("file") else ""
        by_basename.setdefault(bn, []).append(fn)

    stages = []
    passes = []
    for stage_name, plist in golden:
        rng = make_ranges.get(stage_name)
        method = "CUDABackend." + stage_name
        stages.append({
            "name": stage_name,
            "backend_method": method,
            "source": BACKEND_SRC,
            "snapshot_lines": list(rng) if rng else None,
            "pass_count": len(plist),
            "passes": [p["name"] for p in plist],
        })
        # count duplicate runs within the stage (e.g. remove_layout_conversions x3)
        run_counts = {}
        for p in plist:
            run_counts[p["name"]] = run_counts.get(p["name"], 0) + 1
        emitted = set()
        for p in plist:
            if p["name"] in emitted:
                continue
            emitted.add(p["name"])
            cpp = PASS_CPP.get(p["name"])
            keyfns = []
            if cpp:
                keyfns = [f["name"] for f in by_basename.get(os.path.basename(cpp), [])]
            passes.append({
                "name": p["name"],
                "stage": stage_name,
                "python_binding": binding_for(p["name"]),
                "purpose": p["purpose"],
                "sm": p["sm"],
                "chapter": p["chapter"],
                "runs": run_counts[p["name"]],
                "cpp_file": cpp,
                "key_functions": keyfns,
            })

    index = {
        "generated_from": os.path.abspath(ppd),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Self-contained Triton lowering-pipeline dissection index, "
                "distilled from triton-ppd. Regenerate with generate_pipeline.py.",
        "stages": stages,
        "passes": passes,
        "functions": funcs,
        "concepts": CONCEPTS,
    }

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "pipeline.json")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out}")
    print(f"  stages:    {len(stages)}  ({', '.join(s['name'] for s in stages)})")
    print(f"  passes:    {len(passes)} unique")
    print(f"  functions: {len(funcs)}  ({sum(1 for f in funcs if f['core'])} core)")
    print(f"  concepts:  {len(CONCEPTS)}  ({', '.join(CONCEPTS)})")


if __name__ == "__main__":
    main()
