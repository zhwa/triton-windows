---
name: triton-internals
description: "Dissect and navigate the Triton compiler lowering pipeline. Surgical 'scissor' that locates exact hook points — symbol, file, Python pass binding, signature — at every stage/pass/function so you can set breakpoints, bind to Python tests, redirect API calls, manipulate arguments, or emit ready-to-run hook snippets for perf experiments. Also a reference locator for components, ops, and encodings. Cross-platform and version-independent (symbol-first; line numbers resolved live from the source tree). Distilled from the triton-ppd study project; works WITHOUT triton-ppd present. Use for: tracing where convert_layout / coalesce / pipeline / mma is produced/eliminated/lowered, finding the function to break on inside a pass, listing lowering stages and passes, generating monkey-patch/breakpoint snippets, or looking up an op/encoding/component."
argument-hint: "stages | stage <name> | pass <name> | func <name> | trace <concept> | snippet <kind> <name> | hooks | (locator) show/op/encoding"
user-invocable: true
---

# Triton Internals — Pipeline Dissector & Component Locator

Two tools over Triton's compiler internals, distilled from the `triton-ppd`
study project into self-contained JSON (works **without** triton-ppd present):

1. **`dissector.py`** — the surgical "scissor". Per **stage → pass → function**,
   it gives the exact hook points (symbol, `file:line`, Python binding,
   signature) for breakpoints, Python-test binding, API redirection, and arg
   manipulation — and emits ready-to-run hook snippets.
2. **`locator.py`** — reference lookup for components, ops, and encodings.

> **Cross-platform & version-independent.** Pure Python, no OS-specific APIs;
> runs on Linux/macOS/Windows. Function/class **symbols** are the stable key —
> line numbers are resolved *live* from your actual source tree (so they never
> go stale). Paths below use `/`; that works on every OS including Windows.

---

## Part 1 — The Dissector (primary)

### Why

To experiment with the lowering pipeline you need to know, for any stage/pass/
concept: *where is the function (to break on), what's its signature (to change
args), and what's the Python handle (to redirect or skip it)?* The dissector
answers all three from one index — and can hand you the code.

### Commands

```bash
# The 5 lowering stages + pass counts (the pipeline overview)
python .github/skills/triton-internals/scripts/dissector.py stages

# One stage's ordered passes, each with its Python binding + SM condition
python .github/skills/triton-internals/scripts/dissector.py stage make_ttgir

# Full hook sheet for a pass: binding, C++ file, run-count, key functions
python .github/skills/triton-internals/scripts/dissector.py pass remove_layout_conversions

# One function: location (breakpoint), signature (args), owning pass
python .github/skills/triton-internals/scripts/dissector.py func emitIndices

# Trace a concept's lifecycle: who produces / eliminates / lowers it
python .github/skills/triton-internals/scripts/dissector.py trace convert_layout

# Generate ready-to-run hook code
python .github/skills/triton-internals/scripts/dissector.py snippet stage make_ttgir --action wrap
python .github/skills/triton-internals/scripts/dissector.py snippet pass coalesce --action run

# Every prime core intervention point across the pipeline
python .github/skills/triton-internals/scripts/dissector.py hooks

# Search passes + functions + concepts
python .github/skills/triton-internals/scripts/dissector.py search swizzle
```

### What each command gives you

| Command | Hook value |
|---------|-----------|
| `stages` | The 5 backend methods (`make_ttir`…`make_cubin`) you can wrap/override |
| `stage <name>` | Ordered passes + `passes.<area>.add_*` bindings to insert/remove/reorder |
| `pass <name>` | Python binding (redirect), C++ file (breakpoint), run-count, every key function with `file:line` |
| `func <name>` | `file:line` (breakpoint), signature (args to manipulate), purpose, owning pass |
| `trace <concept>` | Where a concept is produced, eliminated, and lowered — with functions |
| `snippet <kind> <name>` | Copy-paste Python monkey-patch / PassManager / debugger code |
| `hooks` | All core functions — the highest-leverage intervention points |

### Concept traces

`trace <concept>` covers the five thesis-critical concepts:

| Concept | Lifecycle traced |
|---------|------------------|
| `convert_layout` | bridge/matmul insert → 3-run RLC garbage-collects → lowered via LinearLayout |
| `coalesce` | generic Blocked → CoalescePass reorders → vector-width lowering |
| `pipeline` | latency assign/schedule → async pipeline materialized → async-copy lowering |
| `shared_memory` | matmul/pipeline alloc → offset assignment → SharedMemoryObject lowering |
| `mma` | accelerate-matmul installs MMA → lower_mma → mma.sync PTX |

### Example — `trace convert_layout`

```
convert_layout   ttg.convert_layout
  Produced by:
    add_convert_to_ttgpuir    (bridge assigns Blocked encoding)
    add_accelerate_matmul     (MMA encodings -> converts around dot)
    LayoutPropagation::getValueAs   RemoveLayoutConversions.cpp:545   (inserts a convert)
  Eliminated by:
    add_remove_layout_conversions  (runs 3x)   3-run fixed-point GC
    LayoutRematerialization::rewriteSlice          RemoveLayoutConversions.cpp:1362
  Lowered by (to LLVM/PTX):
    applyLinearLayout   Utility.cpp:410   emitIndices   Utility.cpp:679
```

(Line numbers shown are resolved from *your* checkout — see Version Independence.)

### The 5 Stages

| Stage | Backend method | Input → Output | Passes |
|-------|----------------|----------------|--------|
| `make_ttir` | `CUDABackend.make_ttir` | Python TTIR → cleaned TTIR | 8 |
| `make_ttgir` | `CUDABackend.make_ttgir` | TTIR → layout-annotated TTGIR | ~33 (SM-dependent) |
| `make_llir` | `CUDABackend.make_llir` | TTGIR → LLVM IR | ~11 |
| `make_ptx` | `CUDABackend.make_ptx` | LLVM IR → PTX text | translate (no passes) |
| `make_cubin` | `CUDABackend.make_cubin` | PTX → cubin | `ptxas` subprocess |

### Hook recipes (and the snippets that generate them)

**Set a breakpoint inside a pass** — `pass <name>` gives the C++ file + key
function. `snippet pass <name> --action break` emits the gdb/lldb commands
(symbol-based, so version independent).

**Run / isolate a pass in a Python test** — `snippet pass <name> --action run`:
```python
from triton._C.libtriton import ir, passes
pm = ir.pass_manager(mod.context)
passes.ttgpuir.add_remove_layout_conversions(pm)   # from the hook sheet
pm.run(mod); print(mod)
```

**Redirect / log / time a stage** — `snippet stage <name> --action wrap` (or
`time`) emits a monkey-patch of the backend `make_*` method:
```python
from triton.backends.nvidia.compiler import CUDABackend
_orig = CUDABackend.make_ttgir
def _wrap(mod, *a, **k):
    print(">>> before make_ttgir\n", mod)
    out = _orig(mod, *a, **k); print(">>> after\n", out); return out
CUDABackend.make_ttgir = staticmethod(_wrap)
```
Use `--backend vulkan` (or amd) to target a different backend class.

**Manipulate arguments for perf play** — `func <name>` gives the signature so
you know which args (a `LinearLayout`, tile shape, `kWidth`…) to tweak when you
intercept the call.

---

## Part 2 — The Locator (reference)

```bash
python .github/skills/triton-internals/scripts/locator.py show LinearLayout
python .github/skills/triton-internals/scripts/locator.py op tt.dot
python .github/skills/triton-internals/scripts/locator.py encoding BlockedEncodingAttr
python .github/skills/triton-internals/scripts/locator.py list
python .github/skills/triton-internals/scripts/locator.py dag
```

`show <component>` gives upstream source anchors, motivation, and the matching
triton-ppd spec sheet / reference chapter / textbook page.

---

## Cross-Platform & Version Independence

**Cross-platform.** All scripts are pure Python (`os.path`, `argparse`, no
`win32`/PowerShell/OS APIs). UTF-8 output is enabled portably. ANSI color
auto-disables when not a TTY or when `NO_COLOR` is set. Invoke with `python`
or `python3` on any OS; forward-slash paths work everywhere.

**Version independence.** The index stores **symbols** (function/class names),
not fixed line numbers. The dissector resolves the *current* line on demand:

1. `--source <triton-root>` (highest priority)
2. `$TRITON_SRC`
3. Auto-detect: walk up from the CWD and the script location for a
   Triton-shaped tree (`include/triton/` or `lib/Dialect/TritonGPU/`).

If resolved, you see the **real** line for your checkout (e.g. `Utility.cpp:711`).
If no source is found, the advisory snapshot line is shown as `file:~NNN
(snapshot)`. Breakpoint snippets are always symbol-based, so they work
regardless of version. Pass→submodule bindings are stable names verified
against `python/src/passes.cc`.

```bash
# Point at any triton checkout to get exact line numbers for that version:
python .../dissector.py --source /path/to/triton func emitIndices
```

---

## Data & Regeneration

| File | Produced by | Contents |
|------|-------------|----------|
| `data/pipeline.json` | `generate_pipeline.py` | stages, passes, functions, concept traces |
| `data/index.json` | `generate_index.py` | components, ops, encodings, chapter DAG |

Both are committed; the CLIs read only these files — **no triton-ppd needed at
runtime**. Regenerate only when triton-ppd updates (cross-platform — set
`$TRITON_PPD_DIR` or pass `--ppd`):

```bash
export TRITON_PPD_DIR=/path/to/triton-ppd      # or pass --ppd each time
python .github/skills/triton-internals/scripts/generate_pipeline.py
python .github/skills/triton-internals/scripts/generate_index.py
```

### What's indexed

| Axis | Count | Examples |
|------|-------|----------|
| Lowering stages | 5 | make_ttir … make_cubin |
| Passes | 45 | add_coalesce, add_remove_layout_conversions, add_accelerate_matmul |
| Functions (hooks) | 179 (15 core) | emitIndices, applyLinearLayout, LayoutPropagation::propagateLayout |
| Concept traces | 5 | convert_layout, coalesce, pipeline, shared_memory, mma |
| Components | 15 | LinearLayout, CoalescePass, ConvertTritonGPUToLLVM |
| Ops | ~75 | tt.dot, tt.load, ttg.convert_layout |
| Encodings | 15 | BlockedEncodingAttr, NvidiaMmaEncodingAttr |

## How This Fits With Other Skills

| Skill | Role |
|-------|------|
| `triton-internals` (this) | **Understand & dissect** Triton internals — where to hook |
| `triton-director` | **Inspect** our own Vulkan pipeline — capture IR, time passes |
| `triton-windows-vulkan` / `-perf` | **Build / optimize** the Vulkan backend |

Typical flow: `dissect trace <concept>` or `dissect pass <name>` → `dissect
snippet …` for the code → set a breakpoint or bind the pass in a Python test.

## Data Source Provenance

| triton-ppd source | Feeds |
|-------------------|-------|
| `docs/spec/golden-pass-list.md` | stages + ordered passes + 3-run pattern |
| `docs/analysis/06-orchestrator-skeleton.md` | backend `make_*` method ranges (advisory) |
| `docs/analysis/03/04/05-algorithm-skeleton-*.md` | per-pass key functions (symbol + advisory line) |
| `docs/spec/*.md`, `docs/analysis/01-dialect-inventory.md` | components, ops, encodings (locator) |

Curated maps (in `generate_pipeline.py`): pass → Python binding (verified
against `python/src/passes.cc`), pass → C++ file, and concept-trace lifecycles.

## Adapting to Upstream Changes

- **New pass / stage**: rerun `generate_pipeline.py`; the golden-pass-list drives it.
- **New binding**: add an entry to `PASS_SUBMODULE` in `generate_pipeline.py`
  (verify the submodule in `python/src/passes.cc`).
- **New concept to trace**: add to the `CONCEPTS` dict in `generate_pipeline.py`
  (use symbols + file paths, no line numbers).
- **Line numbers drifted**: nothing to do — the dissector resolves live from
  `--source`. Symbols are stable; snapshot lines are advisory fallback only.
- **triton-ppd absent**: both CLIs still work from committed JSON; only the
  generators need it, via `--ppd <path>` or `$TRITON_PPD_DIR`.
