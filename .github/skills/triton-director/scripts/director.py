"""triton-director: Beginner-friendly CLI for navigating the triton-windows project.

Usage:
    python .github\\skills\\triton-director\\scripts\\director.py help
    python .github\\skills\\triton-director\\scripts\\director.py scan
    python .github\\skills\\triton-director\\scripts\\director.py init
    python .github\\skills\\triton-director\\scripts\\director.py inspect <kernel.ttir> [--stats] [--diff]
    python .github\\skills\\triton-director\\scripts\\director.py time    <kernel.ttir> [--runs N]
    python .github\\skills\\triton-director\\scripts\\director.py env     [--check]
    python .github\\skills\\triton-director\\scripts\\director.py test

Single entry point for all triton-windows development activities.
"""
import argparse, glob, os, re, subprocess, sys, textwrap

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, "..", "..", "..", ".."))


# ── Colors ────────────────────────────────────────────────────────────────
def _c(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


# ── help ──────────────────────────────────────────────────────────────────
def cmd_help(args):
    """Show all available commands with examples."""
    print(bold("\n  Triton Director — Development Toolkit for triton-windows\n"))
    print("  Commands:\n")
    cmds = [
        ("help",    "Show this help message"),
        ("scan",    "Scan the repo — list skills, tools, docs, and test files"),
        ("skills",  "List every skill and what it can do (commands per skill)"),
        ("init",    "Show the step-by-step journey to build & run from scratch"),
        ("inspect", "Capture IR at every pipeline stage (wraps inspector.py)"),
        ("time",    "Per-pass compilation timing (wraps timer.py)"),
        ("env",     "Show / check environment (conda, MSVC, paths)"),
        ("test",    "Run the Vulkan test suite"),
    ]
    for name, desc in cmds:
        print(f"    {cyan(name):<22s} {desc}")

    print(bold("\n  Quick start:\n"))
    print(f"    {dim('$')} python .github\\skills\\triton-director\\scripts\\director.py {cyan('scan')}")
    print(f"    {dim('$')} python .github\\skills\\triton-director\\scripts\\director.py {cyan('init')}")
    print(f"    {dim('$')} python .github\\skills\\triton-director\\scripts\\director.py {cyan('inspect')} test.ttir --stats")
    print(f"    {dim('$')} python .github\\skills\\triton-director\\scripts\\director.py {cyan('time')} test.ttir --runs 3")
    print()


# ── scan ──────────────────────────────────────────────────────────────────
def cmd_scan(args):
    """Scan the repo for skills, tools, docs, and tests."""
    print(bold("\n  ── Skills ──\n"))
    skills_dir = os.path.join(_repo_root, ".github", "skills")
    if os.path.isdir(skills_dir):
        for entry in sorted(os.listdir(skills_dir)):
            skill_file = os.path.join(skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_file):
                desc = _extract_skill_description(skill_file)
                has_scripts = os.path.isdir(os.path.join(skills_dir, entry, "scripts"))
                badge = f" {cyan('[has scripts]')}" if has_scripts else ""
                print(f"    {green(entry):<45s}{badge}")
                if desc:
                    # Wrap long descriptions
                    for line in textwrap.wrap(desc, 72):
                        print(f"      {dim(line)}")
                print()
    else:
        print(f"    {red('No .github/skills/ directory found')}")

    print(bold("  ── Development Docs ──\n"))
    dev_dir = os.path.join(_repo_root, "development")
    if os.path.isdir(dev_dir):
        for f in sorted(os.listdir(dev_dir)):
            if f.endswith(".md"):
                path = os.path.join(dev_dir, f)
                lines = _count_lines(path)
                title = _extract_title(path)
                print(f"    {f:<40s} {dim(f'{lines} lines'):<14s} {title}")
    else:
        print(f"    {dim('No development/ directory')}")

    print(bold("\n  ── Python Tool Scripts ──\n"))
    for pattern in ["**/*.py", "**/*.ps1"]:
        for p in sorted(glob.glob(os.path.join(skills_dir, pattern), recursive=True)):
            rel = os.path.relpath(p, _repo_root)
            print(f"    {rel}")

    print(bold("\n  ── Vulkan Test Kernels (.ttir) ──\n"))
    test_dir = os.path.join(_repo_root, "third_party", "vulkan", "test")
    if os.path.isdir(test_dir):
        ttir_files = sorted(glob.glob(os.path.join(test_dir, "*.ttir")))
        for f in ttir_files:
            print(f"    {os.path.basename(f)}")
        print(f"\n    {dim(f'{len(ttir_files)} kernel(s) found')}")
    else:
        print(f"    {dim('No test directory found')}")

    print(bold("\n  ── Upstream Triton Tools ──\n"))
    tools_dir = os.path.join(_repo_root, "python", "triton", "tools")
    if os.path.isdir(tools_dir):
        for f in sorted(os.listdir(tools_dir)):
            if f.endswith(".py") and not f.startswith("_"):
                print(f"    python/triton/tools/{f}")
    print()


def _extract_skill_description(skill_file):
    """Extract the description from YAML frontmatter."""
    try:
        with open(skill_file, "r", encoding="utf-8") as f:
            text = f.read(2000)
        m = re.search(r'description:\s*"([^"]+)"', text)
        if m:
            desc = m.group(1)
            # Truncate to first sentence
            first = desc.split(". ")[0] + "."
            return first if len(first) < 120 else first[:117] + "..."
        return None
    except Exception:
        return None


def _extract_skill_meta(skill_file):
    """Extract full description + argument-hint (the skill's capabilities)."""
    meta = {"description": None, "argument_hint": None}
    try:
        with open(skill_file, "r", encoding="utf-8") as f:
            text = f.read(4000)
    except Exception:
        return meta
    md = re.search(r'description:\s*"([^"]+)"', text)
    if md:
        meta["description"] = md.group(1).strip()
    mh = re.search(r'argument-hint:\s*"([^"]+)"', text)
    if mh:
        meta["argument_hint"] = mh.group(1).strip()
    return meta


def _skill_entry_scripts(skill_dir):
    """Return likely entry-point scripts for a skill (excludes helpers/generators)."""
    scripts = os.path.join(skill_dir, "scripts")
    if not os.path.isdir(scripts):
        return []
    py = [f for f in os.listdir(scripts)
          if f.endswith(".py")
          and not f.startswith("_")
          and not f.startswith("generate_")
          and f != "__init__.py"]
    if not py:
        return []
    name = os.path.basename(skill_dir)
    # primary first: a script whose stem matches the skill's last name segment,
    # then alphabetical (puts e.g. dissector before locator)
    seg = name.split("-")[-1]
    py.sort(key=lambda f: (seg not in os.path.splitext(f)[0], f))
    return [os.path.join("scripts", f) for f in py]


def _extract_title(md_file):
    """Extract the first # heading from a markdown file."""
    try:
        with open(md_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return dim(line[2:].strip()[:60])
        return ""
    except Exception:
        return ""


def _count_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


# ── skills ─────────────────────────────────────────────────────────────────
def cmd_skills(args):
    """List every skill and what it can do (description + commands)."""
    skills_dir = os.path.join(_repo_root, ".github", "skills")
    if not os.path.isdir(skills_dir):
        print(red("  No .github/skills/ directory found"))
        return

    entries = sorted(d for d in os.listdir(skills_dir)
                     if os.path.isfile(os.path.join(skills_dir, d, "SKILL.md")))

    # If a skill name is given, show just that one (full detail)
    target = getattr(args, "name", None)
    if target:
        match = next((e for e in entries if target.lower() in e.lower()), None)
        if not match:
            print(yellow(f"  No skill matching '{target}'. Known: " + ", ".join(entries)))
            return
        entries = [match]

    print(bold("\n  Triton-Windows Skills — What Each One Can Do\n"))
    for entry in entries:
        skill_dir = os.path.join(skills_dir, entry)
        meta = _extract_skill_meta(os.path.join(skill_dir, "SKILL.md"))
        scripts = _skill_entry_scripts(skill_dir)

        print(f"  {bold(green(entry))}")
        if meta["description"]:
            # show the actionable part: keep through the first 2 sentences
            desc = meta["description"]
            sents = desc.split(". ")
            blurb = ". ".join(sents[:2]).rstrip(".") + "."
            for line in textwrap.wrap(blurb, 74):
                print(f"      {dim(line)}")
        if meta["argument_hint"]:
            print(f"      {cyan('commands:')} {meta['argument_hint']}")
        for script in scripts[:2]:
            rel = os.path.join(".github", "skills", entry, script).replace("\\", "/")
            print(f"      {dim('run:')} python {rel} <command>")
        print()

    if not target:
        print(dim("  Tip: 'skills <name>' for one skill; 'scan' for docs/tools/tests too.\n"))


# ── init ──────────────────────────────────────────────────────────────────
def cmd_init(args):
    """Show the step-by-step journey from fresh clone to running tests."""
    print(bold("\n  ── From Fresh Clone to 12/12 PASS ──\n"))

    steps = [
        ("1", "Set up environment",
         "conda create -n triton-dev python=3.12 ninja cmake pip\n"
         "conda activate triton-dev\n"
         "pip install pybind11 setuptools wheel"),

        ("2", "Clone LLVM + apply patches",
         "Skill: triton-windows-build  →  Task: clone-llvm\n"
         "Clones LLVM at the commit from cmake/llvm-hash.txt, downloads JSON headers,\n"
         "and applies 3 LLVM MSVC patches (L1-L3)."),

        ("3", "Inspect & patch triton source",
         "python .github\\skills\\triton-windows-build\\scripts\\inspect-build.py --fix\n"
         "Detects 20+ Windows build issues; apply patches T1-T20 from build skill."),

        ("4", "Build LLVM (Release)",
         "Skill: triton-windows-build  →  Task: build-llvm\n"
         "Takes ~30 min. MUST be Release (not Debug — CRT mismatch)."),

        ("5", "Build triton",
         "Skill: triton-windows-build  →  Task: build-triton\n"
         "pip install --no-build-isolation -e .\n"
         "Then: remove AMD backend symlinks (they cause ImportError)."),

        ("6", "Run Vulkan tests",
         "$env:TRITON_BACKENDS_IN_TREE = '1'\n"
         "python third_party\\vulkan\\test\\test_kernels_vulkan.py\n"
         "Expected: 12/12 PASS with compile + dispatch timing."),

        ("7", "Inspect the pipeline",
         "python .github\\skills\\triton-director\\scripts\\director.py inspect test.ttir --stats\n"
         "python .github\\skills\\triton-director\\scripts\\director.py time test.ttir"),
    ]

    for num, title, detail in steps:
        print(f"  {cyan(f'Step {num}')}: {bold(title)}")
        for line in detail.split("\n"):
            print(f"    {line}")
        print()

    print(bold("  ── Vulkan Backend Development Journey ──\n"))
    print("  After building, follow these skills in order:\n")
    phases = [
        ("Phase 1", "triton-windows-vulkan",
         "Foundation: 16 converters, 7 bridge passes, VulkanizePass, runtime"),
        ("Phase 2", "triton-windows-vulkan-perf",
         "C+1→C+6: WorkgroupId, device-local, shared mem, subgroups, coop matrix, discrete GPU"),
        ("Debug",   "triton-director",
         "Pipeline inspection, timing, IR capture, debugging playbooks"),
        ("Optional","triton-windows-opencl",
         "OpenCL C debug output (not required for SPIR-V pipeline)"),
    ]
    for phase, skill, desc in phases:
        print(f"    {cyan(phase):<12s}  Skill: {green(skill)}")
        print(f"    {'':12s}  {dim(desc)}")
        print()

    print(f"  {bold('Reference docs:')}  development/vulkan-backend-guide.md (~2500 lines)")
    print(f"  {'':14s}  development/intel-xpu-backend-study.md (Path A vs C+ analysis)")
    print()


# ── env ───────────────────────────────────────────────────────────────────
def cmd_env(args):
    """Show or check the development environment."""
    print(bold("\n  ── Environment ──\n"))

    checks = []

    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python", py_ver, sys.executable, True))

    # Conda env
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", os.environ.get("CONDA_PREFIX", ""))
    checks.append(("Conda env", conda_env or red("not active"), "", bool(conda_env)))

    # MSVC / cl.exe
    import shutil
    cl = shutil.which("cl")
    checks.append(("cl.exe (MSVC)", cl or red("not on PATH — run vcvars64.bat"), "", bool(cl)))

    # triton import
    try:
        import triton
        checks.append(("triton", triton.__version__, triton.__file__, True))
    except ImportError as e:
        checks.append(("triton", red(f"import failed: {e}"), "", False))

    # Vulkan backend
    try:
        from triton._C.libtriton import vulkan
        checks.append(("Vulkan backend", green("available"), "", True))
    except ImportError:
        checks.append(("Vulkan backend", red("not built"), "", False))

    # LLVM build
    llvm_build = os.path.join(_repo_root, "build", "llvm-project", "build")
    checks.append(("LLVM build", llvm_build if os.path.isdir(llvm_build) else red("not found"), "", os.path.isdir(llvm_build)))

    # TRITON_BACKENDS_IN_TREE
    bit = os.environ.get("TRITON_BACKENDS_IN_TREE", "")
    checks.append(("TRITON_BACKENDS_IN_TREE", bit or yellow("not set"), "", bit == "1"))

    for name, value, detail, ok in checks:
        status = green("✓") if ok else red("✗")
        print(f"    {status} {name:<28s} {value}")
        if detail and args.check:
            print(f"      {dim(detail)}")

    all_ok = all(ok for _, _, _, ok in checks)
    print()
    if all_ok:
        print(f"    {green('All checks passed — ready to develop!')}")
    else:
        print(f"    {yellow('Some checks failed — see triton-windows-build skill for setup.')}")
    print()


# ── inspect (delegates to inspector.py) ───────────────────────────────────
def cmd_inspect(args):
    """Run the pipeline inspector on a .ttir file."""
    script = os.path.join(_script_dir, "inspector.py")
    cmd = [sys.executable, script, args.ttir]
    if args.stats:  cmd.append("--stats")
    if args.diff:   cmd.append("--diff")
    if args.passes: cmd.append("--passes")
    if args.out:    cmd.extend(["--out", args.out])
    if args.json:   cmd.append("--json")
    sys.exit(subprocess.call(cmd))


# ── time (delegates to timer.py) ──────────────────────────────────────────
def cmd_time(args):
    """Run the pipeline timer on .ttir file(s)."""
    script = os.path.join(_script_dir, "timer.py")
    cmd = [sys.executable, script] + args.ttir
    if args.format != "table": cmd.extend(["--format", args.format])
    if args.runs > 1:          cmd.extend(["--runs", str(args.runs)])
    sys.exit(subprocess.call(cmd))


# ── test ──────────────────────────────────────────────────────────────────
def cmd_test(args):
    """Run the Vulkan test suite."""
    test_file = os.path.join(_repo_root, "third_party", "vulkan", "test", "test_kernels_vulkan.py")
    if not os.path.isfile(test_file):
        print(red(f"  Test file not found: {test_file}"))
        sys.exit(1)
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    sys.exit(subprocess.call([sys.executable, test_file]))


# ── main ──────────────────────────────────────────────────────────────────
def main():
    # Portable UTF-8 stdout (works on every platform; no win32-specific check).
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        prog="triton-director",
        description="Beginner-friendly CLI for triton-windows development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("help", help="Show all commands with examples")
    sub.add_parser("scan", help="Scan repo for skills, tools, docs, and tests")
    sub.add_parser("init", help="Show step-by-step journey from fresh clone")

    p_env = sub.add_parser("env", help="Show/check development environment")
    p_env.add_argument("--check", action="store_true", help="Show detailed paths")

    p_inspect = sub.add_parser("inspect", help="Capture IR at each pipeline stage")
    p_inspect.add_argument("ttir", help="Path to .ttir input file")
    p_inspect.add_argument("--stats", action="store_true", help="Show op statistics")
    p_inspect.add_argument("--diff", action="store_true", help="Show stage diffs")
    p_inspect.add_argument("--passes", action="store_true", help="Per-pass capture")
    p_inspect.add_argument("--out", metavar="DIR", help="Write IR files to dir")
    p_inspect.add_argument("--json", action="store_true", help="JSON output")

    p_time = sub.add_parser("time", help="Per-pass compilation timing")
    p_time.add_argument("ttir", nargs="+", help="Path(s) to .ttir file(s)")
    p_time.add_argument("--format", choices=["table", "csv", "json"], default="table")
    p_time.add_argument("--runs", type=int, default=1, help="Runs to average")

    sub.add_parser("test", help="Run the Vulkan GPU test suite")

    p_skills = sub.add_parser("skills", help="List every skill and what it can do")
    p_skills.add_argument("name", nargs="?", help="Show one skill in detail")

    args = parser.parse_args()

    dispatch = {
        "help": cmd_help,
        "scan": cmd_scan,
        "init": cmd_init,
        "env": cmd_env,
        "inspect": cmd_inspect,
        "time": cmd_time,
        "test": cmd_test,
        "skills": cmd_skills,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        cmd_help(args)


if __name__ == "__main__":
    main()
