"""OpenCL C emitter for lowered MLIR memref/arith/cf IR.

Takes MLIR IR that has been lowered through:
  triton-to-linalg → bufferize → linalg-to-loops → lower-affine → scf-to-cf
and produces OpenCL C source code.

The emitter walks the fully-lowered IR line by line. Each MLIR operation maps
to a small C statement. SSA values are tracked in a dict mapping %name → C var.
Element types are tracked per variable to avoid hardcoding "float" everywhere.
"""

import re
from typing import Dict, List, Optional, Tuple


def emit_opencl(mlir_ir: str) -> str:
    """Convert fully-lowered MLIR IR to OpenCL C source code."""
    emitter = OpenCLEmitter()
    return emitter.emit(mlir_ir)


class OpenCLEmitter:
    """Walk lowered MLIR IR and produce OpenCL C."""

    TYPE_MAP = {
        "f16": "half", "f32": "float", "f64": "double",
        "i1": "bool", "i8": "char", "i16": "short", "i32": "int", "i64": "long",
        "index": "int",
    }

    def __init__(self):
        self.indent = 0
        self.lines: List[str] = []
        self.var_counter = 0
        self.ssa_map: Dict[str, str] = {}
        self.ssa_types: Dict[str, str] = {}
        self.block_map: Dict[str, str] = {}
        self._block_args: Dict[str, List[str]] = {}
        self._block_arg_decls: Dict[str, List[tuple]] = {}

    def _line(self, s: str):
        self.lines.append("  " * self.indent + s)

    def _fresh_var(self, prefix="v") -> str:
        self.var_counter += 1
        return f"{prefix}_{self.var_counter}"

    def _map_type(self, mlir_type: str) -> str:
        mlir_type = mlir_type.strip()
        if mlir_type.startswith("memref<*x"):
            elem = mlir_type[len("memref<*x"):-1]
            return f"__global {self.TYPE_MAP.get(elem, elem)}*"
        if mlir_type.startswith("memref<"):
            m = re.match(r"memref<[\d?x]+x(\w+)", mlir_type)
            if m:
                return f"{self.TYPE_MAP.get(m.group(1), m.group(1))}*"
        return self.TYPE_MAP.get(mlir_type, mlir_type)

    def _map_val(self, ssa_name: str) -> str:
        ssa_name = ssa_name.strip().rstrip(",")
        if ssa_name in self.ssa_map:
            return self.ssa_map[ssa_name]
        if ssa_name.startswith("%"):
            return ssa_name.replace("%", "v_").replace("-", "_")
        return ssa_name

    def _type_of(self, ssa_name: str) -> str:
        cvar = self._map_val(ssa_name)
        return self.ssa_types.get(cvar, "float")

    def _def_var(self, ssa: str, prefix: str, ctype: str) -> str:
        cvar = self._fresh_var(prefix)
        self.ssa_map[ssa] = cvar
        self.ssa_types[cvar] = ctype
        return cvar

    # ── top-level ────────────────────────────────────────────────────────

    def emit(self, mlir_ir: str) -> str:
        ir = self._strip_locations(mlir_ir)
        lines = ir.strip().split("\n")
        func_lines = self._extract_function(lines)
        if not func_lines:
            return "// ERROR: no func.func found\n"

        sig_line = func_lines[0]
        body_lines = func_lines[1:]
        fname, args = self._parse_signature(sig_line)

        self.lines = []
        self._line("// Auto-generated OpenCL C from Triton IR")
        self._line(f"__kernel void {fname}(")

        arg_strs = []
        for i, (aname, atype) in enumerate(args):
            ctype = self._map_type(atype)
            cname = f"arg{i}"
            self.ssa_map[aname] = cname
            self.ssa_types[cname] = ctype
            arg_strs.append(f"    {ctype} {cname}")
        self._line(",\n".join(arg_strs))
        self._line(") {")
        self.indent = 1
        self._emit_body(body_lines)
        self.indent = 0
        self._line("}")
        return "\n".join(self.lines) + "\n"

    # ── IR cleaning ─────────────────────────────────────────────────────

    def _strip_locations(self, ir: str) -> str:
        result = []
        for line in ir.split("\n"):
            line = re.sub(r'\s+loc\(.*\)\s*$', '', line)
            line = re.sub(r'\s+loc\([^)]*\)', '', line)
            if re.match(r'\s*#loc\d*\s*=', line):
                continue
            result.append(line)
        return "\n".join(result)

    def _extract_function(self, lines: List[str]) -> Optional[List[str]]:
        in_func = False
        depth = 0
        result = []
        for line in lines:
            if "func.func" in line and not in_func:
                in_func = True
                depth = 0
            if in_func:
                result.append(line)
                depth += line.count("{") - line.count("}")
                if depth <= 0 and len(result) > 1:
                    break
        return result if result else None

    def _parse_signature(self, sig_line: str) -> Tuple[str, List[Tuple[str, str]]]:
        m = re.search(r'@(\w+)\(([^)]*)\)', sig_line)
        if not m:
            return "kernel", []
        fname = m.group(1)
        args = []
        if m.group(2).strip():
            depth = 0
            current = ""
            for ch in m.group(2):
                if ch in "<(":
                    depth += 1
                elif ch in ">)":
                    depth -= 1
                if ch == "," and depth == 0:
                    args.append(self._parse_arg(current.strip()))
                    current = ""
                else:
                    current += ch
            if current.strip():
                args.append(self._parse_arg(current.strip()))
        return fname, args

    def _parse_arg(self, arg: str) -> Tuple[str, str]:
        m = re.match(r"(%[\w]+):\s*(.+)", arg)
        if m:
            return m.group(1), m.group(2).strip()
        return arg, "i32"

    # ── body emission ───────────────────────────────────────────────────

    def _emit_body(self, lines: List[str]):
        # Pre-pass: collect block arguments
        for line in lines:
            m = re.match(r"\s*\^(\w+)\(([^)]+)\):", line.strip())
            if m:
                label, params = m.group(1), m.group(2)
                parts = self._split_typed_args(params)
                arg_names = []
                for ssa, ty in parts:
                    ctype = self._map_type(ty)
                    cvar = self._def_var(ssa, "ba", ctype)
                    arg_names.append((cvar, ctype))
                self._block_args[label] = [n for n, _ in arg_names]
                self._block_arg_decls[label] = arg_names

        for label, decls in self._block_arg_decls.items():
            for cvar, ctype in decls:
                self._line(f"{ctype} {cvar};")

        for line in lines:
            line = line.strip()
            if not line or line == "}" or line.startswith("//"):
                continue
            if line.startswith("^"):
                m = re.match(r"\^(\w+)", line)
                if m:
                    self._line(f"{m.group(1)}:;")
                continue
            self._emit_op(line)

    def _split_typed_args(self, s: str) -> List[Tuple[str, str]]:
        result = []
        depth = 0
        current = ""
        for ch in s:
            if ch in "<(":
                depth += 1
            elif ch in ">)":
                depth -= 1
            if ch == "," and depth == 0:
                p = self._parse_arg(current.strip())
                if p:
                    result.append(p)
                current = ""
            else:
                current += ch
        if current.strip():
            p = self._parse_arg(current.strip())
            if p:
                result.append(p)
        return result

    # ── op dispatch ─────────────────────────────────────────────────────

    def _emit_op(self, line: str):
        if "arith.constant" in line:
            self._emit_constant(line)
        elif re.search(r"arith\.(addi|subi|muli|remsi|remui|andi|ori|xori|shrsi|shrui|shli)\b", line):
            self._emit_binary_int(line)
        elif re.search(r"arith\.(addf|subf|mulf|divf|maximumf|minimumf)\b", line):
            self._emit_binary_float(line)
        elif "arith.cmpi" in line:
            self._emit_cmpi(line)
        elif "arith.cmpf" in line:
            self._emit_cmpf(line)
        elif "arith.index_cast" in line:
            self._emit_index_cast(line)
        elif re.search(r"arith\.ext[su]i\b", line):
            self._emit_ext(line)
        elif "arith.trunci" in line:
            self._emit_ext(line)
        elif re.search(r"arith\.[su]itofp\b", line):
            self._emit_ext(line)
        elif re.search(r"arith\.fpto[su]i\b", line):
            self._emit_ext(line)
        elif re.search(r"arith\.(extf|truncf)\b", line):
            self._emit_ext(line)
        elif "arith.negf" in line:
            self._emit_unary(line, "-")
        elif re.search(r"math\.(sqrt|exp|log|absf|ceil|floor|cos|sin|tanh)\b", line):
            m = re.search(r"math\.(\w+)", line)
            fn = {"absf": "fabs", "ceil": "ceil", "floor": "floor"}.get(m.group(1), m.group(1))
            self._emit_math_call(line, fn)
        elif "memref.reinterpret_cast" in line:
            self._emit_reinterpret_cast(line)
        elif "memref.cast" in line:
            self._emit_memref_cast(line)
        elif "memref.alloc" in line:
            self._emit_alloc(line)
        elif "memref.dealloc" in line:
            pass
        elif "memref.store" in line:
            self._emit_store(line)
        elif "memref.load" in line:
            self._emit_load(line)
        elif "memref.copy" in line:
            self._emit_copy(line)
        elif "cf.br " in line:
            self._emit_br(line)
        elif "cf.cond_br" in line:
            self._emit_cond_br(line)
        elif "return" in line:
            self._line("return;")
        elif "bufferization" in line or "linalg." in line:
            self._line(f"// UNLOWERED: {line}")

    # ── arith ops ───────────────────────────────────────────────────────

    def _emit_constant(self, line: str):
        m = re.match(r"(%[\w]+)\s*=\s*arith\.constant\s+(.+?)\s*:\s*(\S+)", line)
        if not m:
            return
        ssa, val, ty = m.group(1), m.group(2), m.group(3)
        ctype = self._map_type(ty)
        cvar = self._def_var(ssa, "c", ctype)
        val = val.replace("e+0", "e").replace("e+", "e")
        if val == "true":
            val = "1"
        elif val == "false":
            val = "0"
        self._line(f"{ctype} {cvar} = {val};")

    def _emit_binary_int(self, line: str):
        op_map = {
            "arith.addi": "+", "arith.subi": "-", "arith.muli": "*",
            "arith.remsi": "%", "arith.remui": "%",
            "arith.andi": "&", "arith.ori": "|", "arith.xori": "^",
            "arith.shrsi": ">>", "arith.shrui": ">>", "arith.shli": "<<",
        }
        for op, sym in op_map.items():
            if op in line:
                m = re.match(rf"(%[\w]+)\s*=\s*{re.escape(op)}\s+(%[\w]+),\s*(%[\w]+)", line)
                if m:
                    dst, lhs, rhs = m.group(1), m.group(2), m.group(3)
                    ctype = self._type_of(lhs)
                    if ctype.endswith("*"):
                        ctype = "int"
                    cvar = self._def_var(dst, "i", ctype)
                    self._line(f"{ctype} {cvar} = {self._map_val(lhs)} {sym} {self._map_val(rhs)};")
                return

    def _emit_binary_float(self, line: str):
        op_map = {"arith.addf": "+", "arith.subf": "-", "arith.mulf": "*", "arith.divf": "/"}
        func_map = {"arith.maximumf": "fmax", "arith.minimumf": "fmin"}
        for op, sym in op_map.items():
            if op in line:
                m = re.match(rf"(%[\w]+)\s*=\s*{re.escape(op)}\s+(%[\w]+),\s*(%[\w]+)", line)
                if m:
                    dst, lhs, rhs = m.group(1), m.group(2), m.group(3)
                    ctype = self._type_of(lhs)
                    cvar = self._def_var(dst, "f", ctype)
                    self._line(f"{ctype} {cvar} = {self._map_val(lhs)} {sym} {self._map_val(rhs)};")
                return
        for op, fn in func_map.items():
            if op in line:
                m = re.match(rf"(%[\w]+)\s*=\s*{re.escape(op)}\s+(%[\w]+),\s*(%[\w]+)", line)
                if m:
                    dst, lhs, rhs = m.group(1), m.group(2), m.group(3)
                    ctype = self._type_of(lhs)
                    cvar = self._def_var(dst, "f", ctype)
                    self._line(f"{ctype} {cvar} = {fn}({self._map_val(lhs)}, {self._map_val(rhs)});")
                return

    def _emit_cmpi(self, line: str):
        pred_map = {"slt": "<", "sle": "<=", "sgt": ">", "sge": ">=",
                     "eq": "==", "ne": "!=", "ult": "<", "ule": "<=",
                     "ugt": ">", "uge": ">="}
        m = re.match(r"(%[\w]+)\s*=\s*arith\.cmpi\s+(\w+),\s*(%[\w]+),\s*(%[\w]+)", line)
        if m:
            dst, pred, lhs, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
            cvar = self._def_var(dst, "cmp", "bool")
            self._line(f"bool {cvar} = {self._map_val(lhs)} {pred_map.get(pred, '==')} {self._map_val(rhs)};")

    def _emit_cmpf(self, line: str):
        pred_map = {"olt": "<", "ole": "<=", "ogt": ">", "oge": ">=",
                     "oeq": "==", "one": "!="}
        m = re.match(r"(%[\w]+)\s*=\s*arith\.cmpf\s+(\w+),\s*(%[\w]+),\s*(%[\w]+)", line)
        if m:
            dst, pred, lhs, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
            cvar = self._def_var(dst, "cmp", "bool")
            self._line(f"bool {cvar} = {self._map_val(lhs)} {pred_map.get(pred, '==')} {self._map_val(rhs)};")

    def _emit_index_cast(self, line: str):
        m = re.match(r"(%[\w]+)\s*=\s*arith\.index_cast\s+(%[\w]+)\s*:\s*(\S+)\s+to\s+(\S+)", line)
        if m:
            dst, src, _, dst_ty = m.group(1), m.group(2), m.group(3), m.group(4)
            ctype = self._map_type(dst_ty)
            cvar = self._def_var(dst, "cast", ctype)
            self._line(f"{ctype} {cvar} = ({ctype}){self._map_val(src)};")

    def _emit_ext(self, line: str):
        """Handle arith.extsi, arith.trunci, arith.sitofp, arith.fptosi, arith.extf, arith.truncf."""
        m = re.match(r"(%[\w]+)\s*=\s*arith\.\w+\s+(%[\w]+)\s*:\s*(\S+)\s+to\s+(\S+)", line)
        if m:
            dst, src, _, dst_ty = m.group(1), m.group(2), m.group(3), m.group(4)
            ctype = self._map_type(dst_ty)
            cvar = self._def_var(dst, "cv", ctype)
            self._line(f"{ctype} {cvar} = ({ctype}){self._map_val(src)};")

    def _emit_unary(self, line: str, op: str):
        m = re.match(r"(%[\w]+)\s*=\s*arith\.\w+\s+(%[\w]+)", line)
        if m:
            dst, src = m.group(1), m.group(2)
            ctype = self._type_of(src)
            cvar = self._def_var(dst, "u", ctype)
            self._line(f"{ctype} {cvar} = {op}{self._map_val(src)};")

    def _emit_math_call(self, line: str, func: str):
        m = re.match(r"(%[\w]+)\s*=\s*math\.\w+\s+(%[\w]+)", line)
        if m:
            dst, src = m.group(1), m.group(2)
            ctype = self._type_of(src)
            cvar = self._def_var(dst, "m", ctype)
            self._line(f"{ctype} {cvar} = {func}({self._map_val(src)});")

    # ── memref ops ──────────────────────────────────────────────────────

    def _emit_reinterpret_cast(self, line: str):
        m = re.match(
            r"(%[\w]+)\s*=\s*memref\.reinterpret_cast\s+(%[\w]+)\s+to\s+offset:\s*\[([^\]]*)\]",
            line)
        if m:
            dst, src, offset = m.group(1), m.group(2), m.group(3)
            elem = "float"
            type_m = re.search(r"memref<[\d?x]*x?(\w+)", line)
            if type_m:
                elem = self.TYPE_MAP.get(type_m.group(1), type_m.group(1))
            cvar = self._def_var(dst, "ptr", f"{elem}*")
            offset_c = self._map_val(offset) if offset.strip() not in ("0", "") else "0"
            self._line(f"{elem}* {cvar} = {self._map_val(src)} + {offset_c};")

    def _emit_memref_cast(self, line: str):
        m = re.match(r"(%[\w]+)\s*=\s*memref\.cast\s+(%[\w]+)", line)
        if m:
            dst, src = m.group(1), m.group(2)
            src_type = self._type_of(src)
            cvar = self._def_var(dst, "mc", src_type)
            self._line(f"// memref.cast — identity in OpenCL")
            self.ssa_map[dst] = self._map_val(src)  # alias, no new var
            self.ssa_types[self._map_val(src)] = src_type

    def _emit_alloc(self, line: str):
        m = re.match(r"(%[\w]+)\s*=\s*memref\.alloc\(\)\s*:\s*memref<(\d+)x(\w+)>", line)
        if m:
            dst, size, elem = m.group(1), m.group(2), m.group(3)
            ctype = self.TYPE_MAP.get(elem, elem)
            cvar = self._def_var(dst, "buf", f"{ctype}*")
            self._line(f"__private {ctype} {cvar}[{size}];")

    def _emit_store(self, line: str):
        m = re.match(r"memref\.store\s+(%[\w]+),\s*(%[\w]+)\[([^\]]+)\]", line)
        if m:
            val, buf, indices = m.group(1), m.group(2), m.group(3)
            idx_parts = [i.strip() for i in indices.split(",")]
            idx_c = "][".join(self._map_val(i) for i in idx_parts)
            self._line(f"{self._map_val(buf)}[{idx_c}] = {self._map_val(val)};")

    def _emit_load(self, line: str):
        m = re.match(r"(%[\w]+)\s*=\s*memref\.load\s+(%[\w]+)\[([^\]]+)\]", line)
        if m:
            dst, buf, indices = m.group(1), m.group(2), m.group(3)
            idx_parts = [i.strip() for i in indices.split(",")]
            idx_c = "][".join(self._map_val(i) for i in idx_parts)
            ctype = "float"
            type_m = re.search(r"memref<[\d?x]*x?(\w+)>", line)
            if type_m:
                ctype = self.TYPE_MAP.get(type_m.group(1), type_m.group(1))
            cvar = self._def_var(dst, "ld", ctype)
            self._line(f"{ctype} {cvar} = {self._map_val(buf)}[{idx_c}];")

    def _emit_copy(self, line: str):
        m = re.match(r"memref\.copy\s+(%[\w]+),\s*(%[\w]+)", line)
        if m:
            src, dst = m.group(1), m.group(2)
            size_m = re.search(r"memref<(\d+)x", line)
            size = size_m.group(1) if size_m else "256"
            self._line(f"for (int _ci = 0; _ci < {size}; _ci++)")
            self.indent += 1
            self._line(f"{self._map_val(dst)}[_ci] = {self._map_val(src)}[_ci];")
            self.indent -= 1

    # ── control flow ────────────────────────────────────────────────────

    def _emit_br(self, line: str):
        m = re.match(r"cf\.br\s+\^(\w+)(?:\(([^)]+)\))?", line)
        if m:
            target, args = m.group(1), m.group(2)
            if args:
                vals = [v.strip().split(":")[0].strip() for v in args.split(",")]
                target_args = self._block_args.get(target, [])
                for i, v in enumerate(vals):
                    if i < len(target_args):
                        self._line(f"{target_args[i]} = {self._map_val(v)};")
            self._line(f"goto {target};")

    def _emit_cond_br(self, line: str):
        m = re.match(
            r"cf\.cond_br\s+(%[\w]+),\s*\^(\w+)(?:\(([^)]*)\))?,\s*\^(\w+)(?:\(([^)]*)\))?",
            line)
        if m:
            cond = m.group(1)
            true_bb, true_args = m.group(2), m.group(3)
            false_bb, false_args = m.group(4), m.group(5)
            self._line(f"if ({self._map_val(cond)}) {{")
            self.indent += 1
            if true_args:
                vals = [v.strip().split(":")[0].strip() for v in true_args.split(",")]
                for i, v in enumerate(vals):
                    ta = self._block_args.get(true_bb, [])
                    if i < len(ta):
                        self._line(f"{ta[i]} = {self._map_val(v)};")
            self._line(f"goto {true_bb};")
            self.indent -= 1
            self._line("} else {")
            self.indent += 1
            if false_args:
                vals = [v.strip().split(":")[0].strip() for v in false_args.split(",")]
                for i, v in enumerate(vals):
                    fa = self._block_args.get(false_bb, [])
                    if i < len(fa):
                        self._line(f"{fa[i]} = {self._map_val(v)};")
            self._line(f"goto {false_bb};")
            self.indent -= 1
            self._line("}")
