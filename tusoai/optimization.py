from __future__ import annotations

import ast
import importlib.util
import copy
import inspect
import hashlib
import contextlib
import itertools
import json
import os
import random
import re
import resource
import string
import subprocess
import sys
import time
import threading
import trace
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


from tusoai.fs_utils import copyfile_portable, copytree_portable, rmtree_portable
from tusoai.llm import run_prompt, submit_default_prompt_batch, poll_default_prompt_batch
from tusoai.optimization_prompts import (
    build_mutation_base_prompt,
    build_mutation_repair_prompt,
    build_seed_base_prompt,
    build_seed_repair_prompt,
)

MEM_LIMIT = 50
SENSITIVE_DATA_NO_WRITE_HINT = (
    "Do not save, write, persist, export, or log any additional information to files, "
    "dataframes/tables, or external storage beyond what is strictly required for returning the function output."
)
_GPU_STICKY_BY_PID: Dict[int, Optional[int]] = {}


def _indent_code_block(code: str, indent: str) -> str:
    """Rebase code to target indent without adding an extra level."""
    lines = code.strip("\n").splitlines()
    if not lines:
        return code

    first_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        return code

    first = lines[first_idx]
    base_ws = first[: len(first) - len(first.lstrip(" \t"))]

    out: List[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if line.startswith(base_ws):
            relative = line[len(base_ws):]
        else:
            relative = line.lstrip(" \t")
        out.append(f"{indent}{relative}")
    return "\n".join(out)


def _dm_target_def_name(name: str) -> str:
    """Return the Python def name for a target key (supports 'Class.method')."""
    return str(name).split(".")[-1].strip()


def _dm_split_scoped_target(name: str) -> Tuple[List[str], str]:
    parts = [p.strip() for p in str(name).split(".") if p.strip()]
    if not parts:
        return [], ""
    return parts[:-1], parts[-1]


def _dm_find_scoped_python_block(
    content: str,
    scoped_name: str,
    node_types: Tuple[type, ...],
) -> Optional[Tuple[str, str]]:
    """Find a scoped Python block and return (source_block, indent) or None."""
    class_path, target_name = _dm_split_scoped_target(scoped_name)
    if not target_name:
        return None
    if len(class_path) == 0 and ast.ClassDef not in node_types:
        return None
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    lines = content.splitlines()
    line_starts = [0]
    for ln in lines:
        line_starts.append(line_starts[-1] + len(ln) + 1)

    match_node: Optional[ast.AST] = None

    def _walk(nodes: List[ast.stmt], stack: List[str]) -> None:
        nonlocal match_node
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                if ast.ClassDef in node_types and stack == class_path and node.name == target_name:
                    match_node = node
                    return
                _walk(list(node.body), stack + [node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, node_types) and stack == class_path and node.name == target_name:
                    match_node = node
                    return
                # no recursion into function bodies

    _walk(list(tree.body), [])
    if match_node is None or getattr(match_node, "lineno", None) is None or getattr(match_node, "end_lineno", None) is None:
        return None

    decorator_lines = [getattr(dec, "lineno", match_node.lineno) for dec in getattr(match_node, "decorator_list", [])]
    start_lineno = min([match_node.lineno, *decorator_lines])
    start = line_starts[start_lineno - 1]
    end = line_starts[match_node.end_lineno] if match_node.end_lineno < len(line_starts) else len(content)
    block = content[start:end].rstrip()
    first_line = lines[start_lineno - 1]
    indent = first_line[: len(first_line) - len(first_line.lstrip(" \t"))]
    return block, indent


def _dm_find_scoped_function_block(content: str, scoped_name: str) -> Optional[Tuple[str, str]]:
    """
    Find exactly one function block by scoped name (e.g., 'MyClass.sample').
    Returns (source_block, indent) or None.
    """
    return _dm_find_scoped_python_block(content, scoped_name, (ast.FunctionDef, ast.AsyncFunctionDef))


def _dm_find_scoped_class_block(content: str, scoped_name: str) -> Optional[Tuple[str, str]]:
    """Find exactly one class block by name or scoped class name (e.g., 'Outer.Inner')."""
    return _dm_find_scoped_python_block(content, scoped_name, (ast.ClassDef,))


def replace_functions(file_path, function_names, replacement_function):
    """Replace selected functions in a file while preserving method indentation."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    all_functions = extract_functions(content, include_nested=True)
    for target_name in function_names:
        scoped = _dm_find_scoped_function_block(content, target_name)
        if scoped is not None:
            original_block, indent = scoped
            replacement_block = _indent_code_block(replacement_function, indent)
            if original_block in content:
                content = content.replace(original_block, replacement_block, 1)
            continue

        target_def_name = _dm_target_def_name(target_name)
        matches: List[Tuple[str, str]] = []
        for function in all_functions:
            header = re.search(r"^([ \t]*)(?:async\s+)?def\s+(\w+)\(", function, re.MULTILINE)
            if header and header.group(2) == target_def_name:
                matches.append((function, header.group(1)))

        if not matches:
            class_block = _dm_find_scoped_class_block(content, target_name)
            if class_block is None:
                continue
            original_block, indent = class_block
        else:
            original_block, indent = matches[-1]

        replacement_block = _indent_code_block(replacement_function, indent)
        if original_block in content:
            content = content.replace(original_block, replacement_block, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def set_memory_limit():
    limit_in_bytes = MEM_LIMIT * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, limit_in_bytes))
def _prepend_lib_paths(env: dict) -> dict:
    """Ensure conda runtime libs are found before /lib64 without reinstalling."""
    env = dict(env)  # copy
    prefix = env.get("CONDA_PREFIX") or sys.prefix

    # Candidate library dirs (existing ones will be used, missing ignored)
    candidates = [
        Path(prefix) / "lib",
        Path(prefix) / "x86_64-conda-linux-gnu" / "lib",
        Path(prefix) / "lib" / "gcc" / "x86_64-conda-linux-gnu",
    ]

    # Also include any versioned gcc subdirs (e.g., .../gcc/x86_64-conda-linux-gnu/15.1.0)
    gcc_root = Path(prefix) / "lib" / "gcc" / "x86_64-conda-linux-gnu"
    if gcc_root.is_dir():
        for p in gcc_root.iterdir():
            if p.is_dir():
                candidates.append(p)

    existing = [str(p) for p in candidates if p.exists()]
    current = env.get("LD_LIBRARY_PATH", "")
    # Prepend conda paths so they win over /lib64
    env["LD_LIBRARY_PATH"] = ":".join(existing + ([current] if current else []))

    return env

def run_and_evaluate(
    script_name,
    timeout=600,
    val_limit=None,
    python_paths: Optional[Sequence[str]] = None,
    target_functions: Optional[Sequence[str]] = None,
    sensitive_data: bool = False,
    gpu_id: Optional[int] = None,
):
    command = [sys.executable, script_name]
    env = _prepend_lib_paths(os.environ)
    existing = env.get("PYTHONPATH", "")
    extra_paths: List[str] = []

    # Ensure local imports relative to the runner script still resolve.
    script_dir = str(Path(script_name).resolve().parent)
    extra_paths.append(script_dir)

    if python_paths:
        extra_paths.extend(str(p) for p in python_paths if p)

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    ordered_extra: List[str] = []
    for p in extra_paths:
        if p and p not in seen:
            seen.add(p)
            ordered_extra.append(p)

    if ordered_extra:
        extra = ":".join(ordered_extra)
        env["PYTHONPATH"] = extra + (f":{existing}" if existing else "")
    cpu_threads_raw = str(os.environ.get("TUSOAI_CPU_THREADS_PER_JOB", "") or "").strip()
    if cpu_threads_raw:
        try:
            cpu_threads = max(1, int(cpu_threads_raw))
        except Exception:
            cpu_threads = 1
        env["OMP_NUM_THREADS"] = str(cpu_threads)
        env["MKL_NUM_THREADS"] = str(cpu_threads)
        env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        env["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
        env["BLIS_NUM_THREADS"] = str(cpu_threads)
    if gpu_id is not None:
        requested = str(gpu_id)
        current_visible = str(env.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
        if current_visible:
            tokens = [t.strip() for t in current_visible.split(",") if t.strip()]
            try:
                idx = int(gpu_id)
            except Exception:
                idx = None
            if idx is not None and 0 <= idx < len(tokens):
                # Route by index within currently visible devices (works for UUID or remapped lists).
                requested = tokens[idx]
        env["CUDA_VISIBLE_DEVICES"] = requested

    try:

        _wall_start = time.perf_counter()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            preexec_fn=set_memory_limit,
            env=env,
        )
        wall_time_s = time.perf_counter() - _wall_start

        
        stdout = result.stdout
        #print(stdout)
        # --- Core Outputs ---
        # Validation accuracy
        evaluate_match = re.search(r"tuso_evaluate:\s*(-?[\d.]+)", stdout)
        evaluation = float(evaluate_match.group(1)) if evaluate_match else "Function must return accuracy only."

        model_info = ""
        if not sensitive_data:
            target_functions = list(target_functions or [])
            captured_chunks: List[str] = []
            for fn_name in target_functions:
                diagnostics_info_match = re.search(
                    rf"tuso_fnlog_start:{re.escape(fn_name)}\n(.*?)\ntuso_fnlog_end:{re.escape(fn_name)}",
                    stdout,
                    re.DOTALL,
                )
                if diagnostics_info_match:
                    captured_chunks.append(f"[{fn_name}]\n" + diagnostics_info_match.group(1).strip())

            if captured_chunks:
                model_info = "\n\n".join(captured_chunks).strip()
            else:
                diagnostics_info_match = re.search(
                    r"tuso_model_start\n(.*?)\ntuso_model_end", stdout, re.DOTALL)
                model_info = diagnostics_info_match.group(1).strip() if diagnostics_info_match else ""

            if len(model_info) > 2000:
                model_info = model_info[0:2000]
        if isinstance(evaluation, str):
            return evaluation

        if val_limit and evaluation >= val_limit:
            return "Error: validation accuracy is suspiciously high. Some overfitting is likely to have occurred."

        return {
            "evaluation": evaluation,
            "model_info": model_info,
            "runtime":wall_time_s
        }

    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except subprocess.CalledProcessError as e:
        err = e.stderr
        if len(err)>1000:
            return err[-1000:]
        return e.stderr
    except MemoryError:
        return "Error: out of memory"

def copy_file(source, index):
    if not os.path.exists(source):
        print(f"Error: {source} does not exist.")
        return
    
    file_name, file_ext = os.path.splitext(source)
    new_file = f"{file_name}_{index}{file_ext}"
    copyfile_portable(source, new_file)
    return new_file

def extract_functions(content, include_nested=False):
    """
    Simple (no ast, no tokenize) function extractor.

    Robust to:
      - extra whitespace / blank lines
      - multi-line function signatures
      - type annotations, defaults with (), {}, []
      - files that are not fully valid Python (best-effort)

    By default matches only top-level defs.
    Set include_nested=True to also capture indented defs.
    """
    # Find "def name(" lines (optionally indented if include_nested=True)
    pat = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?:async\s+)?def[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*\(",
    )

    lines = content.splitlines(True)  # keep newlines
    line_starts = _line_start_offsets(lines)

    out = []
    for m in pat.finditer(content):
        indent_txt = m.group("indent")
        if not include_nested and indent_txt:
            continue

        def_indent = _indent_width(indent_txt)
        start_pos = m.start()

        colon_pos = _find_def_header_colon(content, m.end() - 1)
        if colon_pos is None:
            continue

        header_line_idx = _pos_to_line_index(line_starts, start_pos)
        colon_line_idx = _pos_to_line_index(line_starts, colon_pos)

        # One-liner def? (any non-space before a comment after the colon)
        line_after_colon = content[colon_pos + 1 : _line_end_pos(content, colon_pos)]
        if _first_non_ws_char(line_after_colon) not in (None, "#"):
            end_line_idx = colon_line_idx
        else:
            # Multi-line body: include lines until we hit a nonblank/noncomment line
            # with indentation <= def_indent
            end_line_idx = colon_line_idx
            k = colon_line_idx + 1
            while k < len(lines):
                s = lines[k]
                if s.strip() == "":
                    end_line_idx = k
                    k += 1
                    continue
                stripped = s.lstrip(" \t")
                if stripped.startswith("#"):
                    end_line_idx = k
                    k += 1
                    continue

                cur_indent = _indent_width(s[: len(s) - len(stripped)])
                if cur_indent <= def_indent:
                    break

                end_line_idx = k
                k += 1

        start_off = line_starts[header_line_idx]
        end_off = line_starts[end_line_idx + 1] if end_line_idx + 1 < len(line_starts) else len(content)
        out.append(content[start_off:end_off].rstrip())

    return out

def extract_function_by_name(file_path, function_name, include_nested=False):
    """
    Extract a specific function by name from a Python file.
    Includes decorators and full body. Handles multi-line defs.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    scoped = _dm_find_scoped_function_block(content, function_name)
    if scoped is not None:
        return scoped[0]

    target_def_name = _dm_target_def_name(function_name)
    matches: List[str] = []
    for fn in extract_functions(content, include_nested=include_nested):
        if re.match(rf"^[ \t]*(?:async\s+)?def[ \t]+{re.escape(target_def_name)}[ \t]*\(", fn):
            matches.append(fn.rstrip())

    if matches:
        return matches[-1]

    class_block = _dm_find_scoped_class_block(content, function_name)
    if class_block is not None:
        return class_block[0]
    return None


def extract_function_by_name_from_code(content: str, function_name: str):
    target_def_name = _dm_target_def_name(function_name)
    matches: List[str] = []
    for fn in extract_functions(content, include_nested=True):
        if re.match(rf"^[ 	]*(?:async\s+)?def[ 	]+{re.escape(target_def_name)}[ 	]*\(", fn):
            matches.append(fn.rstrip())

    if matches:
        return matches[-1]

    class_block = _dm_find_scoped_class_block(content, function_name)
    if class_block is not None:
        return class_block[0]
    return None


def _dm_normalize_history_code_block(code: str, function_name: str) -> str:
    target_def_name = _dm_target_def_name(function_name)
    text = str(code or "")
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("###", "\n###")

    # If markdown function sections exist, prefer the section matching the target function.
    heading_pat = re.compile(r"(?m)^#{2,}\s*([A-Za-z_]\w*)\s*$")
    matches = list(heading_pat.finditer(text))
    if matches:
        selected = None
        for i, m in enumerate(matches):
            if m.group(1) != target_def_name:
                continue
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            selected = text[start:end]
            break
        if selected is not None:
            text = selected

    # Drop any remaining markdown heading lines.
    text = re.sub(r"(?m)^#{2,}.*$", "", text)
    return text.strip()


def _dm_extract_function_block_with_leading_support(code: str, function_name: str) -> Optional[str]:
    """Return target function code with leading top-level support block."""
    target_def_name = _dm_target_def_name(function_name)
    blocks = _dm_extract_target_blocks_with_support(code, [function_name])
    out = blocks.get(target_def_name)
    if out:
        return out
    normalized_code = _dm_normalize_history_code_block(code, function_name)
    return extract_function_by_name_from_code(normalized_code, function_name)


def _dm_extract_target_blocks_with_support(code: str, target_names: Sequence[str]) -> Dict[str, str]:
    targets = {_dm_target_def_name(n) for n in target_names}
    normalized_code = _dm_normalize_history_code_block(code, next(iter(targets), ""))
    lines = normalized_code.splitlines()
    out: Dict[str, str] = {}

    try:
        tree = ast.parse(normalized_code)
        target_nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in targets
        ]
        target_nodes.sort(key=lambda n: n.lineno)

        for i, node in enumerate(target_nodes):
            decorator_lines = [getattr(dec, "lineno", node.lineno) for dec in getattr(node, "decorator_list", [])]
            node_start_line = min([node.lineno, *decorator_lines])
            start_line = target_nodes[i - 1].end_lineno + 1 if i > 0 else 1
            start_line = min(start_line, node_start_line)
            end_line = node.end_lineno
            if start_line is None or end_line is None:
                continue
            out[node.name] = "\n".join(lines[start_line - 1 : end_line]).rstrip()
    except Exception:
        pass

    return out


def _line_start_offsets(lines):
    offs = [0]
    total = 0
    for s in lines:
        total += len(s)
        offs.append(total)
    return offs


def _pos_to_line_index(line_starts, pos):
    # rightmost start <= pos
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return max(0, min(lo, len(line_starts) - 2))


def _line_end_pos(text, pos):
    nl = text.find("\n", pos)
    return len(text) if nl == -1 else nl


def _indent_width(ws):
    return len(ws.expandtabs(8))


def _first_non_ws_char(s):
    for ch in s:
        if ch not in " \t\r\n":
            return ch
    return None


def _find_def_header_colon(src, start_pos):
    """
    Scan forward from start_pos to find the ':' that ends the def signature.
    Tracks (), [], {} nesting; ignores strings and comments (best-effort).
    """
    depth = 0
    i = start_pos
    in_str = None  # "'", '"', "'''", '"""'

    while i < len(src):
        ch = src[i]

        if in_str is not None:
            if len(in_str) == 3:
                if src.startswith(in_str, i):
                    in_str = None
                    i += 3
                    continue
                i += 1
                continue
            else:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue

        # not in string
        if ch == "#":
            j = src.find("\n", i)
            if j == -1:
                return None
            i = j + 1
            continue

        if ch in ("'", '"'):
            if src.startswith(ch * 3, i):
                in_str = ch * 3
                i += 3
                continue
            in_str = ch
            i += 1
            continue

        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == ":" and depth == 0:
            return i

        i += 1

    return None

def extract_function(result: str) -> str:
    """
    Extract all import statements and top-level function *and class* definitions
    from either:
      1) the first ```python ... ``` code block in `result`, if both fences exist, or
      2) the region starting at the line that starts with 'def tuso_model' and ending at:
         - the first subsequent non-empty line whose indentation is <= the 'def' line's indentation
           (i.e., a new top-level line), or
         - the end of the text if no such line exists.

    Returns a string containing imports followed by the full blocks (classes/functions).
    """
    start_marker = "```python"
    end_marker = "```"

    # If the start/end markers don't exist at all, wrap the whole input.
    if start_marker not in result and end_marker not in result:
        result = f"{start_marker}\n{result.rstrip()}\n{end_marker}"

    code_block: str | None = None

    # Case 1: Use fenced code block only if we can find an end fence AFTER the start fence.
    start_pos = result.find(start_marker)
    if start_pos != -1:
        end_pos = result.find(end_marker, start_pos + len(start_marker))
        if end_pos != -1:
            code_block = result[start_pos + len(start_marker) : end_pos]
            # Trim a single leading newline if present (common after ```python)
            if code_block.startswith("\n"):
                code_block = code_block[1:]

    if code_block == None:
        return ""
    # From here on, operate within the chosen code_block.
    lines = code_block.splitlines()
    import_lines: list[str] = []
    blocks: list[str] = []

    current_block: list[str] = []
    in_block = False
    block_indent = 0

    # Collect decorators that immediately precede a class/def
    pending_decorators: list[str] = []

    def starts_block(s: str) -> bool:
        s = s.lstrip()
        return s.startswith("class ") or s.startswith("def ") or s.startswith("async def ")

    for line in lines:
        stripped = line.lstrip()

        # Collect import statements anywhere they appear
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line.lstrip())
            continue

        if in_block:
            # Continue collecting while indented more than the block header,
            # or if the line is blank (to keep blank lines inside the block).
            indent = len(line) - len(stripped)
            if stripped == "" or indent > block_indent:
                current_block.append(line)
                continue
            else:
                # Block ended; store it, then fall through to possibly start a new one
                blocks.append("\n".join(current_block))
                in_block = False
                current_block = []
                # Do not continue; re-evaluate this line below.

        # Handle decorators that immediately precede class/def
        if stripped.startswith("@") and not in_block:
            pending_decorators.append(line)
            continue

        # Start of a new block (class or def)
        if starts_block(line):
            # The block's effective indent is that of the *first decorator* (if any),
            # otherwise the header line itself.
            if pending_decorators:
                block_indent = len(pending_decorators[0]) - len(pending_decorators[0].lstrip())
                current_block = pending_decorators + [line]
            else:
                block_indent = len(line) - len(stripped)
                current_block = [line]
            in_block = True
            pending_decorators = []
            continue

        # Any other non-decorator line clears pending decorators (they didn't belong to a block)
        if pending_decorators:
            pending_decorators = []

        # Ignore other lines outside of blocks

    # Append the last open block if any
    if in_block and current_block:
        blocks.append("\n".join(current_block))

    # Nothing to return?
    if not import_lines and not blocks:
        return ""

    # Combine imports and blocks; keep appearance order within each category
    parts: list[str] = []
    if import_lines:
        parts.extend(import_lines)
        parts.append("")
    parts.extend(blocks)
    return "\n".join(parts)

feedback_example = """
Old function had preprocessing that did median/mode imputation + standard scaling and added PolynomialFeatures, then trained a fixed L2-regularized LogisticRegression (C=1.0) with no tuning.
The optimization drops polynomial expansion, uses sparse-friendly scaling and groups rare categories to stabilize one-hot features.
""".strip()


def _first_two_nonempty_lines(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:2])


def generate_feedback(new_code, old_code, new_perf, old_perf, min_improvement):
    """
    Generates a 2-line, factual summary of old vs new code changes (ML pipeline deltas),
    then appends a standardized outcome line based on performance delta.

    Assumes higher performance is better.
    """
    prompt = f"""
You write high-signal, factual change summaries for ML pipeline edits. Never speculate.
Ignore trivial refactors (renames, formatting, code organization).

TASK
Write EXACTLY two lines summarizing differences between the old function and the proposed optimization.

INPUTS
OLD_CODE:
{old_code}

NEW_CODE:
{new_code}

STYLE EXAMPLE (match this style/tone exactly):
{feedback_example}

FORMAT REQUIREMENTS
- Output exactly TWO lines, no bullets, no numbering, no extra text.
- Line 1 must start with: Old function had ...
- Line 2 must start with: The optimization ...
- Each line must be a single sentence.
- Mention only substantive changes (preprocessing, features, model family, tuning, objective, CV strategy, etc.).
- Use only facts visible in the code (no guesses about intent).
""".strip()

    rp = run_prompt(prompt)
    if isinstance(rp, tuple):
        response, cost = rp
    else:
        response, cost = rp, 0.0

    # Enforce the 2-line contract even if the model slips
    summary = _first_two_nonempty_lines(response)

    # Compute percent change robustly
    denom = abs(old_perf) if abs(old_perf) > 1e-12 else 1.0
    perf_percent = 100.0 * (new_perf - old_perf) / denom

    # Correct threshold logic: improved if >= +min_improvement, worse if <= -min_improvement
    thr = .01
    if perf_percent >= thr:
        outcome = f"Outcome: +{perf_percent:.4f}% (improved)."
    elif perf_percent <= -thr:
        outcome = f"Outcome: {perf_percent:.4f}% (worse)."
    else:
        outcome = "Outcome: ~0% (no meaningful change)."

    feedback = summary + "\n" + outcome
    return feedback, cost

# ======================================================================================
# Task objects
# ======================================================================================

@dataclass(slots=True)
class Task:
    function_name: str
    knowledge_tree: Dict[str, List[str]] = field(default_factory=dict)
    kt_prob: Dict[str, float] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    repo_root: Optional[str] = None


@dataclass(slots=True)
class MethodTask(Task):
    initial_solutions: List[str] = field(default_factory=list)
    use_initial: bool = True


@dataclass(slots=True)
class DataTask(Task):
    data_summary: str = ""
    data_usage: str = ""
    file_description: str = ""
    read_cmd: str = ""


# ======================================================================================
# GA containers
# ======================================================================================

@dataclass
class ModelRecord:
    """Container for one island-wide variant (a full set of functions)."""
    code: str
    file: Path
    accuracy: float
    runtime: float
    model_info: Dict | None = None
    lineage: str = "root"
    functions: Dict[str, str] = field(default_factory=dict)  # fn_name -> fn_code

    def __hash__(self) -> int:
        return hash(self.code)


@dataclass
class Island:
    id: int
    models: List[ModelRecord] = field(default_factory=list)

    def champion(self) -> ModelRecord:
        return max(self.models, key=lambda m: m.accuracy)


# ======================================================================================
# Helpers
# ======================================================================================

_CODE_BLOCK_RE = re.compile(r"BEGIN_CODE\s*(.*?)\s*END_CODE", flags=re.S)
_FEEDBACK_BLOCK_RE = re.compile(r"<f>\s*(.*?)\s*(?:</f>|<\\f>)", flags=re.S | re.I)


def _extract_code_from_reply(llm_reply: str) -> str:
    """
    Extract code between BEGIN_CODE ... END_CODE if present; otherwise fall back to extract_function(reply).
    NOTE: extract_function(reply) is expected to exist in the runtime (external utility).
    """
    if not llm_reply:
        return ""
    m = _CODE_BLOCK_RE.search(llm_reply)
    if m:
        return (m.group(1) or "").strip()
    # fall back to external extractor (kept for backward compatibility)
    return extract_function(llm_reply)


def _extract_feedback_from_reply(llm_reply: str) -> str:
    if not llm_reply:
        return ""
    m = _FEEDBACK_BLOCK_RE.search(llm_reply)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _merge_feedback_with_outcome(base_feedback: str, new_perf: float, old_perf: float) -> str:
    summary = (base_feedback or "").strip()
    denom = abs(old_perf) if abs(old_perf) > 1e-12 else 1.0
    perf_percent = 100.0 * (new_perf - old_perf) / denom
    thr = 0.01
    if perf_percent >= thr:
        outcome = f"Outcome: +{perf_percent:.4f}% (improved)."
    elif perf_percent <= -thr:
        outcome = f"Outcome: {perf_percent:.4f}% (worse)."
    else:
        outcome = "Outcome: ~0% (no meaningful change)."
    return (summary + "\n" + outcome).strip() if summary else outcome


def _ensure_function_name(code: str, function_name: str, placeholder: str = "tuso_model") -> str:
    """Rename the target function or class in `code` to `function_name` (prefers placeholders)."""
    target_def_name = _dm_target_def_name(function_name)
    def_pat = r"(^\s*(?:async\s+)?def\s+){}(\s*\()"
    class_pat = r"(^\s*class\s+){}(\s*(?:\(|:))"

    if re.search(def_pat.format(re.escape(target_def_name)), code, flags=re.M):
        return code
    if re.search(class_pat.format(re.escape(target_def_name)), code, flags=re.M):
        return code
    if re.search(def_pat.format(re.escape(placeholder)), code, flags=re.M):
        return re.sub(
            def_pat.format(re.escape(placeholder)),
            rf"\1{target_def_name}\2",
            code,
            count=1,
            flags=re.M,
        )
    if re.search(class_pat.format(re.escape(placeholder)), code, flags=re.M):
        return re.sub(
            class_pat.format(re.escape(placeholder)),
            rf"\1{target_def_name}\2",
            code,
            count=1,
            flags=re.M,
        )
    renamed = re.sub(
        r"(^\s*(?:async\s+)?def\s+)([A-Za-z_]\w*)(\s*\()",
        rf"\1{target_def_name}\3",
        code,
        count=1,
        flags=re.M,
    )
    if renamed != code:
        return renamed
    return re.sub(
        r"(^\s*class\s+)([A-Za-z_]\w*)(\s*(?:\(|:))",
        rf"\1{target_def_name}\3",
        code,
        count=1,
        flags=re.M,
    )


def _bundle_functions_text(functions: Dict[str, str], ordered_names: Sequence[str]) -> str:
    """Stack all functions into one big string, in a fixed order, for textual clustering."""
    if len(ordered_names) == 1:
        return functions[ordered_names[0]].rstrip() + "\n"
    parts: List[str] = []
    for fn in ordered_names:
        parts.append(f"### {fn}\n{functions[fn].rstrip()}\n")
    return "\n".join(parts).strip() + "\n"


def _dm_code_len_lines(code: str) -> int:
    return sum(1 for ln in str(code or "").splitlines() if ln.strip())


def _instrument_function_print_capture(fn_code: str, function_name: str) -> str:
    """Wrap target function so prints from inside it are tagged for diagnostics extraction."""
    target_def_name = _dm_target_def_name(function_name)
    inner_name = f"_tuso_inner_{target_def_name}"
    header_match = re.search(
        rf"^([ \t]*)(async[ \t]+)?def[ \t]+{re.escape(target_def_name)}([ \t]*\()",
        fn_code,
        flags=re.MULTILINE,
    )
    if not header_match:
        return fn_code

    indent = header_match.group(1) or ""
    is_async = bool(header_match.group(2))

    renamed = re.sub(
        rf"(^[ \t]*(?:async[ \t]+)?def[ \t]+){re.escape(target_def_name)}([ \t]*\()",
        rf"\1{inner_name}\2",
        fn_code,
        count=1,
        flags=re.MULTILINE,
    )

    body_indent = indent + "    "
    def_kw = "async def" if is_async else "def"

    wrapper_lines = [
        "",
        f"{indent}{def_kw} {target_def_name}(*args, **kwargs):",
        f"{body_indent}import io",
        f"{body_indent}import contextlib",
        "",
        f"{body_indent}_tuso_buf = io.StringIO()",
        f"{body_indent}try:",
        f"{body_indent}    with contextlib.redirect_stdout(_tuso_buf):",
        f'{body_indent}        _tuso_target = getattr(args[0], "{inner_name}", None) if args else None',
        f"{body_indent}        if _tuso_target is None:",
        f'{body_indent}            _tuso_target = globals().get("{inner_name}")',
        f"{body_indent}        if not callable(_tuso_target):",
        f'{body_indent}            raise NameError("Instrumented target not found: {inner_name}")',
    ]
    if is_async:
        wrapper_lines.append(f"{body_indent}        _tuso_result = await _tuso_target(*args[1:], **kwargs) if args and hasattr(args[0], \"{inner_name}\") else await _tuso_target(*args, **kwargs)")
    else:
        wrapper_lines.append(f"{body_indent}        _tuso_result = _tuso_target(*args[1:], **kwargs) if args and hasattr(args[0], \"{inner_name}\") else _tuso_target(*args, **kwargs)")
    wrapper_lines.extend([
        f"{body_indent}    return _tuso_result",
        f"{body_indent}finally:",
        f"{body_indent}    _tuso_captured = _tuso_buf.getvalue()",
        f"{body_indent}    if _tuso_captured:",
        f'{body_indent}        print("tuso_fnlog_start:{function_name}")',
        f'{body_indent}        print(_tuso_captured, end="" if _tuso_captured.endswith("\\n") else "\\n")',
        f'{body_indent}        print("tuso_fnlog_end:{function_name}")',
    ])
    wrapper = "\n".join(wrapper_lines).rstrip()

    return renamed.rstrip() + "\n" + wrapper + "\n"


def update_probabilities(probabilities: Dict[str, float], key: str, factor: float) -> None:
    """Multiply probability[key] by factor and renormalize in-place."""
    if not probabilities or key not in probabilities:
        return
    probabilities[key] *= factor
    total = sum(probabilities.values())
    if total <= 0:
        n = len(probabilities)
        for k in probabilities:
            probabilities[k] = 1.0 / n
        return
    for k in probabilities:
        probabilities[k] /= total


def _pick_cluster_champion(models: List[ModelRecord], *, min_improvement: float) -> ModelRecord:
    """Best acc, then among near-best pick highest complexity score."""
    best_acc = max(m.accuracy for m in models)
    close: List[ModelRecord] = []
    for m in models:
        if best_acc == 0:
            if m.accuracy == 0:
                close.append(m)
        else:
            if best_acc - m.accuracy < min_improvement:
                close.append(m)

    # Top performer defines normalization anchors.
    top_model = max(close, key=lambda m: m.accuracy)
    top_code_length = max(1, sum(1 for ln in top_model.code.splitlines() if ln.strip()))
    top_code_runtime = max(1.0, float(top_model.runtime or 0.0))

    def _complexity_score(m: ModelRecord) -> float:
        method_code_length = max(1, sum(1 for ln in m.code.splitlines() if ln.strip()))
        method_code_runtime = max(1.0, float(m.runtime or 0.0))
        return (top_code_length / method_code_length) * (top_code_runtime / method_code_runtime)

    return max(close, key=_complexity_score)


def _dm_history_close_set(models: List["ModelRecord"], *, min_improvement: float) -> List["ModelRecord"]:
    """Return models within min_improvement of the highest accuracy."""
    if not models:
        return []
    best_acc = max(float(m.accuracy) for m in models)
    close: List[ModelRecord] = []
    for m in models:
        if best_acc == 0:
            if float(m.accuracy) == 0:
                close.append(m)
        else:
            if best_acc - float(m.accuracy) < float(min_improvement):
                close.append(m)
    return close or list(models)


def _dm_history_complexity_score(candidate: "ModelRecord", anchor: "ModelRecord") -> float:
    """Match optimization tie-break scoring: (anchor_len/cand_len) * (anchor_rt/cand_rt)."""
    anchor_len = max(1, _dm_code_len_lines(anchor.code))
    anchor_rt = max(1.0, float(anchor.runtime or 0.0))
    cand_len = max(1, _dm_code_len_lines(candidate.code))
    cand_rt = max(1.0, float(candidate.runtime or 0.0))
    return (anchor_len / cand_len) * (anchor_rt / cand_rt)


def _dm_extract_model_info_for_function(model_info: Any, fn_name: str) -> str:
    text = "" if model_info is None else str(model_info)
    if not text.strip():
        return ""
    m = re.search(rf"\[{re.escape(fn_name)}\]\n(.*?)(?:\n\n\[|\Z)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _dm_find_original_record(models: Sequence["ModelRecord"]) -> Optional["ModelRecord"]:
    for m in models:
        lineage = str(getattr(m, "lineage", "") or "")
        if "seed_baseline" in lineage or lineage.endswith("baseline"):
            return m
    return None



# ======================================================================================
# Internal helpers (refactor only; no behavioral changes)
# ======================================================================================

@dataclass
class _DMRunState:
    start_time: float
    debug: bool
    total_cost: float = 0.0

    def now_s(self) -> float:
        return time.time() - self.start_time

    @staticmethod
    def money(x: float) -> str:
        return f"${x:.4f}"

    def add_cost(self, cost: float) -> None:
        self.total_cost += cost


class _DMPrinter:
    def __init__(self, state: _DMRunState):
        self.state = state

    def hdr(self, title: str) -> None:
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)

    def subhdr(self, title: str) -> None:
        print("\n" + "-" * 100)
        print(title)
        print("-" * 100)

    def kv(self, key: str, val: Any, *, indent: int = 0) -> None:
        pad = " " * indent
        if isinstance(val, (dict, list)):
            txt = json.dumps(val, ensure_ascii=False, indent=2)
            print(f"{pad}{key}:")
            for line in txt.splitlines():
                print(f"{pad}  {line}")
        else:
            print(f"{pad}{key}: {val}")

    def brief(self, stage: str, msg: str) -> None:
        print(f"[t={self.state.now_s():7.1f}s | cost={self.state.money(self.state.total_cost)}] {stage}: {msg}")

    def debug_dump(self, stage: str, entry: Dict[str, Any]) -> None:
        if not self.state.debug:
            return
        self.subhdr(f"DEBUG | {stage} | t={self.state.now_s():.1f}s | cost={self.state.money(self.state.total_cost)}")

        entry = dict(entry)  # avoid mutating caller
        prompt = entry.pop("prompt", None)
        reply = entry.pop("reply_raw", None)
        code = entry.pop("extracted_code", None)

        for k in sorted(entry.keys()):
            self.kv(k, entry[k])

        if prompt is not None:
            print("\n--- PROMPT ---")
            print(prompt)
        if reply is not None:
            print("\n--- LLM REPLY (RAW) ---")
            print(reply)
        if code is not None:
            print("\n--- EXTRACTED CODE ---")
            print(code)


_DM_HISTORY_FORMAT = "tusoai.history.v2"
_DM_HISTORY_STRING_REF_KEY = "__tusoai_history_string_ref__"
_DM_HISTORY_LITERAL_KEY = "__tusoai_history_literal__"
_DM_HISTORY_INTERN_MIN_LENGTH = 32
_DM_LOG_LOCKS_GUARD = threading.Lock()
_DM_LOG_LOCKS: Dict[str, threading.Lock] = {}


def _dm_log_path_key(path: Union[str, Path]) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _dm_log_file_lock(path: Union[str, Path]) -> threading.Lock:
    key = _dm_log_path_key(path)
    with _DM_LOG_LOCKS_GUARD:
        lock = _DM_LOG_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DM_LOG_LOCKS[key] = lock
        return lock


def _dm_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _dm_write_compact_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _dm_read_json_list(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else []


def _dm_history_string_counts(obj: Any, counts: Dict[str, int]) -> None:
    if isinstance(obj, str):
        counts[obj] += 1
    elif isinstance(obj, list):
        for item in obj:
            _dm_history_string_counts(item, counts)
    elif isinstance(obj, dict):
        for value in obj.values():
            _dm_history_string_counts(value, counts)


def _dm_encode_history_value(obj: Any, string_refs: Dict[str, int]) -> Any:
    if isinstance(obj, str):
        idx = string_refs.get(obj)
        if idx is not None:
            return {_DM_HISTORY_STRING_REF_KEY: idx}
        return obj
    if isinstance(obj, list):
        return [_dm_encode_history_value(item, string_refs) for item in obj]
    if isinstance(obj, dict):
        encoded = {str(k): _dm_encode_history_value(v, string_refs) for k, v in obj.items()}
        if len(obj) == 1 and next(iter(obj.keys()), None) in {_DM_HISTORY_STRING_REF_KEY, _DM_HISTORY_LITERAL_KEY}:
            return {_DM_HISTORY_LITERAL_KEY: encoded}
        return encoded
    return obj


def _dm_decode_history_value(obj: Any, strings: Sequence[str], *, literal: bool = False) -> Any:
    if isinstance(obj, list):
        return [_dm_decode_history_value(item, strings) for item in obj]
    if isinstance(obj, dict):
        if not literal and len(obj) == 1 and _DM_HISTORY_STRING_REF_KEY in obj:
            idx = obj[_DM_HISTORY_STRING_REF_KEY]
            if not isinstance(idx, int) or idx < 0 or idx >= len(strings):
                raise ValueError(f"Invalid history string reference: {idx!r}")
            return strings[idx]
        if not literal and len(obj) == 1 and _DM_HISTORY_LITERAL_KEY in obj:
            return _dm_decode_history_value(obj[_DM_HISTORY_LITERAL_KEY], strings, literal=True)
        return {str(k): _dm_decode_history_value(v, strings) for k, v in obj.items()}
    return obj


def _dm_encode_history_log(entries: List[dict]) -> Any:
    counts: Dict[str, int] = defaultdict(int)
    _dm_history_string_counts(entries, counts)
    strings = [s for s, count in counts.items() if count > 1 and len(s) >= _DM_HISTORY_INTERN_MIN_LENGTH]
    if not strings:
        return entries
    string_refs = {s: i for i, s in enumerate(strings)}
    return {
        "format": _DM_HISTORY_FORMAT,
        "strings": strings,
        "history": _dm_encode_history_value(entries, string_refs),
    }


def _dm_history_entries_from_data(data: Any) -> List[dict]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    entries = data.get("history", [])
    if data.get("format") == _DM_HISTORY_FORMAT:
        strings = data.get("strings", [])
        if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
            raise ValueError("Invalid compact history string table")
        entries = _dm_decode_history_value(entries, strings)
    return entries if isinstance(entries, list) else []


def _dm_read_history_entries(history_path: Union[str, Path]) -> List[dict]:
    path = Path(history_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    return _dm_history_entries_from_data(data)


@contextlib.contextmanager
def _dm_history_file_lock(history_path: Union[str, Path]):
    """Best-effort cross-process lock for shared optimization histories."""
    lock_path = Path(history_path).with_suffix(Path(history_path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl  # Unix advisory locking; unavailable on some platforms.
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        fh.close()


@dataclass
class _DMLogSinks:
    history_path: Path
    dev_path: Path
    prompt_io_path: Path
    history_log: List[dict] = field(default_factory=list)
    dev_log: List[dict] = field(default_factory=list)
    prompt_io_log: List[dict] = field(default_factory=list)
    multi_machine: bool = False
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _history_seq: int = 0

    def log_dev(self, entry: dict) -> None:
        with _dm_log_file_lock(self.dev_path):
            existing = _dm_read_json_list(self.dev_path)
            existing.append(dict(entry))
            self.dev_log = existing
            _dm_write_json(self.dev_path, existing)

    def log_history(self, entry: dict) -> None:
        with _dm_log_file_lock(self.history_path):
            entry = dict(entry)
            if not self.multi_machine:
                existing = _dm_read_history_entries(self.history_path)
                existing.append(entry)
                self.history_log = existing
                _dm_write_compact_json(self.history_path, _dm_encode_history_log(existing))
                return

            self._history_seq += 1
            entry.setdefault("run_id", self.run_id)
            entry.setdefault("history_seq", self._history_seq)
            with _dm_history_file_lock(self.history_path):
                existing = _dm_read_history_entries(self.history_path)
                seen = {
                    (str(e.get("run_id")), int(e.get("history_seq")))
                    for e in existing
                    if isinstance(e, dict) and e.get("run_id") is not None and e.get("history_seq") is not None
                }
                key = (str(entry.get("run_id")), int(entry.get("history_seq", 0)))
                if key not in seen:
                    existing.append(entry)
                self.history_log = existing
                _dm_write_compact_json(self.history_path, _dm_encode_history_log(existing))

    def log_prompt_io(self, entry: dict) -> None:
        with _dm_log_file_lock(self.prompt_io_path):
            existing = _dm_read_json_list(self.prompt_io_path)
            existing.append(dict(entry))
            self.prompt_io_log = existing
            _dm_write_json(self.prompt_io_path, existing)


def _dm_log_run_wiring(logs: _DMLogSinks, eval_ctx: "_DMEvalContext", *, mode: str) -> None:
    """Persist run wiring metadata needed to rebuild history selections later."""
    logs.log_dev(
        {
            "stage": "run_wiring",
            "mode": mode,
            "reference_filename": str(eval_ctx.reference_filename),
            "base_path": str(eval_ctx.base_path),
            "ordered_fn_names": list(eval_ctx.ordered_fn_names),
            "function_sources": dict(eval_ctx.function_sources),
            "repo_snapshots": {k: str(v) for k, v in eval_ctx.repo_snapshots.items()},
            "timeout": int(eval_ctx.timeout),
            "val_limit": eval_ctx.val_limit,
            "sensitive_data": bool(eval_ctx.sensitive_data),
        }
    )


def _dm_prepare_tasks(
    method_tasks: List["MethodTask"],
    data_tasks: List["DataTask"],
) -> Tuple[List[Tuple[str, str, "Task"]], List[str], Dict[str, Tuple[str, "Task"]], bool]:
    ordered_tasks: List[Tuple[str, str, "Task"]] = []
    for t in method_tasks:
        ordered_tasks.append((t.function_name, "method", t))
    for t in data_tasks:
        ordered_tasks.append((t.function_name, "data", t))

    ordered_fn_names = [fn for fn, _, _ in ordered_tasks]
    task_by_name: Dict[str, Tuple[str, "Task"]] = {fn: (kind, obj) for fn, kind, obj in ordered_tasks}
    compat_single = (len(method_tasks) == 1 and len(data_tasks) == 0)
    return ordered_tasks, ordered_fn_names, task_by_name, compat_single


def _dm_init_task_probs(ordered_fn_names: List[str]) -> Dict[str, float]:
    task_probs: Dict[str, float] = {fn: 1.0 for fn in ordered_fn_names}
    if ordered_fn_names:
        update_probabilities(task_probs, ordered_fn_names[0], 1.0)
    return task_probs


def _dm_extract_base_functions(
    ordered_fn_names: List[str],
    function_sources: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    base_functions: Dict[str, str] = {}
    for fn_name in ordered_fn_names:
        source = function_sources[fn_name]
        file_path = source["file_path"]
        code = extract_function_by_name(file_path, fn_name, include_nested=True)
        if not code:
            tpl = extract_function_by_name(file_path, "tuso_model", include_nested=True)
            if not tpl:
                raise RuntimeError(f"Could not extract baseline for '{fn_name}' from {file_path}")
            code = _ensure_function_name(tpl, fn_name, placeholder="tuso_model")
        base_functions[fn_name] = _ensure_function_name(code, fn_name, placeholder="tuso_model")
    return base_functions


def _dm_load_functions_from_history(
    history_path: str,
    ordered_fn_names: Sequence[str],
    *,
    accuracy_tolerance: float = 0.0,
) -> Optional[Dict[str, str]]:
    path = Path(history_path)
    if not path.exists():
        raise FileNotFoundError(f"load_history path not found: {history_path}")

    entries = _dm_read_history_entries(path)

    candidates: List[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        code = e.get("code")
        acc = e.get("accuracy")
        if not isinstance(code, str):
            continue
        try:
            acc_f = float(acc)
        except Exception:
            continue
        runtime = e.get("runtime")
        try:
            runtime_f = float(runtime) if runtime is not None else float("inf")
        except Exception:
            runtime_f = float("inf")
        candidates.append({"code": code, "accuracy": acc_f, "runtime": runtime_f})

    if not candidates:
        return None

    model_candidates: List[ModelRecord] = [
        ModelRecord(code=c["code"], file=None, accuracy=float(c["accuracy"]), runtime=float(c["runtime"]))
        for c in candidates
    ]
    close = _dm_history_close_set(model_candidates, min_improvement=max(0.0, float(accuracy_tolerance or 0.0)))
    top_accuracy_model = max(close, key=lambda m: float(m.accuracy))
    ranked = sorted(close, key=lambda m: _dm_history_complexity_score(m, top_accuracy_model), reverse=True)

    for cand_m in ranked:
        funcs: Dict[str, str] = {}
        ok = True
        blocks = _dm_extract_target_blocks_with_support(cand_m.code, ordered_fn_names)
        for fn in ordered_fn_names:
            fn_code = blocks.get(fn) or _dm_extract_function_block_with_leading_support(cand_m.code, fn)
            if not fn_code:
                ok = False
                break
            funcs[fn] = _ensure_function_name(fn_code, fn, placeholder="tuso_model")
        if ok:
            return funcs

    return None


def _dm_load_history_records_pool(
    history_path: str,
    ordered_fn_names: Sequence[str],
) -> List["ModelRecord"]:
    path = Path(history_path)
    if not path.exists():
        return []
    entries = _dm_read_history_entries(path)

    out: List[ModelRecord] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        code = e.get("code")
        acc = e.get("accuracy")
        if not isinstance(code, str):
            continue
        try:
            acc_f = float(acc)
        except Exception:
            continue
        runtime = e.get("runtime")
        try:
            runtime_f = float(runtime) if runtime is not None else float("inf")
        except Exception:
            runtime_f = float("inf")

        blocks = _dm_extract_target_blocks_with_support(code, ordered_fn_names)
        functions: Dict[str, str] = {}
        ok = True
        for fn in ordered_fn_names:
            fn_code = blocks.get(fn) or _dm_extract_function_block_with_leading_support(code, fn)
            if not fn_code:
                ok = False
                break
            functions[fn] = _ensure_function_name(fn_code, fn, placeholder="tuso_model")
        if not ok:
            continue

        out.append(
            ModelRecord(
                code=code,
                file=Path(str(e.get("file", ""))) if e.get("file") else None,
                accuracy=acc_f,
                runtime=runtime_f,
                model_info=e.get("model_info"),
                lineage=str(e.get("lineage", "")),
                functions=functions,
            )
        )
    return out



def _dm_record_key(record: "ModelRecord") -> str:
    code = str(getattr(record, "code", "") or "")
    acc = getattr(record, "accuracy", None)
    rt = getattr(record, "runtime", None)
    return hashlib.sha256(f"{acc!r}|{rt!r}|{code}".encode("utf-8", errors="ignore")).hexdigest()


def _dm_sync_shared_history_records(
    history_path: Union[str, Path],
    ordered_fn_names: Sequence[str],
    global_models: List["ModelRecord"],
    history: Dict[int, Dict[int, List["ModelRecord"]]],
    *,
    step: int,
) -> Tuple[int, int]:
    """Pull newly written candidates from a shared history file into local search state."""
    restored = _dm_load_history_records_pool(str(history_path), ordered_fn_names)
    existing = {_dm_record_key(m) for m in global_models if m and getattr(m, "code", None)}
    added = 0
    for rec in restored:
        key = _dm_record_key(rec)
        if key in existing:
            continue
        global_models.append(rec)
        history.setdefault(-1, {}).setdefault(step, []).append(rec)
        existing.add(key)
        added += 1
    return added, len(restored)

def _dm_get_selected_history_summary(
    history_path: str,
    *,
    accuracy_tolerance: float = 0.0,
) -> Optional[Dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return None

    entries = _dm_read_history_entries(path)

    candidates: List[ModelRecord] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        code = e.get("code")
        acc = e.get("accuracy")
        if not isinstance(code, str):
            continue
        try:
            acc_f = float(acc)
        except Exception:
            continue
        runtime = e.get("runtime")
        try:
            runtime_f = float(runtime) if runtime is not None else float("inf")
        except Exception:
            runtime_f = float("inf")
        candidates.append(ModelRecord(code=code, file=None, accuracy=acc_f, runtime=runtime_f))

    if not candidates:
        return None

    close = _dm_history_close_set(candidates, min_improvement=max(0.0, float(accuracy_tolerance or 0.0)))
    top_accuracy_model = max(close, key=lambda m: float(m.accuracy))
    selected = max(close, key=lambda m: _dm_history_complexity_score(m, top_accuracy_model))
    return {
        "history_count": int(len(candidates)),
        "best_accuracy": float(max(m.accuracy for m in candidates)),
        "accuracy": float(selected.accuracy),
        "runtime": float(selected.runtime),
        "code_len_lines": int(_dm_code_len_lines(selected.code)),
        "selection_score": float(_dm_history_complexity_score(selected, top_accuracy_model)),
    }


def _dm_snapshot_dynamic_state(
    task_probs: Dict[str, float],
    category_probs: Dict[str, float],
    feedbacks: Dict[str, Dict[str, List[str]]],
    task_by_name: Dict[str, Tuple[str, "Task"]],
) -> Dict[str, Any]:
    feedbacks_plain: Dict[str, Dict[str, List[str]]] = {}
    for fn, by_key in (feedbacks or {}).items():
        feedbacks_plain[fn] = {k: list(v or []) for k, v in (by_key or {}).items()}
    task_kt_probs: Dict[str, Dict[str, float]] = {}
    for fn, (_, task) in task_by_name.items():
        task_kt_probs[fn] = dict(getattr(task, "kt_prob", None) or {})
    return {
        "task_probs": dict(task_probs or {}),
        "category_probs": dict(category_probs or {}),
        "feedbacks": feedbacks_plain,
        "task_kt_probs": task_kt_probs,
    }


def _dm_load_dynamic_state_from_history(history_path: str) -> Optional[Dict[str, Any]]:
    path = Path(history_path)
    if not path.exists():
        return None
    entries = _dm_read_history_entries(path)
    if not isinstance(entries, list):
        return None
    for e in reversed(entries):
        if not isinstance(e, dict):
            continue
        if all(k in e for k in ("task_probs", "category_probs", "feedbacks", "task_kt_probs")):
            return {
                "task_probs": dict(e.get("task_probs") or {}),
                "category_probs": dict(e.get("category_probs") or {}),
                "feedbacks": dict(e.get("feedbacks") or {}),
                "task_kt_probs": dict(e.get("task_kt_probs") or {}),
            }
    return None


def _dm_init_output_and_logs(
    folder: str,
    history_name: Optional[str] = None,
    *,
    multi_machine: bool = False,
) -> Tuple[Path, _DMLogSinks]:
    run_id = f"{int(time.time())}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
    base_path = Path(folder) / (f"code_{run_id}" if multi_machine else "code")
    base_path.mkdir(parents=True, exist_ok=True)
    for p in base_path.iterdir():
        if p.is_file():
            p.unlink()
        else:
            rmtree_portable(p)

    history_root = Path(folder) / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    if history_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(history_name).strip()).strip("_")
        folder_name = f"history_{safe_name or run_id}"
    else:
        folder_name = f"history_{run_id}"

    history_dir = history_root / folder_name
    if history_dir.exists() and not multi_machine:
        history_dir = history_root / f"{folder_name}_{run_id}"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "history.json"
    dev_path = history_dir / (f"dev_{run_id}.json" if multi_machine else "dev.json")
    prompt_io_path = history_dir / (f"prompt_io_{run_id}.json" if multi_machine else "prompt_io.json")

    if not history_path.exists():
        history_path.write_text("[]", encoding="utf-8")
    dev_path.write_text("[]", encoding="utf-8")
    prompt_io_path.write_text("[]", encoding="utf-8")

    existing_history = _dm_read_history_entries(history_path) if multi_machine else []
    return base_path, _DMLogSinks(
        history_path=history_path,
        dev_path=dev_path,
        prompt_io_path=prompt_io_path,
        history_log=existing_history,
        multi_machine=multi_machine,
        run_id=run_id,
    )


@dataclass
class _DMEvalContext:
    base_path: Path
    reference_filename: str
    ordered_fn_names: List[str]
    function_sources: Dict[str, Dict[str, Any]]
    repo_snapshots: Dict[str, Path]
    timeout: int
    val_limit: Any
    sensitive_data: bool
    printer: _DMPrinter
    logs: _DMLogSinks


def _dm_collect_function_sources(
    method_tasks: List["MethodTask"],
    data_tasks: List["DataTask"],
    reference_filename: str,
) -> Dict[str, Dict[str, Any]]:
    def _resolve_source_path(raw_source: str) -> Path:
        candidate = Path(raw_source)
        if candidate.exists() or candidate.suffix == ".py" or "/" in raw_source or "\\" in raw_source:
            return candidate
        spec = importlib.util.find_spec(raw_source)
        if spec and spec.origin and spec.origin not in {"built-in", "frozen"}:
            return Path(spec.origin)
        return candidate

    def _infer_package_copy_context(abs_path: Path) -> Tuple[Optional[Path], Optional[str]]:
        if not abs_path.exists() or abs_path.suffix != ".py":
            return None, None
        package_dir = abs_path.parent
        if not (package_dir / "__init__.py").exists():
            return None, None
        while (package_dir.parent / "__init__.py").exists():
            package_dir = package_dir.parent
        package_parent = package_dir.parent
        try:
            local_rel = str(abs_path.relative_to(package_parent))
        except ValueError:
            return None, None
        return package_dir, local_rel

    def _resolve_repo_source(repo_root_path: Path, raw_source: str) -> Tuple[Path, str]:
        repo_root_abs = repo_root_path.resolve()
        src = _resolve_source_path(raw_source)

        if src.is_absolute():
            abs_path = src
        else:
            # Prefer explicit path under repo root.
            direct = repo_root_abs / src
            if direct.exists():
                abs_path = direct
            else:
                # If caller passed e.g. "repo_root/foo.py", avoid doubling by
                # stripping the repo_root prefix first.
                try:
                    rel_from_repo_root = Path(raw_source).relative_to(repo_root_path)
                    abs_path = repo_root_abs / rel_from_repo_root
                except ValueError:
                    abs_path = direct

        try:
            repo_rel_path = str(abs_path.resolve().relative_to(repo_root_abs))
        except ValueError:
            repo_rel_path = str(Path(raw_source))

        return abs_path, repo_rel_path

    def _resolve_local_source(ref_file: Path, raw_source: str) -> Path:
        src = _resolve_source_path(raw_source)
        if src.is_absolute():
            return src

        cwd_candidate = Path(raw_source)
        if cwd_candidate.exists():
            return cwd_candidate

        # Avoid duplicating ref parent prefix when source already contains it.
        ref_parent = ref_file.parent
        try:
            rel_from_parent = Path(raw_source).relative_to(ref_parent)
            return ref_parent / rel_from_parent
        except ValueError:
            return ref_parent / src

    sources: Dict[str, Dict[str, Any]] = {}
    ref_path = Path(reference_filename)
    ref_resolved = ref_path.resolve()
    for task in list(method_tasks) + list(data_tasks):
        source_path = task.source_path or reference_filename
        repo_root = task.repo_root
        if repo_root and not task.source_path:
            raise ValueError(f"repo_root set but source_path missing for function '{task.function_name}'")
        if repo_root:
            repo_root_path = Path(repo_root)
            abs_path, rel_path = _resolve_repo_source(repo_root_path, source_path)
            is_reference_file = abs_path.resolve() == ref_resolved
            sources[task.function_name] = {
                "file_path": str(abs_path),
                "repo_root": str(repo_root_path),
                "repo_rel_path": rel_path,
                "local_rel_path": None,
                "is_reference_file": is_reference_file,
                "package_copy_dir": None,
            }
        else:
            abs_path = _resolve_local_source(ref_path, source_path)
            try:
                local_rel_path = str(abs_path.relative_to(ref_path.parent))
                package_copy_dir = None
            except ValueError:
                package_copy_dir, inferred_rel = _infer_package_copy_context(abs_path)
                local_rel_path = inferred_rel or abs_path.name
            if abs_path.resolve() == ref_path.resolve():
                local_rel_path = None
            is_reference_file = abs_path.resolve() == ref_resolved
            sources[task.function_name] = {
                "file_path": str(abs_path),
                "repo_root": None,
                "repo_rel_path": None,
                "local_rel_path": local_rel_path,
                "is_reference_file": is_reference_file,
                "package_copy_dir": str(package_copy_dir) if package_copy_dir else None,
            }
    return sources


def _dm_init_repo_snapshots(
    function_sources: Dict[str, Dict[str, Any]],
    base_path: Path,
) -> Dict[str, Path]:
    repo_roots = sorted({src["repo_root"] for src in function_sources.values() if src.get("repo_root")})
    snapshots: Dict[str, Path] = {}
    if not repo_roots:
        return snapshots
    snapshot_root = base_path / "repo_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for repo_root in repo_roots:
        repo_path = Path(str(repo_root))
        if not repo_path.exists():
            raise FileNotFoundError(f"repo_root not found: {repo_root}")
        repo_key = hashlib.sha1(str(repo_path).encode("utf-8")).hexdigest()[:8]
        snapshot_path = snapshot_root / f"{repo_path.name}_{repo_key}"
        copytree_portable(repo_path, snapshot_path)
        snapshots[str(repo_root)] = snapshot_path
    return snapshots


def _dm_write_dynamic_repo_import_guard(
    base_path: Path,
    safe_tag: str,
    repo_workspaces: Dict[str, Path],
) -> Optional[Path]:
    """Create a sitecustomize import guard for repo-root optimization runs.

    The runner should import edited modules from the copied dynamic repo workspace.
    If a runner hard-codes the original repo path into sys.path, imports can silently
    resolve to the unedited source tree. This guard fails immediately when Python
    tries to import modules from those original repo roots.
    """
    if not repo_workspaces:
        return None

    forbidden_roots = [str(Path(root).resolve()) for root in repo_workspaces]
    allowed_roots = [str(path.resolve()) for path in repo_workspaces.values()]
    guard_dir = base_path / f"{safe_tag}_import_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = guard_dir / "sitecustomize.py"
    guard_code = r"""
import importlib.machinery
from pathlib import Path

_FORBIDDEN_ROOTS = [Path(p).resolve() for p in __FORBIDDEN_ROOTS__]
_ALLOWED_ROOTS = [Path(p).resolve() for p in __ALLOWED_ROOTS__]
_ORIGINAL_FIND_SPEC = importlib.machinery.PathFinder.find_spec


def _under(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except Exception:
        return False


def _check_import_path(fullname, raw_path):
    if not raw_path or raw_path in ("built-in", "frozen"):
        return
    try:
        path = Path(raw_path).resolve()
    except Exception:
        return
    if any(_under(path, allowed) for allowed in _ALLOWED_ROOTS):
        return
    for forbidden in _FORBIDDEN_ROOTS:
        if _under(path, forbidden):
            expected = ", ".join(str(p) for p in _ALLOWED_ROOTS)
            raise ImportError(
                "TusoAI dynamic repository import check failed: "
                f"module '{fullname}' resolved to '{path}', which is inside the original repo root '{forbidden}'. "
                f"Expected imports to resolve from the dynamic edited repo workspace(s): {expected}. "
                "The runner script is executed normally, so this guard validates the runner's imports as well as the optimized code's imports. "
                "This usually means the runner hard-codes the method repository path or inserts it ahead of TusoAI's dynamic workspace on sys.path. "
                "Update the runner to import the method repository through paths relative to the runner, not through an absolute hard-coded path."
            )


def _guarded_find_spec(cls, fullname, path=None, target=None):
    spec = _ORIGINAL_FIND_SPEC(fullname, path, target)
    if spec is None:
        return None
    _check_import_path(fullname, getattr(spec, "origin", None))
    for location in getattr(spec, "submodule_search_locations", None) or []:
        _check_import_path(fullname, location)
    return spec


importlib.machinery.PathFinder.find_spec = classmethod(_guarded_find_spec)
"""
    guard_code = guard_code.replace("__FORBIDDEN_ROOTS__", json.dumps(forbidden_roots))
    guard_code = guard_code.replace("__ALLOWED_ROOTS__", json.dumps(allowed_roots))
    sitecustomize.write_text(guard_code, encoding="utf-8")
    return guard_dir


def _dm_build_eval_python_paths(
    reference_filename: str,
    repo_workspaces: Dict[str, Path],
    import_guard_dir: Optional[Path] = None,
) -> List[str]:
    def _path_is_or_contains(path: Path, other: Path) -> bool:
        try:
            other.relative_to(path)
            return True
        except ValueError:
            return False

    python_paths: List[str] = []
    if import_guard_dir is not None:
        python_paths.append(str(import_guard_dir))
    for path in sorted(repo_workspaces.values(), key=lambda p: str(p)):
        for candidate in (str(path), str(path.parent)):
            if candidate not in python_paths:
                python_paths.append(candidate)
    ref_parent_path = Path(reference_filename).resolve().parent
    # Avoid exposing original repo roots through the runner's original directory.
    # Namespace packages can merge the dynamic copy with the original repo if both
    # parents are on sys.path, causing the import guard to see the forbidden
    # original location even though the dynamic workspace is listed first.
    exposes_original_repo = any(
        _path_is_or_contains(ref_parent_path, Path(repo_root).resolve())
        for repo_root in repo_workspaces
    )
    ref_parent = str(ref_parent_path)
    if not exposes_original_repo and ref_parent not in python_paths:
        python_paths.append(ref_parent)
    return python_paths


def _dm_prepare_eval_workspace(
    base_path: Path,
    reference_filename: str,
    repo_snapshots: Dict[str, Path],
    function_sources: Dict[str, Dict[str, Any]],
    safe_tag: str,
) -> Tuple[Path, Dict[str, Path]]:
    eval_path = base_path / f"{safe_tag}.py"
    ref_path = Path(reference_filename).resolve()
    copied_package_dirs: set[str] = set()
    for source in function_sources.values():
        if source.get("repo_root"):
            continue
        package_copy_dir = source.get("package_copy_dir")
        if package_copy_dir:
            package_dir = Path(package_copy_dir)
            if package_copy_dir not in copied_package_dirs and package_dir.exists():
                copytree_portable(package_dir, base_path / package_dir.name, dirs_exist_ok=True)
                copied_package_dirs.add(package_copy_dir)
        local_rel = source.get("local_rel_path")
        if not local_rel:
            continue
        src_path = Path(source["file_path"])
        dst_path = base_path / local_rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile_portable(src_path, dst_path)
    repo_workspaces: Dict[str, Path] = {}
    for repo_root, snapshot_path in repo_snapshots.items():
        repo_name = Path(str(repo_root)).name
        repo_key = hashlib.sha1(str(repo_root).encode("utf-8")).hexdigest()[:8]
        workspace_parent = base_path / f"{safe_tag}_repo_{repo_key}"
        workspace = workspace_parent / repo_name
        workspace.parent.mkdir(parents=True, exist_ok=True)
        copytree_portable(snapshot_path, workspace)
        repo_workspaces[str(repo_root)] = workspace
        try:
            ref_rel_path = ref_path.relative_to(Path(str(repo_root)).resolve())
        except ValueError:
            continue
        workspace_runner = workspace / ref_rel_path
        if workspace_runner.exists():
            eval_path = workspace_runner
    if eval_path == base_path / f"{safe_tag}.py":
        copyfile_portable(ref_path, eval_path)
    return eval_path, repo_workspaces


def _dm_apply_function_updates(
    functions: Dict[str, str],
    ordered_fn_names: Sequence[str],
    function_sources: Dict[str, Dict[str, Any]],
    eval_path: Path,
    repo_workspaces: Dict[str, Path],
    capture_function_prints: bool = True,
) -> None:
    for fn in ordered_fn_names:
        source = function_sources[fn]
        fn_code = _ensure_function_name(functions[fn], fn, placeholder="tuso_model")
        if capture_function_prints:
            fn_code = _instrument_function_print_capture(fn_code, fn)
        if source.get("is_reference_file"):
            target_path = eval_path
            replace_functions(target_path, [fn], fn_code)
            continue
        repo_root = source.get("repo_root")
        if repo_root:
            rel_path = source.get("repo_rel_path")
            if not rel_path:
                raise ValueError(f"repo_root set but repo_rel_path missing for fn '{fn}'")
            target_path = repo_workspaces[str(repo_root)] / rel_path
        else:
            local_rel = source.get("local_rel_path")
            if local_rel:
                target_path = eval_path.parent / local_rel
            else:
                target_path = eval_path
        replace_functions(target_path, [fn], fn_code)


def _dm_eval_bundle_with_sources(
    functions: Dict[str, str],
    *,
    lineage: str,
    base_path: Path,
    reference_filename: str,
    ordered_fn_names: Sequence[str],
    function_sources: Dict[str, Dict[str, Any]],
    repo_snapshots: Dict[str, Path],
    timeout: int,
    val_limit: Any,
    sensitive_data: bool = False,
    gpu_id: Optional[int] = None,
) -> Tuple[Union["ModelRecord", str], Optional[dict]]:
    safe_tag = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    eval_path, repo_workspaces = _dm_prepare_eval_workspace(
        base_path=base_path,
        reference_filename=reference_filename,
        repo_snapshots=repo_snapshots,
        function_sources=function_sources,
        safe_tag=safe_tag,
    )
    try:
        _dm_apply_function_updates(
            functions=functions,
            ordered_fn_names=ordered_fn_names,
            function_sources=function_sources,
            eval_path=eval_path,
            repo_workspaces=repo_workspaces,
            capture_function_prints=not sensitive_data,
        )
        import_guard_dir = _dm_write_dynamic_repo_import_guard(base_path, safe_tag, repo_workspaces)
        python_paths = _dm_build_eval_python_paths(reference_filename, repo_workspaces, import_guard_dir)
        metrics = run_and_evaluate(
            eval_path,
            timeout,
            val_limit,
            python_paths=python_paths,
            target_functions=ordered_fn_names,
            sensitive_data=sensitive_data,
            gpu_id=gpu_id,
        )
        if not isinstance(metrics, dict):
            return str(metrics), None

        bundle = _bundle_functions_text(functions, ordered_fn_names)
        return ModelRecord(
            code=bundle,
            file=eval_path,
            accuracy=metrics["evaluation"],
            runtime=metrics.get("runtime"),
            model_info=metrics.get("model_info"),
            lineage=lineage,
            functions=dict(functions),
        ), metrics
    finally:
        try:
            if eval_path.exists():
                eval_path.unlink()
        except Exception:
            pass
        for ws in repo_workspaces.values():
            try:
                rmtree_portable(ws.parent)
            except Exception:
                pass
        import_guard_path = base_path / f"{safe_tag}_import_guard"
        try:
            if import_guard_path.exists():
                rmtree_portable(import_guard_path)
        except Exception:
            pass


def _dm_write_and_evaluate(
    ctx: _DMEvalContext,
    functions: Dict[str, str],
    tag: str,
    *,
    edited_fn: Optional[str] = None,
    dev_context: Optional[dict] = None,
) -> Union["ModelRecord", str]:
    rec_or_err, metrics = _dm_eval_bundle_with_sources(
        functions,
        lineage=tag,
        base_path=ctx.base_path,
        reference_filename=ctx.reference_filename,
        ordered_fn_names=ctx.ordered_fn_names,
        function_sources=ctx.function_sources,
        repo_snapshots=ctx.repo_snapshots,
        timeout=ctx.timeout,
        val_limit=ctx.val_limit,
        sensitive_data=getattr(ctx, "sensitive_data", False),
    )
    if not isinstance(rec_or_err, ModelRecord):
        return str(rec_or_err)
    rec = rec_or_err
    bundle = _bundle_functions_text(functions, ctx.ordered_fn_names)

    if dev_context is not None:
        ctx2 = dict(dev_context)
        ctx2.update(
            {
                "eval_file": str(rec.file),
                "eval_metrics": metrics,
                "edited_fn": edited_fn,
                "bundle_lines": sum(1 for ln in bundle.splitlines() if ln.strip()),
            }
        )
        ctx.logs.log_dev(ctx2)
        ctx.printer.debug_dump("dev_log", dict(ctx2))

    return rec


def _dm_hint_block(global_hints: List[str], task_hints: Optional[List[str]]) -> str:
    all_hints = list(global_hints) + list(task_hints or [])
    return "\n- ".join(all_hints) if all_hints else ""


def _dm_data_context_block(kind: str, task: "Task") -> str:
    if kind != "data":
        return ""
    dt: "DataTask" = task  # type: ignore
    return (
        "<data_context>\n"
        "Here is an overview of the file:\n"
        f"{dt.file_description}\n"
        f"{dt.data_summary}\n\n"
        f"Use this exact read command to load the file: {dt.read_cmd}\n"
        f"Your goal is to {dt.data_usage} using this data.\n"
        "</data_context>\n"
    )


# -------------------------------------------
# Prompt builders: seeding
# -------------------------------------------

def _dm_seed_base_prompt(
    *,
    task_description: str,
    init_idea: str,
    hint_block: str,
    base_fn_code: str,
    fn: str,
) -> str:
    return build_seed_base_prompt(
        task_description=task_description,
        init_idea=init_idea,
        hint_block=hint_block,
        base_fn_code=base_fn_code,
        fn=fn,
    )


def _dm_seed_repair_prompt(
    *,
    task_description: str,
    error_msg: str,
    suggestion: str,
    fn: str,
) -> str:
    return build_seed_repair_prompt(
        task_description=task_description,
        error_msg=error_msg,
        suggestion=suggestion,
        fn=fn,
    )


# -------------------------------
# NEW seeding function (parallel + no duplicate evals)
# - Dispatches jobs immediately for:
#   (a) baseline/original bundle (if any task uses initial)
#   (b) every initial_solutions entry for every method task
# - Each job does: prompt -> fix loop -> extract -> evaluate (single pass per attempt)
# - Builds an eval_cache keyed by full bundle code so combo stage can reuse results
# -------------------------------

def _dm_seed_method_variant_pools(
    *,
    method_tasks: List["MethodTask"],
    base_functions: Dict[str, str],
    global_hints: List[str],
    task_description: str,
    bug_retries: int,
    skip_timeout: bool,
    filename: str,
    state: _DMRunState,
    printer: _DMPrinter,
    logs: _DMLogSinks,
    eval_ctx: _DMEvalContext,
    attempts_by_fn: Dict[str, set],
    n_jobs: int = 1,
    gpu_ids: Optional[List[int]] = None,
    eval_cache: Optional[Dict[str, "ModelRecord"]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], Optional[List["ModelRecord"]], Dict[str, "ModelRecord"]]:
    import hashlib
    try:
        import multiprocess as mp  # type: ignore
    except Exception:  # pragma: no cover
        import multiprocessing as mp  # type: ignore

    n_jobs = max(1, int(n_jobs))
    eval_cache = eval_cache or {}

    method_variants: Dict[str, List[Dict[str, Any]]] = {}
    method_fns = [mt.function_name for mt in method_tasks]
    single_method_task = (len(method_tasks) == 1)
    single_task_records: Optional[List["ModelRecord"]] = [] if single_method_task else None

    printer.subhdr("SEEDING | build variant pools") if hasattr(printer, "subhdr") else None
    printer.brief("seed", f"start | method_tasks={len(method_tasks)} | n_jobs={n_jobs} | bug_retries={bug_retries} | skip_timeout={skip_timeout}")

    # ---- add originals to pools (bookkeeping only)
    any_use_initial = False
    for mt in method_tasks:
        fn = mt.function_name
        method_variants[fn] = []
        if mt.use_initial:
            any_use_initial = True
            method_variants[fn].append({"label": "original", "code": base_functions[fn], "source": "base"})
            printer.brief("seed_pool", f"fn={fn} added original baseline variant (not LLM-generated)")

    # Evaluation params (avoid pickling eval_ctx)
    eval_params = {
        "base_path": eval_ctx.base_path,
        "reference_filename": str(eval_ctx.reference_filename),
        "ordered_fn_names": list(eval_ctx.ordered_fn_names),
        "function_sources": dict(eval_ctx.function_sources),
        "repo_snapshots": dict(eval_ctx.repo_snapshots),
        "timeout": int(eval_ctx.timeout),
        "val_limit": eval_ctx.val_limit,
        "sensitive_data": bool(getattr(eval_ctx, "sensitive_data", False)),
    }

    def _bundle_key(functions: Dict[str, str]) -> str:
        h = hashlib.sha1()
        for fn in eval_params["ordered_fn_names"]:
            code = _ensure_function_name(functions[fn], fn, placeholder="tuso_model")
            h.update(fn.encode("utf-8"))
            h.update(b"\0")
            h.update(code.encode("utf-8"))
            h.update(b"\0\0")
        return h.hexdigest()

    def _eval_bundle(functions: Dict[str, str], lineage: str, gpu_id: Optional[int] = None) -> Union["ModelRecord", str]:
        rec_or_err, _ = _dm_eval_bundle_with_sources(
            functions,
            lineage=lineage,
            base_path=eval_params["base_path"],
            reference_filename=eval_params["reference_filename"],
            ordered_fn_names=eval_params["ordered_fn_names"],
            function_sources=eval_params["function_sources"],
            repo_snapshots=eval_params["repo_snapshots"],
            timeout=eval_params["timeout"],
            val_limit=eval_params["val_limit"],
            gpu_id=gpu_id,
        )
        return rec_or_err

    # ---- build seed jobs
    jobs: List[Dict[str, Any]] = []
    if any_use_initial:
        gpu_cycle = list(gpu_ids or [])
        jobs.append(
            {
                "job_type": "baseline",
                "lineage": f"{filename}seed_baseline",
                "gpu_id": (gpu_cycle[0] if gpu_cycle else None),
            }
        )

    llm_job_count = 0
    for mt in method_tasks:
        fn = mt.function_name
        for init_idx, init in enumerate(mt.initial_solutions or []):
            hint_block = _dm_hint_block(global_hints, mt.hints or [])
            base_prompt = _dm_seed_base_prompt(
                task_description=task_description,
                init_idea=init,
                hint_block=hint_block,
                base_fn_code=base_functions[fn],
                fn=fn,
            )
            jobs.append(
                {
                    "job_type": "llm_init",
                    "fn": fn,
                    "init": init,
                    "init_idx": init_idx,
                    "base_prompt": base_prompt,
                    "attempts_snapshot": set(attempts_by_fn.get(fn, set())),
                    "lineage": f"{filename}{fn}_{init_idx}_{re.sub(r'[^a-zA-Z0-9]+', '_', init)[:20]}",
                    "gpu_id": None,
                }
            )
            llm_job_count += 1

    printer.brief(
        "seed_dispatch",
        f"jobs_total={len(jobs)} (baseline_jobs={1 if any_use_initial else 0}, llm_jobs={llm_job_count}) | parallelism=n_jobs={n_jobs}",
    )
    if any_use_initial:
        baseline_job = next((j for j in jobs if j.get("job_type") == "baseline"), None)
        if baseline_job is not None:
            printer.brief("seed_dispatch", f"baseline_gpu={baseline_job.get('gpu_id')}")
    if llm_job_count:
        per_fn = {mt.function_name: len(mt.initial_solutions or []) for mt in method_tasks}
        printer.brief("seed_dispatch", f"llm_jobs_by_fn={per_fn}")

    # ---- worker does single pipeline (prompt -> repair loop -> extract -> eval)
    def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
        local_cost = 0.0
        dev_entries: List[dict] = []
        attempted_codes: List[str] = []

        if job["job_type"] == "baseline":
            funcs = dict(base_functions)
            rec_or_err = _eval_bundle(funcs, lineage=job["lineage"], gpu_id=job.get("gpu_id"))
            return {
                "job_type": "baseline",
                "ok": isinstance(rec_or_err, ModelRecord),
                "record": rec_or_err if isinstance(rec_or_err, ModelRecord) else None,
                "error": None if isinstance(rec_or_err, ModelRecord) else rec_or_err,
                "local_cost": local_cost,
                "dev_entries": dev_entries,
                "attempted_codes": attempted_codes,
            }

        fn = job["fn"]
        init = job["init"]
        base_prompt = job["base_prompt"]
        attempts_snapshot: set = job["attempts_snapshot"]
        lineage = job["lineage"]

        suggestion = ""
        error_msg = ""

        for try_idx in range(bug_retries):
            prompt = base_prompt if not error_msg else _dm_seed_repair_prompt(
                task_description=task_description,
                error_msg=error_msg,
                suggestion=suggestion,
                fn=fn,
            )

            if error_msg == "Error: timed out" and skip_timeout:
                break

            reply, cost = run_prompt(prompt)
            local_cost += float(cost)
            logs.log_prompt_io(
                {
                    "stage": "seed_llm_prompt",
                    "function": fn,
                    "init": init,
                    "try_idx": try_idx,
                    "prompt": prompt,
                    "reply": reply,
                    "cost": float(cost or 0.0),
                }
            )

            dev_entries.append(
                {
                    "stage": "seed_llm",
                    "function": fn,
                    "init": init,
                    "try_idx": try_idx,
                    "error_prev": error_msg,
                    "reply_len": len(reply or ""),
                }
            )

            if not reply:
                error_msg = "LLM empty reply"
                continue

            suggestion = _extract_code_from_reply(reply)
            suggestion = _ensure_function_name(suggestion, fn, placeholder="tuso_model")
            if not suggestion:
                error_msg = "extraction failure"
                continue

            if suggestion in attempts_snapshot:
                error_msg = "duplicate or extraction failure"
                break

            attempts_snapshot.add(suggestion)
            attempted_codes.append(suggestion)

            funcs = dict(base_functions)
            funcs[fn] = suggestion

            rec_or_err = _eval_bundle(funcs, lineage=lineage, gpu_id=job.get("gpu_id"))
            dev_entries.append(
                {
                    "stage": "seed_eval",
                    "function": fn,
                    "init": init,
                    "try_idx": try_idx,
                    "lineage": lineage,
                    "eval_ok": isinstance(rec_or_err, ModelRecord),
                    "eval_err": None if isinstance(rec_or_err, ModelRecord) else str(rec_or_err),
                    "eval_acc": float(rec_or_err.accuracy) if isinstance(rec_or_err, ModelRecord) else None,
                    "eval_rt": float(rec_or_err.runtime) if (isinstance(rec_or_err, ModelRecord) and rec_or_err.runtime is not None) else None,
                }
            )

            if isinstance(rec_or_err, ModelRecord):
                return {
                    "job_type": "llm_init",
                    "ok": True,
                    "fn": fn,
                    "init": init,
                    "suggestion": suggestion,
                    "record": rec_or_err,
                    "error": None,
                    "local_cost": local_cost,
                    "dev_entries": dev_entries,
                    "attempted_codes": attempted_codes,
                    "lineage": lineage,
                }

            error_msg = str(rec_or_err)

        return {
            "job_type": "llm_init",
            "ok": False,
            "fn": fn,
            "init": init,
            "suggestion": suggestion,
            "record": None,
            "error": error_msg or "retries exhausted",
            "local_cost": local_cost,
            "dev_entries": dev_entries,
            "attempted_codes": attempted_codes,
            "lineage": lineage,
        }

    baseline_rec: Optional["ModelRecord"] = None
    ok_llm = 0
    fail_llm = 0

    def _handle_result(r: Dict[str, Any]) -> None:
        nonlocal baseline_rec, ok_llm, fail_llm

        state.add_cost(float(r.get("local_cost", 0.0)))

        # burn attempted codes globally (even on failures)
        if r.get("job_type") == "llm_init":
            fn = r.get("fn")
            if fn:
                for code in r.get("attempted_codes", []) or []:
                    attempts_by_fn.setdefault(fn, set()).add(code)

        # print high-level per-attempt progress (DEFAULT ON)
        for e in r.get("dev_entries", []) or []:
            if e.get("stage") == "seed_llm":
                printer.brief(
                    "seed_llm",
                    f"fn={e['function']} init='{e['init']}' try={e['try_idx']} reply_len={e['reply_len']}",
                )
            elif e.get("stage") == "seed_eval":
                if e.get("eval_ok"):
                    printer.brief(
                        "seed_eval",
                        f"OK fn={e['function']} init='{e['init']}' try={e['try_idx']} acc={e['eval_acc']:.6f} rt={e['eval_rt'] if e['eval_rt'] is not None else 'NA'}",
                    )
                else:
                    err_msg = e.get("eval_err")
                    if state.debug and err_msg:
                        printer.brief(
                            "seed_eval",
                            f"FAIL fn={e['function']} init='{e['init']}' try={e['try_idx']} err={err_msg}",
                        )
                    else:
                        printer.brief(
                            "seed_eval",
                            f"FAIL fn={e['function']} init='{e['init']}' try={e['try_idx']}",
                        )

        # still write dev log (as before)
        for entry in r.get("dev_entries", []) or []:
            ee = dict(entry)
            ee["total_cost"] = float(state.total_cost)
            logs.log_dev(ee)

        if r.get("job_type") == "baseline":
            if r.get("ok") and r.get("record"):
                baseline_rec = r["record"]
                eval_cache[_bundle_key(baseline_rec.functions)] = baseline_rec
                printer.brief(
                    "seed_baseline",
                    f"OK acc={baseline_rec.accuracy:.6f} rt={baseline_rec.runtime if baseline_rec.runtime is not None else 'NA'}",
                )
            else:
                baseline_err = str(r.get("error") or "unknown error")
                if state.debug:
                    printer.brief("seed_baseline", f"FAIL err={baseline_err}")
                else:
                    printer.brief("seed_baseline", "FAIL")
                raise RuntimeError(f"Initial baseline seed failed: {baseline_err}")
            return

        # llm init job
        if r.get("ok") and r.get("record"):
            rec: ModelRecord = r["record"]
            fn = r["fn"]
            init = r["init"]

            eval_cache[_bundle_key(rec.functions)] = rec

            # avoid duplicate pool insert
            already = any(v.get("code") == r.get("suggestion") for v in method_variants.get(fn, []))
            if already:
                printer.brief("seed_pool", f"SKIP dup fn={fn} init='{init}'")
                return

            method_variants[fn].append(
                {
                    "label": f"init:{init}",
                    "code": r["suggestion"],
                    "source": "llm",
                    "seed_single_accuracy": rec.accuracy,
                    "seed_single_runtime": rec.runtime,
                    "tag": r.get("lineage"),
                }
            )
            ok_llm += 1
            printer.brief(
                "seed_pool",
                f"kept init fn={fn} init='{init}' acc={rec.accuracy:.6f} rt={rec.runtime if rec.runtime is not None else 'NA'} "
                f"(pool={len(method_variants[fn])})",
            )

            if single_task_records is not None:
                single_task_records.append(rec)
        else:
            fail_llm += 1
            init_err = str(r.get("error") or "unknown error")
            if state.debug:
                printer.brief("seed_pool", f"FAIL init fn={r.get('fn')} init='{r.get('init')}' err={init_err}")
            else:
                printer.brief("seed_pool", f"FAIL init fn={r.get('fn')} init='{r.get('init')}'")

    # run baseline first (if required) so failures are surfaced immediately
    llm_jobs = [j for j in jobs if j.get("job_type") != "baseline"]
    if any_use_initial:
        baseline_job = next((j for j in jobs if j.get("job_type") == "baseline"), None)
        if baseline_job is not None:
            _handle_result(_worker(baseline_job))

    # ---- run jobs in parallel and stream results
    if n_jobs > 1 and llm_jobs:
        printer.brief("seed_run", f"running jobs in parallel (n_jobs={n_jobs})")
        # Use threads here so the local _worker closure remains callable without
        # multiprocessing pickling constraints.
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for r in ex.map(_worker, llm_jobs):
                _handle_result(r)
    else:
        printer.brief("seed_run", "running jobs sequentially (n_jobs=1)")
        for j in llm_jobs:
            _handle_result(_worker(j))

    # ---- attach baseline eval to each "original" entry (if baseline exists)
    if baseline_rec is not None:
        for mt in method_tasks:
            if mt.use_initial:
                fn = mt.function_name
                # by convention original is first entry if present
                for v in method_variants.get(fn, []):
                    if v.get("label") == "original":
                        v["seed_single_accuracy"] = baseline_rec.accuracy
                        v["seed_single_runtime"] = baseline_rec.runtime
                        v["tag"] = baseline_rec.lineage
                        printer.brief(
                            "seed_original",
                            f"fn={fn} original baseline acc={baseline_rec.accuracy:.6f} rt={baseline_rec.runtime if baseline_rec.runtime is not None else 'NA'}",
                        )
                        break

        if single_task_records is not None and baseline_rec not in single_task_records:
            single_task_records.insert(0, baseline_rec)

    # ---- validate per-task pools are non-empty (unchanged behavior)
    for mt in method_tasks:
        fn = mt.function_name
        if len(method_variants.get(fn, [])) == 0:
            raise RuntimeError(f"No successful initial variants for method task '{fn}'.")

    # ---- summary prints
    printer.brief("seed_summary", f"llm_jobs_ok={ok_llm} llm_jobs_fail={fail_llm} | total_cost={state.money(state.total_cost)}")
    printer.brief(
        "seed_summary",
        "pools=" + ", ".join(f"{fn}:{len(vs)}" for fn, vs in method_variants.items()),
    )

    return method_variants, method_fns, single_task_records, eval_cache



def _dm_cluster_models_by_code(
    models: List["ModelRecord"],
    n_clusters: int,
) -> Dict[int, List["ModelRecord"]]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [m.code for m in models]
    X = TfidfVectorizer(stop_words="english").fit_transform(texts)
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

    clustered: Dict[int, List[ModelRecord]] = defaultdict(list)
    for m, lab in zip(models, labels):
        clustered[int(lab)].append(m)
    return clustered


def _dm_evaluate_seed_combinations(
    *,
    method_variants: Dict[str, List[Dict[str, Any]]],
    method_fns: List[str],
    base_functions: Dict[str, str],
    max_islands: Optional[int],
    min_improvement: float,
    state: _DMRunState,
    printer: _DMPrinter,
    eval_ctx: _DMEvalContext,
) -> List["ModelRecord"]:
    # Evaluate all combinations across method tasks (data tasks remain baseline)
    variant_lists = [method_variants[fn] for fn in method_fns]
    total_combos = 1
    for lst in variant_lists:
        total_combos *= max(1, len(lst))

    printer.brief("seed_combo", f"evaluating combinations: {total_combos} total")

    combo_records: List[ModelRecord] = []
    combo_idx = 0

    for combo in itertools.product(*variant_lists):
        combo_idx += 1
        funcs = dict(base_functions)  # includes data task baselines too

        selected: Dict[str, str] = {}
        for fn, variant in zip(method_fns, combo):
            funcs[fn] = variant["code"]
            selected[fn] = variant.get("label", "unknown")

        tag = f"seed_combo_{combo_idx}"
        rec = _dm_write_and_evaluate(
            eval_ctx,
            funcs,
            tag,
            edited_fn=None,
            dev_context={
                "stage": "seed_combo_eval",
                "combo_idx": combo_idx,
                "combo_total": total_combos,
                "selected": dict(selected),
                "max_islands": max_islands,
                "total_cost": float(state.total_cost),
            },
        )

        if isinstance(rec, ModelRecord):
            combo_records.append(rec)
            printer.brief(
                "seed_combo",
                f"OK {combo_idx}/{total_combos} acc={rec.accuracy:.6f} rt={rec.runtime:.3f}s selected={selected}",
            )
        else:
            printer.brief("seed_combo", f"FAIL {combo_idx}/{total_combos} selected={selected} err={rec}")

    if not combo_records:
        raise RuntimeError("No valid initial combinations - all combinations failed when run together.")

    printer.brief("seed_combo", f"successful combos={len(combo_records)}")

    # Optionally cluster combos to max_islands and pick champions
    if max_islands is None or len(combo_records) <= max_islands:
        seed_records = combo_records
        printer.brief("seed_select", f"using all combos as islands (count={len(seed_records)})")
        return seed_records

    k = min(max_islands, len(combo_records))
    clustered = _dm_cluster_models_by_code(combo_records, k)

    printer.brief("seed_select", f"clustered combos into k={k} (max_islands={max_islands})")

    if state.debug:
        printer.debug_dump(
            "seed_cluster_summary",
            {"stage": "seed_cluster_summary", "k": k, "cluster_sizes": {cid: len(ms) for cid, ms in clustered.items()}},
        )

    seed_records = []
    for cid, models in clustered.items():
        champ = _pick_cluster_champion(models, min_improvement=min_improvement)
        seed_records.append(champ)
        printer.brief("seed_select", f"cluster={cid} champ_acc={champ.accuracy:.6f} rt={champ.runtime:.3f}s")

    return seed_records


def _dm_create_islands_from_seed(
    *,
    seed_records: List["ModelRecord"],
    max_islands: int,
    logs: _DMLogSinks,
    printer: _DMPrinter,
    state: _DMRunState,
) -> Tuple[List["Island"], Dict[int, Dict[int, List["ModelRecord"]]], List["ModelRecord"]]:
    if max_islands <= 0:
        raise ValueError(f"max_islands must be >= 1, got {max_islands}")

    # If too many, keep only the top max_islands by accuracy.
    records: List["ModelRecord"] = list(seed_records)
    if len(records) > max_islands:
        records = sorted(
            records,
            key=lambda r: float(getattr(r, "accuracy", float("-inf"))),
            reverse=True,
        )[:max_islands]

    # If too few, duplicate the best-performing island/model until we hit max_islands.
    if records and len(records) < max_islands:
        best = max(records, key=lambda r: float(getattr(r, "accuracy", float("-inf"))))
        while len(records) < max_islands:
            records.append(copy.deepcopy(best))

    islands: List["Island"] = []
    history: Dict[int, Dict[int, List["ModelRecord"]]] = {}
    global_models: List["ModelRecord"] = []

    for isl_id, rec in enumerate(records):
        islands.append(Island(id=isl_id, models=[rec]))
        history.setdefault(isl_id, {})[0] = [rec]
        global_models.append(rec)
        logs.log_history(
            {
                "stage": "seed_combo",
                "island": isl_id,
                "accuracy": rec.accuracy,
                "runtime": rec.runtime,
                "code": rec.code,
                "total_cost": float(state.total_cost),
            }
        )

    printer.brief("seed", f"done starting_islands={len(islands)} (from combinations)")
    return islands, history, global_models



def _dm_choose_weighted_key(probs: Dict[str, float]) -> str:
    names = list(probs.keys())
    weights = [probs[n] for n in names]
    return random.choices(names, weights=weights, k=1)[0]


# -------------------------------------------
# Prompt builders: mutation
# -------------------------------------------

def _dm_prompt_body_for_category(
    *,
    category: str,
    task: "Task",
    prompt_samples: int,
    diagnostic_prompts: List[str],
    ablation_prompts: List[str],
    simplify_prompts: List[str],
    feedbacks: Dict[str, Dict[str, List[str]]],
    fn_name: str,
    n_feedback_buffer: int,
) -> Tuple[str, Optional[str], str]:
    """
    Returns: (prompt_body, ptype_used, feedback_key)
    """
    ptype_used: Optional[str] = None
    feedback_key: str = ""

    if category == "prompt":
        if not task.kt_prob:
            return "", None, ""  # caller will treat as error, matching original behavior path

        ptype_used = random.choices(list(task.kt_prob), weights=list(task.kt_prob.values()), k=1)[0]
        feedback_key = ptype_used

        sampled_prompts = random.sample(
            task.knowledge_tree[ptype_used],
            k=min(prompt_samples, len(task.knowledge_tree[ptype_used])),
        )
        prompt_options = "\n\n".join(f"Option {i+1}:\n{p}" for i, p in enumerate(sampled_prompts))

        prompt_body = f"""
<strategy_task>
Select the most promising 1-2 options below and implement them directly.
Prefer changes that plausibly improve the evaluation score, not cosmetic refactors.
Avoid adding feature flags (no new booleans/default-false gating).
</strategy_task>

<options>
{prompt_options}
</options>
""".strip()

    elif category == "ablation":
        feedback_key = "ablation"
        sampled = random.sample(ablation_prompts, min(prompt_samples, len(ablation_prompts)))
        alter_str = "\n\n".join(f"Option {i+1}:\n{p}" for i, p in enumerate(sampled))

        prompt_body = f"""
<strategy_task>
Perform a SINGLE-FACTOR ablation to learn what drives performance.
Change exactly ONE meaningful component; keep everything else as similar as possible.

Add print() statements that reveal method behavior and will be useful for the next optimization generation.
The goal is insight, not winning this run.
</strategy_task>

<options>
{alter_str}
</options>

<print_requirements>
- Print >10 lines total, but keep each line concise
- Use labeled, parseable lines, e.g.:
  ABL: change=<...>
  ABL: key_stat=<...>
  ABL: failure_mode=<...>
  ABL: next_hypothesis=<...>
</print_requirements>
""".strip()

    elif category == "simplify":
        feedback_key = "simplify"
        sampled = random.sample(simplify_prompts, min(prompt_samples, len(simplify_prompts)))
        alter_str = "\n\n".join(f"Option {i+1}:\n{p}" for i, p in enumerate(sampled))

        prompt_body = f"""
<strategy_task>
Improve the method by reducing runtime/complexity while preserving or improving evaluation score.
Prioritize principled simplifications, vectorization, and removing non-essential work.
</strategy_task>

<options>
{alter_str}
</options>
""".strip()

    elif category == "ablation_2":
        feedback_key = "ablation"
        prompt_body = """
<strategy_task>
Use the accumulated ablation insights/feedback to implement a targeted performance improvement.
Remove ALL print() statements from the final code.
Do not introduce feature flags that disable the improvement by default.
</strategy_task>
""".strip()

    elif category == "diagnostic":
        feedback_key = ""  # no feedback for diagnostic pass
        sampled = random.sample(diagnostic_prompts, min(prompt_samples, len(diagnostic_prompts)))
        alter_str = "\n\n".join(f"Option {i+1}:\n{p}" for i, p in enumerate(sampled))

        prompt_body = f"""
<strategy_task>
Add diagnostic instrumentation to expose method internals and failure modes so that the NEXT generation of optimizations can be better targeted.
The primary goal is actionable insight.
</strategy_task>

<options>
{alter_str}
</options>

<print_requirements>
- Print very comprehensive information (>10 lines)
- Make prints directly useful for follow-up optimization decisions:
  DIAG: assumptions=<...>
  DIAG: key_stats=<...>
  DIAG: sensitivity=<...>
  DIAG: top_errors=<...>
  DIAG: next_actions=<...>
</print_requirements>
""".strip()

    elif category == "diagnostic_2":
        feedback_key = "diagnostic_2"
        diag_info = getattr(task, "model_info", None)  # not used; actual info comes from src_record in caller
        prompt_body = ""  # filled by caller (needs src_record.model_info)

    else:
        return "", None, ""  # unknown; caller handles

    # append recent feedback (structured)
    if feedback_key:
        recent_fb = (feedbacks.get(fn_name, {}).get(feedback_key, []) or [])[-n_feedback_buffer:]
        if recent_fb:
            prompt_body += "\n\n<prior_feedback>\n" + "\n".join(recent_fb) + "\n</prior_feedback>"


    return prompt_body, ptype_used, feedback_key


def _dm_mutation_base_prompt(
    *,
    task_description: str,
    fn_name: str,
    kind: str,
    ctx_block: str,
    prompt_body: str,
    hint_block: str,
    src_code: str,
) -> str:
    return build_mutation_base_prompt(
        task_description=task_description,
        fn_name=fn_name,
        kind=kind,
        ctx_block=ctx_block,
        prompt_body=prompt_body,
        hint_block=hint_block,
        src_code=src_code,
    )


def _dm_mutation_repair_prompt(
    *,
    task_description: str,
    category: str,
    error_msg: str,
    suggestion: str,
    src_code: str,
    hint_block: str,
    fn_name: str,
) -> str:
    return build_mutation_repair_prompt(
        task_description=task_description,
        category=category,
        error_msg=error_msg,
        suggestion=suggestion,
        src_code=src_code,
        hint_block=hint_block,
        fn_name=fn_name,
    )


@dataclass
class _DMMutationContext:
    task_by_name: Dict[str, Tuple[str, "Task"]]
    task_probs: Dict[str, float]
    category_probs: Dict[str, float]
    feedbacks: Dict[str, Dict[str, List[str]]]
    attempts_by_fn: Dict[str, set]
    global_hints: List[str]
    task_description: str

    prompt_samples: int
    diagnostic_prompts: List[str]
    ablation_prompts: List[str]
    simplify_prompts: List[str]

    timeout: int
    bug_retries: int
    n_feedback_buffer: int
    skip_timeout: bool
    prompt_decay: float
    prompt_importance: float
    min_improvement: float

    eval_ctx: _DMEvalContext
    printer: _DMPrinter
    logs: _DMLogSinks
    state: _DMRunState


def _dm_single_pass_mutation(
    *,
    mctx: _DMMutationContext,
    src_record: "ModelRecord",
    root_accuracy: float,
    fn_name: str,
    kind: str,
    task: "Task",
    category: str,
    gen: int,
    island_id: int,
) -> Tuple[Optional["ModelRecord"], str, Optional[str]]:
    ptype_used: Optional[str] = None
    feedback_key: str = ""

    # init kt_prob if missing (same behavior)
    if not task.kt_prob:
        keys = list(task.knowledge_tree.keys())
        if keys:
            task.kt_prob = {k: 1.0 for k in keys}
            update_probabilities(task.kt_prob, keys[0], 1.0)

    hint_block = _dm_hint_block(mctx.global_hints, task.hints or [])
    ctx_block = _dm_data_context_block(kind, task)
    src_code = src_record.functions[fn_name]

    # Build prompt_body (category-specific)
    prompt_body, ptype_used, feedback_key = _dm_prompt_body_for_category(
        category=category,
        task=task,
        prompt_samples=mctx.prompt_samples,
        diagnostic_prompts=mctx.diagnostic_prompts,
        ablation_prompts=mctx.ablation_prompts,
        simplify_prompts=mctx.simplify_prompts,
        feedbacks=mctx.feedbacks,
        fn_name=fn_name,
        n_feedback_buffer=mctx.n_feedback_buffer,
    )
    print("FEEDBACK KEY:", feedback_key)
    
    if category == "diagnostic_2":
        feedback_key = "diagnostic_2"
        diag_info = _dm_extract_model_info_for_function(src_record.model_info, fn_name)
        prompt_body = f"""
<strategy_task>
Based on the diagnostic information below, implement a targeted improvement that is most likely to increase evaluation score.
Remove ALL print() statements from the final code.
Avoid feature flags / default-false toggles.
</strategy_task>

<diagnostic_info>
{diag_info}
</diagnostic_info>
""".strip()

    if category == "prompt" and not prompt_body:
        return None, "kt_prob missing for prompt category", ptype_used
    if category not in {"prompt", "ablation", "ablation_2", "diagnostic", "diagnostic_2", "simplify"}:
        return None, f"unknown category: {category}", ptype_used

    base_prompt = _dm_mutation_base_prompt(
        task_description=mctx.task_description,
        fn_name=fn_name,
        kind=kind,
        ctx_block=ctx_block,
        prompt_body=prompt_body,
        hint_block=hint_block,
        src_code=src_code,
    )

    suggestion: str = ""
    error_msg: str = ""

    mctx.printer.brief(
        "mutate",
        f"gen={gen} isl={island_id} fn={fn_name} kind={kind} cat={category} ptype={ptype_used}",
    )

    for retry_idx in range(mctx.bug_retries):
        final_prompt = base_prompt if not error_msg else _dm_mutation_repair_prompt(
            task_description=mctx.task_description,
            category=category,
            error_msg=error_msg,
            suggestion=suggestion,
            src_code=src_code,
            hint_block=hint_block,
            fn_name=fn_name,
        )

        if error_msg == "Error: timed out" and mctx.skip_timeout:
            mctx.printer.brief("mutate", f"skip_timeout gen={gen} isl={island_id} fn={fn_name}")
            return None, error_msg, ptype_used

        reply, cost = run_prompt(final_prompt)
        mctx.state.add_cost(cost)
        mctx.logs.log_prompt_io(
            {
                "stage": "mutate_llm_prompt",
                "gen": gen,
                "island": island_id,
                "function": fn_name,
                "category": category,
                "retry_idx": retry_idx,
                "prompt": final_prompt,
                "reply": reply,
                "cost": float(cost or 0.0),
            }
        )

        dev_entry = {
            "stage": "mutate_llm",
            "gen": gen,
            "island": island_id,
            "function": fn_name,
            "task_kind": kind,
            "category": category,
            "ptype": ptype_used,
            "retry_idx": retry_idx,
            "prompt": final_prompt,
            "reply_raw": reply,
            "error_prev": error_msg,
            "task_probs": dict(mctx.task_probs),
            "category_probs": dict(mctx.category_probs) if mctx.category_probs else None,
            "task_kt_prob": dict(task.kt_prob),
            "total_cost": float(mctx.state.total_cost),
        }
        mctx.logs.log_dev(dict(dev_entry))
        mctx.printer.debug_dump("mutate_llm", dict(dev_entry))

        if not reply:
            error_msg = "LLM failed"
            mctx.printer.brief("mutate", f"LLM empty gen={gen} isl={island_id} fn={fn_name} retry={retry_idx}")
            continue

        suggestion = _extract_code_from_reply(reply)
        inline_feedback = _extract_feedback_from_reply(reply)
        suggestion = _ensure_function_name(suggestion, fn_name, placeholder="tuso_model")

        if mctx.state.debug:
            mctx.printer.debug_dump(
                "mutate_extract",
                {
                    "stage": "mutate_extract",
                    "gen": gen,
                    "island": island_id,
                    "function": fn_name,
                    "retry_idx": retry_idx,
                    "extracted_code": suggestion,
                    "inline_feedback": inline_feedback,
                },
            )

        if not suggestion or suggestion in mctx.attempts_by_fn[fn_name]:
            mctx.printer.brief("mutate", f"dup/extract-fail gen={gen} isl={island_id} fn={fn_name}")
            return None, "duplicate or extraction failure", ptype_used
        mctx.attempts_by_fn[fn_name].add(suggestion)

        new_functions = dict(src_record.functions)
        new_functions[fn_name] = suggestion
        tag = "".join(random.choices(string.ascii_letters + string.digits, k=20))

        task_probs_before = dict(mctx.task_probs)
        cat_probs_before = dict(mctx.category_probs) if mctx.category_probs else {}
        kt_probs_before = dict(task.kt_prob)

        cand = _dm_write_and_evaluate(
            mctx.eval_ctx,
            new_functions,
            tag,
            edited_fn=fn_name,
            dev_context={
                "stage": "mutate_eval",
                "gen": gen,
                "island": island_id,
                "function": fn_name,
                "task_kind": kind,
                "category": category,
                "ptype": ptype_used,
                "tag": tag,
                "prompt": final_prompt,
                "reply_raw": reply,
                "extracted_code": suggestion,
                "inline_feedback": inline_feedback,
                "src_accuracy": src_record.accuracy,
                "root_accuracy": root_accuracy,
                "task_probs_before": task_probs_before,
                "category_probs_before": cat_probs_before,
                "task_kt_prob_before": kt_probs_before,
                "total_cost": float(mctx.state.total_cost),
            },
        )

        if not isinstance(cand, ModelRecord):
            error_msg = cand
            mctx.printer.brief("mutate", f"eval-fail gen={gen} isl={island_id} fn={fn_name} err={error_msg}")
            if retry_idx == mctx.bug_retries - 1 and mctx.prompt_decay != 1.0 and category == "prompt" and ptype_used:
                update_probabilities(task.kt_prob, ptype_used, 1.0 / mctx.prompt_decay)
                if mctx.state.debug:
                    mctx.printer.debug_dump(
                        "prob_update_on_fail",
                        {
                            "stage": "prob_update_on_fail",
                            "gen": gen,
                            "island": island_id,
                            "function": fn_name,
                            "ptype": ptype_used,
                            "task_kt_prob_after": dict(task.kt_prob),
                        },
                    )
            continue

        improved_vs_src = cand.accuracy > src_record.accuracy + mctx.min_improvement
        kt_factor = mctx.prompt_importance if improved_vs_src else (1.0 / mctx.prompt_decay)
        if category == "prompt" and ptype_used:
            update_probabilities(task.kt_prob, ptype_used, kt_factor)

        improved_vs_root = cand.accuracy > root_accuracy + mctx.min_improvement
        factor = mctx.prompt_importance if improved_vs_root else (1.0 / mctx.prompt_decay)
        update_probabilities(mctx.task_probs, fn_name, factor)

        delta = cand.accuracy - src_record.accuracy
        mctx.printer.brief(
            "mutate",
            f"OK gen={gen} isl={island_id} fn={fn_name} cat={category} acc={cand.accuracy:.6f} ({delta:+.6f}) rt={cand.runtime:.3f}s",
        )

        if mctx.state.debug:
            mctx.printer.debug_dump(
                "prob_updates",
                {
                    "stage": "prob_updates",
                    "gen": gen,
                    "island": island_id,
                    "function": fn_name,
                    "category": category,
                    "ptype": ptype_used,
                    "improved_vs_src": improved_vs_src,
                    "improved_vs_root": improved_vs_root,
                    "task_probs_before": task_probs_before,
                    "task_probs_after": dict(mctx.task_probs),
                    "category_probs_before": cat_probs_before,
                    "category_probs_after": dict(mctx.category_probs) if mctx.category_probs else None,
                    "task_kt_prob_before": kt_probs_before,
                    "task_kt_prob_after": dict(task.kt_prob),
                },
            )

        if feedback_key and inline_feedback:
            fb = _merge_feedback_with_outcome(
                inline_feedback,
                new_perf=cand.accuracy,
                old_perf=src_record.accuracy,
            )
            if fn_name not in mctx.feedbacks:
                mctx.feedbacks[fn_name] = {}
            mctx.feedbacks[fn_name][feedback_key].append(fb)

            fb_entry = {
                "stage": "feedback",
                "gen": gen,
                "island": island_id,
                "function": fn_name,
                "task_kind": kind,
                "category": category,
                "ptype": ptype_used,
                "feedback_key": feedback_key,
                "feedback": fb,
                "total_cost": float(mctx.state.total_cost),
            }
            mctx.logs.log_dev(dict(fb_entry))
            mctx.printer.debug_dump("feedback", dict(fb_entry))

        return cand, "", ptype_used

    return None, error_msg or "retries exhausted", ptype_used


def _dm_make_child(
    *,
    mctx: _DMMutationContext,
    parent: "ModelRecord",
    gen: int,
    island_id: int,
) -> Optional["ModelRecord"]:
    fn_name = _dm_choose_weighted_key(mctx.task_probs)
    kind, task = mctx.task_by_name[fn_name]

    primary_category = _dm_choose_weighted_key(mctx.category_probs) if mctx.category_probs else None
    if not primary_category:
        raise RuntimeError("paradigm_probs (category_probs) must be provided and non-empty.")

    root_acc = parent.accuracy

    child, err, _ = _dm_single_pass_mutation(
        mctx=mctx,
        src_record=parent,
        root_accuracy=root_acc,
        fn_name=fn_name,
        kind=kind,
        task=task,
        category=primary_category,
        gen=gen,
        island_id=island_id,
    )
    if child is None:
        mctx.printer.brief("mutate", f"FAIL gen={gen} isl={island_id} fn={fn_name} cat={primary_category} err={err}")
        if mctx.state.debug:
            mctx.printer.debug_dump(
                "mutate_fail",
                {
                    "stage": "mutate_fail",
                    "gen": gen,
                    "island": island_id,
                    "function": fn_name,
                    "category": primary_category,
                    "error": err,
                },
            )
        return None

    if primary_category in {"diagnostic", "ablation"}:
        second = "diagnostic_2" if primary_category == "diagnostic" else "ablation_2"
        mctx.printer.brief("mutate", f"second_pass gen={gen} isl={island_id} fn={fn_name} {primary_category} -> {second}")
        child2, _, _ = _dm_single_pass_mutation(
            mctx=mctx,
            src_record=child,
            root_accuracy=root_acc,
            fn_name=fn_name,
            kind=kind,
            task=task,
            category=second,
            gen=gen,
            island_id=island_id,
        )
        if child2 is not None:
            child = child2

    return child


def _dm_cluster_select_islands(
    *,
    global_models: List["ModelRecord"],
    target_clusters: int,
    min_improvement: float,
    debug: bool,
    printer: _DMPrinter,
) -> List["Island"]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [m.code for m in global_models if m and m.code]
    models_for_texts = [m for m in global_models if m and m.code]

    printer.brief("cluster", f"candidates={len(texts)} target_clusters={target_clusters}")

    if not (len(texts) >= target_clusters and target_clusters >= 1):
        return []

    X = TfidfVectorizer(stop_words="english").fit_transform(texts)
    labels = KMeans(n_clusters=target_clusters, random_state=42).fit_predict(X)

    clustered: Dict[int, List[ModelRecord]] = defaultdict(list)
    for m, lab in zip(models_for_texts, labels):
        clustered[int(lab)].append(m)

    if debug:
        printer.debug_dump(
            "cluster_summary",
            {
                "stage": "cluster_summary",
                "target_clusters": target_clusters,
                "cluster_sizes": {cid: len(ms) for cid, ms in clustered.items()},
            },
        )

    new_islands: List[Island] = []
    for new_id, models in enumerate(clustered.values()):
        if not models:
            continue
        champ = _pick_cluster_champion(models, min_improvement=min_improvement)
        new_islands.append(Island(id=new_id, models=[champ]))
        printer.brief("cluster_pick", f"cluster={new_id} champ_acc={champ.accuracy:.6f} rt={champ.runtime:.3f}s")

    return new_islands

# -------------------------------
# Parallel combo evaluator with cache reuse
# - Only runs bundles not already in eval_cache
# - Reuses cached ModelRecords for identical bundles (no re-run)
# -------------------------------

def _dm_evaluate_seed_combinations_parallel(
    *,
    method_variants: Dict[str, List[Dict[str, Any]]],
    method_fns: List[str],
    base_functions: Dict[str, str],
    max_islands: Optional[int],
    min_improvement: float,
    state: _DMRunState,
    printer: _DMPrinter,
    logs: _DMLogSinks,
    eval_ctx: _DMEvalContext,
    n_jobs: int,
    eval_cache: Dict[str, "ModelRecord"],
) -> List["ModelRecord"]:
    import hashlib

    n_jobs = max(1, int(n_jobs))

    eval_params = {
        "base_path": eval_ctx.base_path,
        "reference_filename": str(eval_ctx.reference_filename),
        "ordered_fn_names": list(eval_ctx.ordered_fn_names),
        "function_sources": dict(eval_ctx.function_sources),
        "repo_snapshots": dict(eval_ctx.repo_snapshots),
        "timeout": int(eval_ctx.timeout),
        "val_limit": eval_ctx.val_limit,
        "sensitive_data": bool(getattr(eval_ctx, "sensitive_data", False)),
    }

    def _bundle_key(functions: Dict[str, str]) -> str:
        h = hashlib.sha1()
        for fn in eval_params["ordered_fn_names"]:
            code = _ensure_function_name(functions[fn], fn, placeholder="tuso_model")
            h.update(fn.encode("utf-8"))
            h.update(b"\0")
            h.update(code.encode("utf-8"))
            h.update(b"\0\0")
        return h.hexdigest()

    def _eval_bundle(functions: Dict[str, str], lineage: str) -> Union["ModelRecord", str]:
        rec_or_err, _ = _dm_eval_bundle_with_sources(
            functions,
            lineage=lineage,
            base_path=eval_params["base_path"],
            reference_filename=eval_params["reference_filename"],
            ordered_fn_names=eval_params["ordered_fn_names"],
            function_sources=eval_params["function_sources"],
            repo_snapshots=eval_params["repo_snapshots"],
            timeout=eval_params["timeout"],
            val_limit=eval_params["val_limit"],
        )
        return rec_or_err

    # Build all combos; split into cached vs to-evaluate
    variant_lists = [method_variants[fn] for fn in method_fns]
    total_combos = 1
    for lst in variant_lists:
        total_combos *= max(1, len(lst))

    printer.brief("seed_combo", f"evaluating combinations: {total_combos} total (n_jobs={n_jobs})")

    to_eval_jobs: List[Dict[str, Any]] = []
    combo_records: List["ModelRecord"] = []

    combo_idx = 0
    for combo in itertools.product(*variant_lists):
        combo_idx += 1
        funcs = dict(base_functions)

        selected: Dict[str, str] = {}
        for fn, variant in zip(method_fns, combo):
            funcs[fn] = variant["code"]
            selected[fn] = variant.get("label", "unknown")

        tag = f"seed_combo_{combo_idx}"
        key = _bundle_key(funcs)

        if key in eval_cache:
            # reuse without re-running (clone lineage for traceability)
            cached = eval_cache[key]
            combo_records.append(
                ModelRecord(
                    code=cached.code,
                    file=cached.file,
                    accuracy=cached.accuracy,
                    runtime=cached.runtime,
                    model_info=cached.model_info,
                    lineage=tag,
                    functions=dict(cached.functions),
                )
            )
            continue

        to_eval_jobs.append(
            {
                "funcs": funcs,
                "tag": tag,
                "combo_idx": combo_idx,
                "combo_total": total_combos,
                "selected": selected,
                "key": key,
            }
        )

    if to_eval_jobs:
        printer.brief("seed_combo", f"dispatching {len(to_eval_jobs)} combo jobs (cached={len(combo_records)})")
    else:
        printer.brief("seed_combo", f"all combos satisfied by cache (cached={len(combo_records)})")
        return combo_records

    def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
        # no LLM here; single evaluation attempt
        rec_or_err = _eval_bundle(job["funcs"], lineage=job["tag"])
        return {
            "ok": isinstance(rec_or_err, ModelRecord),
            "record": rec_or_err if isinstance(rec_or_err, ModelRecord) else None,
            "error": None if isinstance(rec_or_err, ModelRecord) else rec_or_err,
            "tag": job["tag"],
            "key": job["key"],
            "combo_idx": job["combo_idx"],
            "combo_total": job["combo_total"],
            "selected": job["selected"],
            "dev_entry": {
                "stage": "seed_combo_eval",
                "combo_idx": job["combo_idx"],
                "combo_total": job["combo_total"],
                "selected": dict(job["selected"]),
                "max_islands": max_islands,
            },
        }

    results: List[Dict[str, Any]] = []
    if n_jobs > 1:
        # Use threads so local closures stay callable without pickling.
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            for r in ex.map(_worker, to_eval_jobs):
                results.append(r)
    else:
        for j in to_eval_jobs:
            results.append(_worker(j))

    for r in results:
        e = dict(r["dev_entry"])
        e["total_cost"] = float(state.total_cost)  # combo eval has no LLM cost here
        logs.log_dev(e)
        printer.debug_dump("seed_combo_eval", e)

        if r["ok"] and r["record"]:
            rec: ModelRecord = r["record"]
            eval_cache[r["key"]] = rec
            combo_records.append(rec)
            printer.brief(
                "seed_combo",
                f"OK {r['combo_idx']}/{r['combo_total']} acc={rec.accuracy:.6f} rt={rec.runtime:.3f}s selected={r['selected']}",
            )
        else:
            printer.brief("seed_combo", f"FAIL {r['combo_idx']}/{r['combo_total']} selected={r['selected']} err={r['error']}")

    if not combo_records:
        raise RuntimeError("No valid initial combinations - all combinations failed when run together.")

    printer.brief("seed_combo", f"successful combos={len(combo_records)}")
    return combo_records

# ============================================================
# NEW helper: one parallel "job" for continuous evolution
# - Rebuild islands + champions from ALL candidates (clustering)
# - Sample an island at random, take its champion
# - Run the full mutation pipeline (prompt->repair->extract->eval [+ optional 2nd pass])
# - Return the new child + metadata + dev log entries + prob update instructions
# ============================================================

def _dm_evolution_worker_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker does:
      - pick (cluster_id, parent) from precomputed champions
      - sample fn/category
      - run mutation pipeline (prompt -> extract -> eval), with retries
      - return candidate + prob updates + feedback items

    Expects in job:
      - champions: List[Tuple[int, ModelRecord]]   # computed in parent via sklearn
      - task_specs: Dict[str, Dict[str, Any]]      # picklable task metadata
      - task_probs, category_probs, feedbacks_snapshot, attempts_snapshot, etc.
      - eval params: reference_filename, base_path, ordered_fn_names, function_sources, repo_snapshots, timeout, val_limit
    """
    import os
    import time
    import shutil
    import string
    import random
    import traceback
    from pathlib import Path
    from collections import defaultdict
    from types import SimpleNamespace
    import hashlib

    try:
        random.seed(int(job.get("seed", 0)))
        pid = os.getpid()
        requested_gpu = job.get("gpu_id")
        if pid not in _GPU_STICKY_BY_PID:
            _GPU_STICKY_BY_PID[pid] = requested_gpu
        sticky_gpu_id = _GPU_STICKY_BY_PID.get(pid)

        champions = list(job.get("champions", []) or [])
        if not champions:
            return {
                "ok": False,
                "error": "no champions provided",
                "local_cost": 0.0,
                "dev_entries": [{"stage": "worker_init", "error": "no champions provided"}],
                "attempted_codes": {},
            }

        chosen_cluster_id, parent = random.choice(champions)

        # -------------------------
        # Eval helper (full bundle)
        # -------------------------
        base_path = Path(job["base_path"])
        ordered_fn_names = list(job["ordered_fn_names"])
        reference_filename = str(job["reference_filename"])
        function_sources = dict(job["function_sources"])
        repo_snapshots = {k: Path(v) for k, v in (job.get("repo_snapshots", {}) or {}).items()}
        timeout = int(job["timeout"])
        val_limit = job.get("val_limit", None)

        def _eval_full(functions: Dict[str, str], lineage: str) -> Union["ModelRecord", str]:
            safe_tag = f"{os.getpid()}_" + "".join(random.choices(string.ascii_letters + string.digits, k=18))
            eval_path, repo_workspaces = _dm_prepare_eval_workspace(
                base_path=base_path,
                reference_filename=reference_filename,
                repo_snapshots=repo_snapshots,
                function_sources=function_sources,
                safe_tag=safe_tag,
            )
            try:
                _dm_apply_function_updates(
                    functions=functions,
                    ordered_fn_names=ordered_fn_names,
                    function_sources=function_sources,
                    eval_path=eval_path,
                    repo_workspaces=repo_workspaces,
                    capture_function_prints=not bool(job.get("sensitive_data", False)),
                )
                import_guard_dir = _dm_write_dynamic_repo_import_guard(base_path, safe_tag, repo_workspaces)
                python_paths = _dm_build_eval_python_paths(reference_filename, repo_workspaces, import_guard_dir)
                metrics = run_and_evaluate(
                    eval_path,
                    timeout,
                    val_limit,
                    python_paths=python_paths,
                    target_functions=ordered_fn_names,
                    sensitive_data=bool(job.get("sensitive_data", False)),
                    gpu_id=sticky_gpu_id,
                )
                if not isinstance(metrics, dict):
                    return str(metrics)

                bundle = _bundle_functions_text(functions, ordered_fn_names)
                return ModelRecord(
                    code=bundle,
                    file=eval_path,
                    accuracy=metrics["evaluation"],
                    runtime=metrics.get("runtime"),
                    model_info=metrics.get("model_info"),
                    lineage=lineage,
                    functions=dict(functions),
                )
            finally:
                try:
                    if eval_path.exists():
                        eval_path.unlink()
                except Exception:
                    pass
                for ws in repo_workspaces.values():
                    try:
                        rmtree_portable(ws.parent)
                    except Exception:
                        pass
                import_guard_path = base_path / f"{safe_tag}_import_guard"
                try:
                    if import_guard_path.exists():
                        rmtree_portable(import_guard_path)
                except Exception:
                    pass

        # -------------------------
        # Choose task + category (same sampling)
        # -------------------------
        task_probs: Dict[str, float] = dict(job.get("task_probs", {}))
        category_probs: Dict[str, float] = dict(job.get("category_probs", {}))
        if not task_probs:
            return {"ok": False, "error": "empty task_probs", "local_cost": 0.0, "dev_entries": [], "attempted_codes": {}}
        if not category_probs:
            return {"ok": False, "error": "empty category_probs", "local_cost": 0.0, "dev_entries": [], "attempted_codes": {}}

        fn_names = list(task_probs.keys())
        fn_weights = [task_probs[n] for n in fn_names]
        fn_name = random.choices(fn_names, weights=fn_weights, k=1)[0]

        cats = list(category_probs.keys())
        cat_weights = [category_probs[c] for c in cats]
        primary_category = random.choices(cats, weights=cat_weights, k=1)[0]

        # -------------------------
        # Task specs -> Task-like shim (so your helpers work unchanged)
        # -------------------------
        task_specs: Dict[str, Dict[str, Any]] = job["task_specs"]
        if fn_name not in task_specs:
            return {"ok": False, "error": f"missing task_specs for fn={fn_name}", "local_cost": 0.0, "dev_entries": [], "attempted_codes": {}}

        spec = task_specs[fn_name]
        kind: str = spec.get("kind", "method")

        task = SimpleNamespace(
            hints=list(spec.get("hints", []) or []),
            knowledge_tree=dict(spec.get("knowledge_tree", {}) or {}),
            kt_prob=dict(spec.get("kt_prob", {}) or {}),
            file_description=spec.get("file_description", ""),
            data_summary=spec.get("data_summary", ""),
            data_usage=spec.get("data_usage", ""),
            read_cmd=spec.get("read_cmd", ""),
        )

        # -------------------------
        # Snapshots + params
        # -------------------------
        global_hints: List[str] = list(job.get("global_hints", []))
        task_description: str = str(job.get("task_description", ""))

        feedbacks_snapshot: Dict[str, Dict[str, List[str]]] = job.get("feedbacks_snapshot", {}) or {}
        attempts_snapshot: Dict[str, set] = job.get("attempts_snapshot", {}) or {}

        prompt_samples = int(job.get("prompt_samples", 5))
        diagnostic_prompts: List[str] = list(job.get("diagnostic_prompts", []))
        ablation_prompts: List[str] = list(job.get("ablation_prompts", []))
        simplify_prompts: List[str] = list(job.get("simplify_prompts", []))
        n_feedback_buffer = int(job.get("n_feedback_buffer", 5))

        bug_retries = int(job.get("bug_retries", 2))
        batch_queries = max(0, int(job.get("batch_queries", 0)))
        skip_timeout = bool(job.get("skip_timeout", False))
        prompt_decay = float(job.get("prompt_decay", 1.0))
        prompt_importance = float(job.get("prompt_importance", 1.0))
        min_improvement = float(job.get("min_improvement", 0.0))

        dev_entries: List[dict] = []
        local_cost: float = 0.0
        attempted_codes_by_fn: Dict[str, List[str]] = defaultdict(list)
        prompt_io_entries: List[dict] = []
        pending_batch_handles: Dict[str, Dict[str, Any]] = {}
        pending_batch_results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        pending_batch_context: Dict[str, Dict[str, Any]] = {}

        def _prompt_key(prompt: str, fn: str, cat: str, retry_idx: int) -> str:
            h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
            return f"{fn}|{cat}|{retry_idx}|{h}"

        def _poll_all_provider_batches() -> None:
            for k in list(pending_batch_handles.keys()):
                handle = pending_batch_handles.get(k)
                if handle is None:
                    continue
                try:
                    done, result_map = poll_default_prompt_batch(handle)
                except Exception as _e:
                    ctx = pending_batch_context.get(k, {})
                    dev_entries.append({
                        "stage": "provider_batch_poll_error",
                        "function": ctx.get("function"),
                        "category": ctx.get("category"),
                        "retry_idx": ctx.get("retry_idx"),
                        "error": str(_e),
                    })
                    pending_batch_handles.pop(k, None)
                    continue

                if not done:
                    continue

                custom_ids = list(handle.get("custom_ids", []) or [])
                for cid in custom_ids:
                    if cid in result_map:
                        pending_batch_results[k].append(result_map[cid])
                pending_batch_handles.pop(k, None)

        # init kt_prob worker-local if missing (same behavior)
        if not getattr(task, "kt_prob", None):
            keys = list(getattr(task, "knowledge_tree", {}).keys())
            if keys:
                task.kt_prob = {k: 1.0 for k in keys}
                update_probabilities(task.kt_prob, keys[0], 1.0)

        def _run_one_pass(
            src_record: "ModelRecord",
            category: str,
            root_accuracy: float,
        ) -> Tuple[Optional["ModelRecord"], str, Optional[str], str, List[dict], List[dict]]:
            nonlocal local_cost

            ptype_used: Optional[str] = None
            feedback_key: str = ""

            # uses your helper (no new prompts)
            prompt_body, ptype_used, feedback_key = _dm_prompt_body_for_category(
                category=category,
                task=task,
                prompt_samples=prompt_samples,
                diagnostic_prompts=diagnostic_prompts,
                ablation_prompts=ablation_prompts,
                simplify_prompts=simplify_prompts,
                feedbacks=feedbacks_snapshot,
                fn_name=fn_name,
                n_feedback_buffer=n_feedback_buffer,
            )

            # keep your diagnostic_2 block (as before)
            if category == "diagnostic_2":
                feedback_key = "diagnostic_2"
                diag_info = _dm_extract_model_info_for_function(src_record.model_info, fn_name)
                prompt_body = f"""
<strategy_task>
Based on the diagnostic information below, implement a targeted improvement that is most likely to increase evaluation score.
Remove ALL print() statements from the final code.
Avoid feature flags / default-false toggles.
</strategy_task>

<diagnostic_info>
{diag_info}
</diagnostic_info>
""".strip()

            if category == "prompt" and not getattr(task, "kt_prob", None):
                return None, "kt_prob missing for prompt category", ptype_used, feedback_key, [], []

            if category not in {"prompt", "ablation", "ablation_2", "diagnostic", "diagnostic_2", "simplify"}:
                return None, f"unknown category: {category}", ptype_used, feedback_key, [], []

            hint_block = _dm_hint_block(global_hints, getattr(task, "hints", None) or [])
            ctx_block = _dm_data_context_block(kind, task)
            src_code = src_record.functions[fn_name]

            base_prompt = _dm_mutation_base_prompt(
                task_description=task_description,
                fn_name=fn_name,
                kind=kind,
                ctx_block=ctx_block,
                prompt_body=prompt_body,
                hint_block=hint_block,
                src_code=src_code,
            )

            suggestion = ""
            error_msg = ""
            prob_updates: List[dict] = []
            feedback_items: List[dict] = []

            for retry_idx in range(bug_retries):
                final_prompt = base_prompt if not error_msg else _dm_mutation_repair_prompt(
                    task_description=task_description,
                    category=category,
                    error_msg=error_msg,
                    suggestion=suggestion,
                    src_code=src_code,
                    hint_block=hint_block,
                    fn_name=fn_name,
                )

                if error_msg == "Error: timed out" and skip_timeout:
                    return None, error_msg, ptype_used, feedback_key, prob_updates, feedback_items

                reply = ""
                cost = 0.0
                queue_key = _prompt_key(final_prompt, fn_name, category, retry_idx)
                used_provider_batch = False

                if batch_queries > 0:
                    # Opportunistically poll every pending provider batch whenever we are about
                    # to generate/evaluate a new optimization candidate (no separate receiver loop).
                    _poll_all_provider_batches()

                    if queue_key not in pending_batch_handles and not pending_batch_results.get(queue_key):
                        try:
                            pending_batch_handles[queue_key] = submit_default_prompt_batch([final_prompt] * batch_queries)
                            pending_batch_context[queue_key] = {
                                "function": fn_name,
                                "category": category,
                                "retry_idx": int(retry_idx),
                                "source_accuracy": float(src_record.accuracy),
                                "source_lineage": str(getattr(src_record, "lineage", "") or ""),
                                "source_code_sha1": hashlib.sha1((src_code or "").encode("utf-8")).hexdigest(),
                                "prompt_sha1": hashlib.sha1((final_prompt or "").encode("utf-8")).hexdigest(),
                            }
                            dev_entries.append({"stage": "provider_batch_submit", "function": fn_name, "category": category, "retry_idx": retry_idx, "batch_size": int(batch_queries)})
                        except Exception as _e:
                            dev_entries.append({"stage": "provider_batch_submit_error", "function": fn_name, "category": category, "retry_idx": retry_idx, "error": str(_e)})

                    if pending_batch_results.get(queue_key):
                        reply, cost = pending_batch_results[queue_key].pop(0)
                        used_provider_batch = True

                if not used_provider_batch:
                    reply, cost = run_prompt(final_prompt)

                local_cost += float(cost)
                prompt_io_entries.append(
                    {
                        "stage": "worker_mutate_llm_prompt",
                        "function": fn_name,
                        "task_kind": kind,
                        "category": category,
                        "retry_idx": retry_idx,
                        "prompt": final_prompt,
                        "reply": reply,
                        "cost": float(cost or 0.0),
                        "provider_batch": bool(used_provider_batch),
                        "provider_batch_context": dict(pending_batch_context.get(queue_key, {})),
                    }
                )

                dev_entries.append(
                    {
                        "stage": "mutate_llm",
                        "function": fn_name,
                        "task_kind": kind,
                        "category": category,
                        "ptype": ptype_used,
                        "retry_idx": retry_idx,
                        "error_prev": error_msg,
                        "reply_len": len(reply or ""),
                    }
                )

                if not reply:
                    error_msg = "LLM failed"
                    continue

                suggestion = _extract_code_from_reply(reply)
                inline_feedback = _extract_feedback_from_reply(reply)
                suggestion = _ensure_function_name(suggestion, fn_name, placeholder="tuso_model")
                if not suggestion:
                    error_msg = "extraction failure"
                    continue

                if suggestion in attempts_snapshot.get(fn_name, set()):
                    return None, "duplicate or extraction failure", ptype_used, feedback_key, prob_updates, feedback_items

                attempts_snapshot.setdefault(fn_name, set()).add(suggestion)
                attempted_codes_by_fn[fn_name].append(suggestion)

                new_functions = dict(src_record.functions)
                new_functions[fn_name] = suggestion
                tag = f"t{int(time.time())}_{os.getpid()}_" + "".join(random.choices(string.ascii_letters + string.digits, k=10))

                cand = _eval_full(new_functions, lineage=tag)

                dev_entries.append(
                    {
                        "stage": "mutate_eval",
                        "function": fn_name,
                        "task_kind": kind,
                        "category": category,
                        "ptype": ptype_used,
                        "tag": tag,
                        "eval_ok": isinstance(cand, ModelRecord),
                        "eval_err": None if isinstance(cand, ModelRecord) else str(cand),
                        "src_accuracy": float(src_record.accuracy),
                        "root_accuracy": float(root_accuracy),
                        "cand_accuracy": float(cand.accuracy) if isinstance(cand, ModelRecord) else None,
                    }
                )

                if not isinstance(cand, ModelRecord):
                    error_msg = str(cand)
                    if retry_idx == bug_retries - 1 and prompt_decay != 1.0 and category == "prompt" and ptype_used:
                        prob_updates.append({"type": "kt_prob", "fn": fn_name, "ptype": ptype_used, "factor": 1.0 / prompt_decay})
                    continue

                improved_vs_src = cand.accuracy > src_record.accuracy + min_improvement
                improved_vs_root = cand.accuracy > root_accuracy + min_improvement

                if category == "prompt" and ptype_used:
                    kt_factor = prompt_importance if improved_vs_src else (1.0 / prompt_decay)
                    prob_updates.append({"type": "kt_prob", "fn": fn_name, "ptype": ptype_used, "factor": kt_factor})

                task_factor = prompt_importance if improved_vs_root else (1.0 / prompt_decay)
                prob_updates.append({"type": "task_probs", "fn": fn_name, "factor": task_factor})

                if feedback_key and inline_feedback:
                    fb = _merge_feedback_with_outcome(
                        inline_feedback,
                        new_perf=cand.accuracy,
                        old_perf=src_record.accuracy,
                    )
                    feedback_items.append({"fn": fn_name, "feedback_key": feedback_key, "feedback": fb})
                    dev_entries.append({"stage": "feedback", "function": fn_name, "feedback_key": feedback_key, "feedback_len": len(fb or "")})

                return cand, "", ptype_used, feedback_key, prob_updates, feedback_items

            return None, error_msg or "retries exhausted", ptype_used, feedback_key, prob_updates, feedback_items

        root_acc = parent.accuracy
        child, err, ptype_used, feedback_key, prob_updates, feedback_items = _run_one_pass(parent, primary_category, root_accuracy=root_acc)
        if child is None:
                return {
                    "ok": False,
                    "error": err,
                    "local_cost": local_cost,
                    "dev_entries": dev_entries,
                    "prompt_io_entries": prompt_io_entries,
                    "attempted_codes": dict(attempted_codes_by_fn),
                    "chosen_cluster_id": int(chosen_cluster_id),
                    "target_clusters": int(job.get("target_clusters", 1)),
                "fn": fn_name,
                "primary_category": primary_category,
            }

        if primary_category in {"diagnostic", "ablation"}:
            second = "diagnostic_2" if primary_category == "diagnostic" else "ablation_2"
            child2, _, _, _, prob_updates2, feedback_items2 = _run_one_pass(child, second, root_accuracy=root_acc)
            prob_updates.extend(prob_updates2)
            feedback_items.extend(feedback_items2)
            if child2 is not None:
                child = child2

        out = {
            "ok": True,
            "record": child,
            "local_cost": local_cost,
            "dev_entries": dev_entries,
            "prompt_io_entries": prompt_io_entries,
            "attempted_codes": dict(attempted_codes_by_fn),
            "prob_updates": prob_updates,
            "feedback_items": feedback_items,
            "fn": fn_name,
            "primary_category": primary_category,
            "chosen_cluster_id": int(chosen_cluster_id),
            "target_clusters": int(job.get("target_clusters", 1)),
        }
        return out

    except Exception as e:
        return {
            "ok": False,
            "error": f"worker_exception: {type(e).__name__}: {e}",
            "local_cost": 0.0,
            "dev_entries": [{"stage": "worker_exception", "traceback": traceback.format_exc()}],
            "prompt_io_entries": [],
            "attempted_codes": {},
        }

def _dm_build_task_specs(task_by_name: Dict[str, Tuple[str, Any]]) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for fn, (kind, task) in task_by_name.items():
        spec: Dict[str, Any] = {
            "kind": kind,
            "hints": list(getattr(task, "hints", None) or []),
            "knowledge_tree": dict(getattr(task, "knowledge_tree", None) or {}),
            "kt_prob": dict(getattr(task, "kt_prob", None) or {}),
        }
        if kind == "data":
            spec["file_description"] = str(getattr(task, "file_description", "") or "")
            spec["data_summary"] = str(getattr(task, "data_summary", "") or "")
            spec["data_usage"] = str(getattr(task, "data_usage", "") or "")
            spec["read_cmd"] = str(getattr(task, "read_cmd", "") or "")
        specs[fn] = spec
    return specs

# ======================================================================================
# Main entry point (now much shorter; functionality preserved)
# ======================================================================================

def optimize(
    method_tasks: List["MethodTask"],
    data_tasks: List["DataTask"],
    reference_filename: str,
    *,
    paradigm_probs: Optional[Dict[str, float]] = None,  # category probs (prompt/diagnostic/ablation/...)
    n_generations: int = 5,          # (deprecated) no longer used in continuous mode
    children_per_model: int = 2,     # (deprecated) no longer used in continuous mode
    timeout: int = 300,
    bug_retries: int = 2,
    n_feedback_buffer: int = 5,
    skip_timeout: bool = True,
    prompt_samples: int = 5,
    diagnostic_prompts: Optional[List[str]] = None,
    ablation_prompts: Optional[List[str]] = None,
    simplify_prompts: Optional[List[str]] = None,
    drop_island_iter: int = 10,      # [OK] NOW MINUTES (default 10)
    prompt_decay: float = 1.1,
    prompt_importance: float = 2.0,
    global_hints: Optional[List[str]] = None,
    filename: str = "",
    output_dir: str = ".",
    history_name: str = "",
    TIME_LIMIT: int = 60,
    task_description: str = "",
    val_limit=None,
    debug: bool = False,
    min_improvement: float = 0.01,
    n_jobs: int = 1,
    max_islands: Optional[int] = None,  # cap initial islands via clustering (or None for all)
    memory_limit_gb: Optional[int] = 50,
    load_history: Optional[str] = None,
    sensitive_data: bool = False,
    batch_queries: int = 0,
    gpu_ids: Optional[List[int]] = None,
    cpu_threads_per_job: Optional[int] = None,
    multi_machine: bool = False,
) -> Tuple["ModelRecord", Dict[int, Dict[int, List["ModelRecord"]]]]:
    TIME_LIMIT = TIME_LIMIT * 60
    global MEM_LIMIT
    if memory_limit_gb is not None:
        MEM_LIMIT = int(memory_limit_gb)
    if diagnostic_prompts is None:
        diagnostic_prompts = _dm_load_prompt_list("diagnostic")
    if ablation_prompts is None:
        ablation_prompts = _dm_load_prompt_list("ablation")
    if simplify_prompts is None:
        simplify_prompts = _dm_load_prompt_list("simplify")
    global_hints = list(global_hints or [])
    if sensitive_data and SENSITIVE_DATA_NO_WRITE_HINT not in global_hints:
        global_hints.append(SENSITIVE_DATA_NO_WRITE_HINT)
    n_jobs = max(1, int(n_jobs))
    if cpu_threads_per_job is None:
        cpu_threads_per_job = max(1, (os.cpu_count() or 1) // n_jobs)
    os.environ["TUSOAI_CPU_THREADS_PER_JOB"] = str(max(1, int(cpu_threads_per_job)))
    drop_island_minutes = max(1, int(drop_island_iter))

    state = _DMRunState(start_time=time.time(), debug=debug)
    printer = _DMPrinter(state)

    ordered_tasks, ordered_fn_names, task_by_name, compat_single = _dm_prepare_tasks(method_tasks, data_tasks)
    task_probs = _dm_init_task_probs(ordered_fn_names)
    if paradigm_probs is None:
        paradigm_probs = _dm_default_paradigm_probs()
    category_probs = _dm_apply_sensitive_data_category_constraints(paradigm_probs, sensitive_data=sensitive_data)

    printer.hdr("DISCOVER_METHOD RUN START")
    printer.brief(
        "config",
        f"tasks={len(ordered_fn_names)} (method={len(method_tasks)}, data={len(data_tasks)}) | "
        f"timeout={timeout}s | bug_retries={bug_retries} | drop_every={drop_island_minutes}min | "
        f"debug={debug} | max_islands={max_islands} | n_jobs={n_jobs} | multi_machine={multi_machine} | TIME_LIMIT={TIME_LIMIT}s",
    )
    if debug:
        printer.debug_dump(
            "config_details",
            {
                "task_description": task_description,
                "TIME_LIMIT": TIME_LIMIT,
                "prompt_samples": prompt_samples,
                "n_feedback_buffer": n_feedback_buffer,
                "skip_timeout": skip_timeout,
                "prompt_decay": prompt_decay,
                "prompt_importance": prompt_importance,
                "min_improvement": min_improvement,
                "compat_single": compat_single,
                "initial_task_probs": dict(task_probs),
                "initial_category_probs": dict(category_probs) if category_probs else None,
                "global_hints": list(global_hints),
                "ordered_fn_names": list(ordered_fn_names),
                "max_islands": max_islands,
                "n_jobs": n_jobs,
                "drop_island_minutes": drop_island_minutes,
            },
        )

    function_sources = _dm_collect_function_sources(method_tasks, data_tasks, reference_filename)
    base_functions = _dm_extract_base_functions(ordered_fn_names, function_sources)
    loaded_dynamic_state: Optional[Dict[str, Any]] = None
    if load_history:
        loaded_functions = _dm_load_functions_from_history(
            load_history,
            ordered_fn_names,
            accuracy_tolerance=min_improvement,
        )
        if loaded_functions:
            base_functions.update(loaded_functions)
        loaded_dynamic_state = _dm_load_dynamic_state_from_history(load_history)
    base_path, logs = _dm_init_output_and_logs(output_dir, history_name=history_name, multi_machine=multi_machine)
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    eval_ctx = _DMEvalContext(
        base_path=base_path,
        reference_filename=reference_filename,
        ordered_fn_names=ordered_fn_names,
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=timeout,
        val_limit=val_limit,
        sensitive_data=bool(sensitive_data),
        printer=printer,
        logs=logs,
    )
    _dm_log_run_wiring(logs, eval_ctx, mode="discover_method")

    attempts_by_fn: Dict[str, set[str]] = {fn: {base_functions[fn]} for fn in ordered_fn_names}
    feedbacks: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # ==================================================================================
    # SEEDING: parallel LLM init + (optional) baseline + (optional) parallel combos
    # ==================================================================================
    eval_cache: Dict[str, "ModelRecord"] = {}

    method_variants, method_fns, single_task_seed_records, eval_cache = _dm_seed_method_variant_pools(
        method_tasks=method_tasks,
        base_functions=base_functions,
        global_hints=global_hints,
        task_description=task_description,
        bug_retries=bug_retries,
        skip_timeout=skip_timeout,
        filename=filename,
        state=state,
        printer=printer,
        logs=logs,
        eval_ctx=eval_ctx,
        attempts_by_fn=attempts_by_fn,
        n_jobs=n_jobs,
        gpu_ids=gpu_ids,
        eval_cache=eval_cache,
    )

    if single_task_seed_records is not None:
        combo_records = single_task_seed_records
        if not combo_records:
            raise RuntimeError("No valid initial variants - all seeding evaluations failed.")
    else:
        combo_records = _dm_evaluate_seed_combinations_parallel(
            method_variants=method_variants,
            method_fns=method_fns,
            base_functions=base_functions,
            max_islands=max_islands,
            min_improvement=min_improvement,
            state=state,
            printer=printer,
            logs=logs,
            eval_ctx=eval_ctx,
            n_jobs=n_jobs,
            eval_cache=eval_cache,
        )

    islands, history, global_models = _dm_create_islands_from_seed(
        seed_records=combo_records,
        logs=logs,
        printer=printer,
        state=state,
        max_islands=max_islands
    )

    # ==================================================================================
    # Mutation context (parent-held mutable state)
    # ==================================================================================
    if not category_probs:
        raise RuntimeError("paradigm_probs (category_probs) must be provided and non-empty.")

    # Ensure kt_prob exists (parent-side) so merges/updates always have a target dict
    for fn, (kind, task) in task_by_name.items():
        if not getattr(task, "kt_prob", None):
            keys = list(getattr(task, "knowledge_tree", {}).keys())
            if keys:
                task.kt_prob = {k: 1.0 for k in keys}
                update_probabilities(task.kt_prob, keys[0], 1.0)

    # ==================================================================================
    # Continuous parallel evolution (no generations)
    #
    # IMPORTANT: sklearn clustering is done in the PARENT once per batch, then we pass
    # "champions" into workers. Workers do NOT run sklearn.
    # ==================================================================================
    if not global_models:
        raise RuntimeError("No seed models were produced; cannot start evolution.")

    global_best = max((m for m in global_models if m), key=lambda m: m.accuracy)
    printer.brief("best", f"initial best acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s")
    original_rec = _dm_find_original_record(global_models)
    if original_rec is not None:
        printer.brief("best", f"original baseline acc={original_rec.accuracy:.6f} rt={original_rec.runtime:.3f}s")

    # [OK] do not count initialization time towards dropping / optimization time
    optimization_start_ts = time.time()

    active_islands_target = max(1, len(islands))
    next_drop_ts = optimization_start_ts + drop_island_minutes * 60.0
    step = 0

    try:
        import multiprocess as mp  # type: ignore
    except Exception:  # pragma: no cover
        import multiprocessing as mp  # type: ignore

    # parent-side sklearn imports (kept out of worker)
    from collections import defaultdict as _dd
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    while True:
        # TIME_LIMIT still measured from run start (unchanged)
        if time.time() - state.start_time > TIME_LIMIT:
            printer.brief("stop", "Time limit exceeded. Stopping optimization.")
            break

        # optimization time (used for printing + island dropping)
        opt_now_s = time.time() - optimization_start_ts

        # time-based island dropping (every X minutes) based on optimization time
        now_ts = time.time()
        while now_ts >= next_drop_ts and active_islands_target > 1:
            active_islands_target -= 1
            next_drop_ts += drop_island_minutes * 60.0
            printer.brief("drop", f"active_islands_target -> {active_islands_target} at opt_t={opt_now_s:.1f}s")

        # candidates with code
        cand_with_code = [m for m in global_models if m and getattr(m, "code", None)]
        if not cand_with_code:
            printer.brief("stop", "No candidates with code; cannot continue optimization.")
            break

        active_islands = max(1, min(active_islands_target, len(cand_with_code)))

        # print current optimization time + active islands
        printer.hdr(f"OPTIMIZATION t={opt_now_s:.1f}s")
        printer.brief(
            "status",
            f"active_islands={active_islands} | candidates={len(global_models)} | best={global_best.accuracy:.6f}",
        )

        # snapshots for parallel jobs (picklable)
        feedbacks_snapshot: Dict[str, Dict[str, List[str]]] = {}
        for fn, by_key in feedbacks.items():
            feedbacks_snapshot[fn] = {k: v[-n_feedback_buffer:] for k, v in by_key.items()}

        attempts_snapshot = {fn: set(s) for fn, s in attempts_by_fn.items()}

        # Build task_specs fresh each batch so workers see updated kt_prob
        task_specs = _dm_build_task_specs(task_by_name)

        # -------------------------
        # Parent-side sklearn clustering (same as original behavior)
        # -------------------------
        models_for_texts = [m for m in cand_with_code if m and getattr(m, "code", None)]
        texts = [m.code for m in models_for_texts]
        if not texts:
            printer.brief("stop", "No candidates with code (unexpected).")
            break

        k = max(1, min(active_islands, len(texts)))
        X = TfidfVectorizer(stop_words="english").fit_transform(texts)
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

        clustered = _dd(list)
        for m, lab in zip(models_for_texts, labels):
            clustered[int(lab)].append(m)

        champions: List[Tuple[int, ModelRecord]] = []
        for cid, ms in clustered.items():
            if not ms:
                continue
            champ = _pick_cluster_champion(ms, min_improvement=min_improvement)
            champions.append((int(cid), champ))

        if not champions:
            printer.brief("stop", "No champions after clustering; cannot continue optimization.")
            break

        # -------------------------
        # Build worker jobs (NO Task objects; NO sklearn; NO full models list)
        # -------------------------
        jobs: List[Dict[str, Any]] = []
        gpu_cycle = list(gpu_ids or [])
        for job_idx in range(n_jobs):
            gpu_id = gpu_cycle[job_idx % len(gpu_cycle)] if gpu_cycle else None
            jobs.append(
                {
                    "champions": champions,                 # [OK] computed in parent via sklearn
                    "task_specs": task_specs,               # [OK] picklable

                    "target_clusters": int(k),              # optional (logging)
                    "min_improvement": float(min_improvement),
                    "seed": random.randint(0, 2**31 - 1),

                    "task_probs": dict(task_probs),
                    "category_probs": dict(category_probs),

                    "feedbacks_snapshot": feedbacks_snapshot,
                    "attempts_snapshot": attempts_snapshot,

                    "global_hints": list(global_hints),
                    "task_description": task_description,

                    "prompt_samples": int(prompt_samples),
                    "diagnostic_prompts": list(diagnostic_prompts),
                    "ablation_prompts": list(ablation_prompts),
                    "simplify_prompts": list(simplify_prompts),
                    "sensitive_data": bool(sensitive_data),
                    "n_feedback_buffer": int(n_feedback_buffer),

                    "timeout": int(timeout),
                    "bug_retries": int(bug_retries),
                    "skip_timeout": bool(skip_timeout),
                    "batch_queries": int(batch_queries),
                    "gpu_id": gpu_id,
                    "prompt_decay": float(prompt_decay),
                    "prompt_importance": float(prompt_importance),
                    "val_limit": val_limit,

                    "reference_filename": reference_filename,
                    "base_path": str(eval_ctx.base_path),
                    "ordered_fn_names": list(ordered_fn_names),
                    "function_sources": dict(eval_ctx.function_sources),
                    "repo_snapshots": {k: str(v) for k, v in eval_ctx.repo_snapshots.items()},
                }
            )

        # run in parallel (each job produces at most one child)
        results: List[Dict[str, Any]] = []
        if n_jobs > 1:
            with mp.Pool(processes=n_jobs) as pool:
                for r in pool.imap_unordered(_dm_evolution_worker_job, jobs, chunksize=1):
                    results.append(r)
        else:
            for j in jobs:
                results.append(_dm_evolution_worker_job(j))

        any_success = False

        for r in results:
            state.add_cost(float(r.get("local_cost", 0.0)))

            # dev logs (parent-only to avoid concurrent JSON writes)
            for entry in r.get("dev_entries", []) or []:
                e = dict(entry)
                e["total_cost"] = float(state.total_cost)
                logs.log_dev(e)
                if debug:
                    printer.debug_dump(e.get("stage", "evolve"), dict(e))
            for pio in r.get("prompt_io_entries", []) or []:
                pp = dict(pio)
                pp["total_cost"] = float(state.total_cost)
                logs.log_prompt_io(pp)

            # burn attempted codes globally (even if job fails)
            attempted = r.get("attempted_codes", {}) or {}
            for fn, codes in attempted.items():
                for code in codes:
                    attempts_by_fn.setdefault(fn, set()).add(code)

            if not r.get("ok"):
                continue

            child: ModelRecord = r["record"]
            any_success = True
            step += 1

            # merge feedbacks
            for fb_item in r.get("feedback_items", []) or []:
                fn = fb_item["fn"]
                key = fb_item["feedback_key"]
                feedbacks[fn][key].append(fb_item["feedback"])

            # apply probability updates (parent-authoritative)
            for upd in r.get("prob_updates", []) or []:
                if upd.get("type") == "task_probs":
                    update_probabilities(task_probs, upd["fn"], float(upd["factor"]))
                elif upd.get("type") == "kt_prob":
                    fn = upd["fn"]
                    ptype = upd["ptype"]
                    _, task = task_by_name[fn]
                    if not getattr(task, "kt_prob", None):
                        keys = list(getattr(task, "knowledge_tree", {}).keys())
                        if keys:
                            task.kt_prob = {k: 1.0 for k in keys}
                            update_probabilities(task.kt_prob, keys[0], 1.0)
                    if getattr(task, "kt_prob", None):
                        update_probabilities(task.kt_prob, ptype, float(upd["factor"]))

            # add child to global pool
            global_models.append(child)

            logs.log_history(
                {
                    "stage": "evolve",
                    "step": step,
                    "opt_time_s": float(time.time() - optimization_start_ts),
                    "active_islands": int(active_islands),
                    "chosen_cluster_id": r.get("chosen_cluster_id"),
                    "lineage": getattr(child, "lineage", ""),
                    "accuracy": child.accuracy,
                    "runtime": child.runtime,
                    "code": child.code,
                    "total_cost": float(state.total_cost),
                }
            )

            # keep history shaped like Dict[int, Dict[int, List[ModelRecord]]]
            cid = int(r.get("chosen_cluster_id", 0) or 0)
            history.setdefault(cid, {}).setdefault(step, []).append(child)

            if child.accuracy > global_best.accuracy:
                prev_best = global_best
                global_best = child
                printer.brief(
                    "best",
                    f"NEW BEST opt_t={time.time() - optimization_start_ts:.1f}s acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s",
                )
                printer.brief(
                    "improve",
                    (
                        f"Optimization {step}: performance {prev_best.accuracy:.6f}->{global_best.accuracy:.6f}, "
                        f"runtime {(prev_best.runtime if prev_best.runtime is not None else float('nan')):.3f}"
                        f"->{(global_best.runtime if global_best.runtime is not None else float('nan')):.3f} seconds, "
                        f"code length {_dm_code_len_lines(prev_best.code)}->{_dm_code_len_lines(global_best.code)} lines."
                    ),
                )

        if not any_success:
            printer.brief(
                "loop",
                f"no successful children at opt_t={time.time() - optimization_start_ts:.1f}s (active_islands={active_islands})",
            )

    printer.hdr("DISCOVER_METHOD RUN END")
    printer.brief(
        "final",
        f"best_acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s | total_cost={state.money(state.total_cost)}",
    )
    return global_best, history

# ==================================================================================================
# FIX 1 IMPLEMENTATION: do NOT pass ModelRecord objects to workers.
# - Parent sends ONLY picklable "champion payload" dicts (functions/code/metrics as primitives)
# - Worker reconstructs ModelRecord locally, runs existing _dm_evolution_worker_job,
#   then converts the returned child ModelRecord back into a picklable payload for the parent.
# - Parent reconstructs ModelRecord from payload and updates global state.
#
# Drop-in replacement for your discover_method() (keeps your signature).
# Requires: ModelRecord, _dm_evolution_worker_job, _dm_build_task_specs, _pick_cluster_champion,
#           _dm_* seeding helpers already defined as in your file.
# ==================================================================================================

def _dm__safe_model_info_str(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        try:
            return repr(x)
        except Exception:
            return "<unreprable model_info>"

def _dm__record_to_payload(rec: "ModelRecord") -> Dict[str, Any]:
    return {
        "code": str(rec.code),
        "file": str(rec.file),
        "accuracy": float(rec.accuracy),
        "runtime": float(rec.runtime) if rec.runtime is not None else None,
        "model_info_str": _dm__safe_model_info_str(rec.model_info),
        "lineage": str(rec.lineage),
        "functions": dict(rec.functions),
    }

def _dm__payload_to_record(p: Dict[str, Any]) -> "ModelRecord":
    return ModelRecord(
        code=p.get("code", ""),
        file=Path(p.get("file", "")) if p.get("file") else Path(""),
        accuracy=float(p.get("accuracy", 0.0)),
        runtime=float(p["runtime"]) if p.get("runtime") is not None else 0.0,
        model_info=p.get("model_info_str", None),  # keep as STRING to stay pickle-safe
        lineage=p.get("lineage", "payload"),
        functions=dict(p.get("functions", {})),
    )

def _dm_evolution_worker_job_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker wrapper:
      - job["champions"] is a list of picklable dict payloads (NOT ModelRecord)
      - reconstruct ModelRecord champs locally
      - call your existing _dm_evolution_worker_job(job2)
      - convert returned child ModelRecord -> record_payload dict
    """
    champs_payload = job.get("champions", []) or []
    champs: List[Tuple[int, ModelRecord]] = []
    for cp in champs_payload:
        cid = int(cp["cid"])
        # model_info_str is already string; safe
        parent = ModelRecord(
            code=cp.get("code", ""),
            file=Path(""),  # not needed for evaluation; worker writes its own files
            accuracy=float(cp.get("accuracy", 0.0)),
            runtime=float(cp["runtime"]) if cp.get("runtime") is not None else 0.0,
            model_info=cp.get("model_info_str", None),
            lineage=cp.get("lineage", "champion"),
            functions=dict(cp.get("functions", {})),
        )
        champs.append((cid, parent))

    job2 = dict(job)
    job2["champions"] = champs

    out = _dm_evolution_worker_job(job2)

    # Convert child ModelRecord to payload to avoid pickling model_info objects back to parent
    if out.get("ok") and out.get("record") is not None:
        child: ModelRecord = out["record"]
        out = dict(out)
        out.pop("record", None)
        out["record_payload"] = _dm__record_to_payload(child)

    return out


def _dm_default_paradigm_probs() -> Dict[str, float]:
    return {"prompt": 0.7, "diagnostic": 0.1, "ablation": 0.1, "simplify": 0.1}


def _dm_apply_sensitive_data_category_constraints(
    probs: Dict[str, float],
    *,
    sensitive_data: bool,
) -> Dict[str, float]:
    out = dict(probs or {})
    if not sensitive_data:
        return out

    if "diagnostic" in out:
        out["diagnostic"] = 0.0

    total = sum(v for v in out.values() if v > 0)
    if total <= 0:
        raise RuntimeError("All category probabilities became zero after sensitive_data constraints.")

    return {k: (max(v, 0.0) / total) for k, v in out.items()}


def _dm_load_prompt_list(kind: str) -> List[str]:
    try:
        from tusoai.prompts import load_ablation_prompts, load_diagnostic_prompts, load_simplify_prompts

        if kind == "diagnostic":
            return load_diagnostic_prompts()
        if kind == "ablation":
            return load_ablation_prompts()
        if kind == "simplify":
            return load_simplify_prompts()
    except FileNotFoundError:
        return []
    return []


def optimize(
    method_tasks: List["MethodTask"],
    data_tasks: List["DataTask"],
    reference_filename: str,
    *,
    paradigm_probs: Optional[Dict[str, float]] = None,  # category probs (prompt/diagnostic/ablation/...)
    n_generations: int = 5,          # (deprecated) no longer used in continuous mode
    children_per_model: int = 2,     # (deprecated) no longer used in continuous mode
    timeout: int = 300,
    bug_retries: int = 2,
    n_feedback_buffer: int = 5,
    skip_timeout: bool = True,
    prompt_samples: int = 5,
    diagnostic_prompts: List[str] = None,
    ablation_prompts: List[str] = None,
    simplify_prompts: List[str] = None,
    drop_island_iter: int = 10,      # minutes
    prompt_decay: float = 1.1,
    prompt_importance: float = 2.0,
    global_hints: Optional[List[str]] = None,
    filename: str = "",
    output_dir: str = ".",
    history_name: str = "",
    TIME_LIMIT: int = 60,
    task_description: str = "",
    val_limit=None,
    debug: bool = False,
    min_improvement: float = 0.01,
    n_jobs: int = 1,
    max_islands: Optional[int] = None,
    COST_LIMIT=10,
    memory_limit_gb: Optional[int] = 50,
    load_history: Optional[str] = None,
    sensitive_data: bool = False,
    batch_queries: int = 0,
    gpu_ids: Optional[List[int]] = None,
    cpu_threads_per_job: Optional[int] = None,
    multi_machine: bool = False,
) -> Tuple["ModelRecord", Dict[int, Dict[int, List["ModelRecord"]]]]:

    # -------------------------
    # Basic setup
    # -------------------------
    TIME_LIMIT = TIME_LIMIT * 60
    global MEM_LIMIT
    if memory_limit_gb is not None:
        MEM_LIMIT = int(memory_limit_gb)
    if diagnostic_prompts is None:
        diagnostic_prompts = _dm_load_prompt_list("diagnostic")
    if ablation_prompts is None:
        ablation_prompts = _dm_load_prompt_list("ablation")
    if simplify_prompts is None:
        simplify_prompts = _dm_load_prompt_list("simplify")

    global_hints = list(global_hints or [])
    if sensitive_data and SENSITIVE_DATA_NO_WRITE_HINT not in global_hints:
        global_hints.append(SENSITIVE_DATA_NO_WRITE_HINT)
    n_jobs = max(1, int(n_jobs))
    if cpu_threads_per_job is None:
        cpu_threads_per_job = max(1, (os.cpu_count() or 1) // n_jobs)
    os.environ["TUSOAI_CPU_THREADS_PER_JOB"] = str(max(1, int(cpu_threads_per_job)))
    drop_island_minutes = max(1, int(drop_island_iter))

    state = _DMRunState(start_time=time.time(), debug=debug)
    printer = _DMPrinter(state)

    ordered_tasks, ordered_fn_names, task_by_name, compat_single = _dm_prepare_tasks(method_tasks, data_tasks)
    task_probs = _dm_init_task_probs(ordered_fn_names)
    if paradigm_probs is None:
        paradigm_probs = _dm_default_paradigm_probs()
    category_probs = _dm_apply_sensitive_data_category_constraints(paradigm_probs, sensitive_data=sensitive_data)

    printer.hdr("DISCOVER_METHOD RUN START")
    printer.brief(
        "config",
        f"tasks={len(ordered_fn_names)} (method={len(method_tasks)}, data={len(data_tasks)}) | "
        f"timeout={timeout}s | bug_retries={bug_retries} | drop_every={drop_island_minutes}min | "
        f"debug={debug} | max_islands={max_islands} | n_jobs={n_jobs} | multi_machine={multi_machine} | TIME_LIMIT={TIME_LIMIT}s",
    )

    if not category_probs:
        raise RuntimeError("paradigm_probs (category_probs) must be provided and non-empty.")

    function_sources = _dm_collect_function_sources(method_tasks, data_tasks, reference_filename)
    base_functions = _dm_extract_base_functions(ordered_fn_names, function_sources)
    loaded_dynamic_state: Optional[Dict[str, Any]] = None
    if load_history:
        loaded_functions = _dm_load_functions_from_history(
            load_history,
            ordered_fn_names,
            accuracy_tolerance=min_improvement,
        )
        if loaded_functions:
            base_functions.update(loaded_functions)
        loaded_dynamic_state = _dm_load_dynamic_state_from_history(load_history)

    base_path, logs = _dm_init_output_and_logs(output_dir, history_name=history_name, multi_machine=multi_machine)
    repo_snapshots = _dm_init_repo_snapshots(function_sources, base_path)

    eval_ctx = _DMEvalContext(
        base_path=base_path,
        reference_filename=reference_filename,
        ordered_fn_names=ordered_fn_names,
        function_sources=function_sources,
        repo_snapshots=repo_snapshots,
        timeout=timeout,
        val_limit=val_limit,
        sensitive_data=bool(sensitive_data),
        printer=printer,
        logs=logs,
    )
    _dm_log_run_wiring(logs, eval_ctx, mode="evolve_method")
    shared_history_path = str(logs.history_path) if multi_machine else None
    if multi_machine and not load_history:
        loaded_dynamic_state = _dm_load_dynamic_state_from_history(str(logs.history_path))

    attempts_by_fn: Dict[str, set[str]] = {fn: {base_functions[fn]} for fn in ordered_fn_names}
    feedbacks: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # Ensure parent-side kt_prob exists so updates always have targets
    for fn, (kind, task) in task_by_name.items():
        if not getattr(task, "kt_prob", None):
            keys = list(getattr(task, "knowledge_tree", {}).keys())
            if keys:
                task.kt_prob = {k: 1.0 for k in keys}
                update_probabilities(task.kt_prob, keys[0], 1.0)

    if loaded_dynamic_state:
        loaded_task_probs = loaded_dynamic_state.get("task_probs", {})
        if loaded_task_probs:
            for fn in ordered_fn_names:
                if fn in loaded_task_probs:
                    task_probs[fn] = float(loaded_task_probs[fn])
        loaded_cat_probs = loaded_dynamic_state.get("category_probs", {})
        if loaded_cat_probs:
            for k in list(category_probs.keys()):
                if k in loaded_cat_probs:
                    category_probs[k] = float(loaded_cat_probs[k])
        for fn, by_key in (loaded_dynamic_state.get("feedbacks", {}) or {}).items():
            if fn not in feedbacks:
                feedbacks[fn] = defaultdict(list)
            for key, vals in (by_key or {}).items():
                feedbacks[fn][key] = list(vals or [])
        for fn, probs in (loaded_dynamic_state.get("task_kt_probs", {}) or {}).items():
            if fn in task_by_name and isinstance(probs, dict) and probs:
                _, task = task_by_name[fn]
                task.kt_prob = {str(k): float(v) for k, v in probs.items()}
        printer.brief("history", "loaded dynamic task/category probabilities, kt_prob and feedbacks from history")

    # -------------------------
    # SEEDING (your existing functions)
    # -------------------------
    eval_cache: Dict[str, "ModelRecord"] = {}

    seed_history_path = load_history or (str(logs.history_path) if multi_machine and _dm_load_history_records_pool(str(logs.history_path), ordered_fn_names) else None)
    if seed_history_path:
        printer.brief("seed", "history provided: skipping initialization seeding/combo stage")
        selected_hist = _dm_get_selected_history_summary(
            seed_history_path,
            accuracy_tolerance=min_improvement,
        )
        if selected_hist is not None:
            printer.brief(
                "history_boot",
                (
                    f"loaded history entries={int(selected_hist['history_count'])} | "
                    f"top_acc={selected_hist['best_accuracy']:.6f} | "
                    f"selected initial method from history (pre-eval) "
                    f"acc={selected_hist['accuracy']:.6f} rt={selected_hist['runtime']:.3f}s "
                    f"code_len={int(selected_hist['code_len_lines'])} lines "
                    f"score={selected_hist['selection_score']:.6f}"
                ),
            )
        restored = _dm_load_history_records_pool(seed_history_path, ordered_fn_names)
        if restored:
            islands = {0: list(restored)}
            history = {0: {0: list(restored)}}
            global_models = list(restored)
            printer.brief("history", f"restored solution pool size={len(restored)} from history")
        else:
            raise RuntimeError(
                "history was provided but no valid history records could be restored; "
                "refusing to fall back to initial/base code."
            )
    else:
        method_variants, method_fns, single_task_seed_records, eval_cache = _dm_seed_method_variant_pools(
            method_tasks=method_tasks,
            base_functions=base_functions,
            global_hints=global_hints,
            task_description=task_description,
            bug_retries=bug_retries,
            skip_timeout=skip_timeout,
            filename=filename,
            state=state,
            printer=printer,
            logs=logs,
            eval_ctx=eval_ctx,
            attempts_by_fn=attempts_by_fn,
            n_jobs=n_jobs,
            gpu_ids=gpu_ids,
            eval_cache=eval_cache,
        )

        if single_task_seed_records is not None:
            combo_records = single_task_seed_records
            if not combo_records:
                raise RuntimeError("No valid initial variants - all seeding evaluations failed.")
        else:
            combo_records = _dm_evaluate_seed_combinations_parallel(
                method_variants=method_variants,
                method_fns=method_fns,
                base_functions=base_functions,
                max_islands=max_islands,
                min_improvement=min_improvement,
                state=state,
                printer=printer,
                logs=logs,
                eval_ctx=eval_ctx,
                n_jobs=n_jobs,
                eval_cache=eval_cache,
            )

        islands, history, global_models = _dm_create_islands_from_seed(
            seed_records=combo_records,
            logs=logs,
            printer=printer,
            state=state,
            max_islands=max_islands
        )

    if not global_models:
        raise RuntimeError("No seed models were produced; cannot start evolution.")

    global_best = max((m for m in global_models if m), key=lambda m: m.accuracy)
    printer.brief("best", f"initial best acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s")
    original_rec = _dm_find_original_record(global_models)
    if original_rec is not None:
        printer.brief("best", f"original baseline acc={original_rec.accuracy:.6f} rt={original_rec.runtime:.3f}s")

    # -------------------------
    # Streaming optimization loop (true "as jobs finish, launch next")
    # -------------------------
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import multiprocessing as _mp

    # (optional) fork context helps on Unix; on Windows this will fail if forced.
    mp_context = None
    try:
        mp_context = _mp.get_context("fork")
    except Exception:
        mp_context = None

    optimization_start_ts = time.time()
    active_islands_target = max(1, len(islands))
    next_drop_ts = optimization_start_ts + drop_island_minutes * 60.0
    step = 0

    printer.hdr("OPTIMIZATION START (STREAMING)")

    # parent-side sklearn (used to compute champions payload for each submission)
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _compute_champions_payload(active_islands: int) -> Tuple[List[Dict[str, Any]], int]:
        cand_with_code = [m for m in global_models if m and getattr(m, "code", None)]
        if not cand_with_code:
            return [], 0

        texts = [m.code for m in cand_with_code]
        k = max(1, min(active_islands, len(texts)))

        if k == 1:
            champ = _pick_cluster_champion(cand_with_code, min_improvement=min_improvement)
            payload = [{
                "cid": 0,
                "functions": dict(champ.functions),
                "accuracy": float(champ.accuracy),
                "runtime": float(champ.runtime) if champ.runtime is not None else None,
                "model_info_str": _dm__safe_model_info_str(champ.model_info),
                "lineage": str(champ.lineage),
                "code": str(champ.code),
            }]
            return payload, 1

        X = TfidfVectorizer(stop_words="english").fit_transform(texts)
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
        except TypeError:
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

        clustered: Dict[int, List[ModelRecord]] = defaultdict(list)
        for m, lab in zip(cand_with_code, labels):
            clustered[int(lab)].append(m)

        payload: List[Dict[str, Any]] = []
        for cid, ms in clustered.items():
            if not ms:
                continue
            champ = _pick_cluster_champion(ms, min_improvement=min_improvement)
            payload.append({
                "cid": int(cid),
                "functions": dict(champ.functions),
                "accuracy": float(champ.accuracy),
                "runtime": float(champ.runtime) if champ.runtime is not None else None,
                "model_info_str": _dm__safe_model_info_str(champ.model_info),
                "lineage": str(champ.lineage),
                "code": str(champ.code),
            })

        return payload, k

    def _build_one_job(active_islands: int) -> Optional[Dict[str, Any]]:
        champions_payload, k = _compute_champions_payload(active_islands)
        if not champions_payload:
            return None

        # snapshots
        feedbacks_snapshot: Dict[str, Dict[str, List[str]]] = {
            fn: {key: vals[-n_feedback_buffer:] for key, vals in by_key.items()}
            for fn, by_key in feedbacks.items()
        }
        attempts_snapshot = {fn: set(s) for fn, s in attempts_by_fn.items()}

        # task specs rebuilt every submission so kt_prob updates are reflected
        task_specs = _dm_build_task_specs(task_by_name)

        return {
            "champions": champions_payload,      # [OK] picklable dicts only
            "target_clusters": int(k),
            "min_improvement": float(min_improvement),
            "seed": random.randint(0, 2**31 - 1),

            "task_probs": dict(task_probs),
            "category_probs": dict(category_probs),

            "feedbacks_snapshot": feedbacks_snapshot,
            "attempts_snapshot": attempts_snapshot,

            "task_specs": task_specs,
            "global_hints": list(global_hints),
            "task_description": task_description,

            "prompt_samples": int(prompt_samples),
            "diagnostic_prompts": list(diagnostic_prompts),
            "ablation_prompts": list(ablation_prompts),
            "simplify_prompts": list(simplify_prompts),
            "sensitive_data": bool(sensitive_data),
            "n_feedback_buffer": int(n_feedback_buffer),

            "timeout": int(timeout),
            "bug_retries": int(bug_retries),
            "skip_timeout": bool(skip_timeout),
            "batch_queries": int(batch_queries),
            "gpu_id": None,
            "prompt_decay": float(prompt_decay),
            "prompt_importance": float(prompt_importance),
            "val_limit": val_limit,

            "reference_filename": reference_filename,
            "base_path": str(eval_ctx.base_path),
            "ordered_fn_names": list(ordered_fn_names),
            "function_sources": dict(eval_ctx.function_sources),
            "repo_snapshots": {k: str(v) for k, v in eval_ctx.repo_snapshots.items()},
        }

    inflight: Dict[Any, float] = {}
    submitted = 0
    last_status_ts = 0.0

    ex_kwargs = dict(max_workers=n_jobs)
    if mp_context is not None:
        ex_kwargs["mp_context"] = mp_context  # type: ignore

    with ProcessPoolExecutor(**ex_kwargs) as ex:
        while True:
            # hard stop: total run time (from run start, same as you had)
            if time.time() - state.start_time > TIME_LIMIT:
                printer.brief("stop", "Time limit exceeded. Stopping optimization.")
                break

            if float(state.total_cost) > COST_LIMIT:
                printer.brief("stop", "Cost limit exceeded. Stopping optimization.")
                break
            
            # drop islands by optimization time (not init time)
            now_ts = time.time()
            opt_now_s = now_ts - optimization_start_ts
            while now_ts >= next_drop_ts and active_islands_target > 1:
                active_islands_target -= 1
                next_drop_ts += drop_island_minutes * 60.0
                printer.brief("drop", f"active_islands_target -> {active_islands_target} at opt_t={opt_now_s:.1f}s")

            if multi_machine and shared_history_path:
                added_shared, total_shared = _dm_sync_shared_history_records(
                    shared_history_path, ordered_fn_names, global_models, history, step=step
                )
                if added_shared:
                    global_best = max((m for m in global_models if m), key=lambda m: m.accuracy)
                    printer.brief(
                        "multi_machine",
                        f"pulled {added_shared} shared candidates from {total_shared} history records",
                    )
                latest_dynamic_state = _dm_load_dynamic_state_from_history(shared_history_path)
                if latest_dynamic_state:
                    for fn, val in (latest_dynamic_state.get("task_probs", {}) or {}).items():
                        if fn in task_probs:
                            task_probs[fn] = float(val)
                    for cat, val in (latest_dynamic_state.get("category_probs", {}) or {}).items():
                        if cat in category_probs:
                            category_probs[cat] = float(val)
                    for fn, probs in (latest_dynamic_state.get("task_kt_probs", {}) or {}).items():
                        if fn in task_by_name and isinstance(probs, dict) and probs:
                            _, task = task_by_name[fn]
                            task.kt_prob = {str(k): float(v) for k, v in probs.items()}
                    for fn, by_key in (latest_dynamic_state.get("feedbacks", {}) or {}).items():
                        if fn not in feedbacks:
                            feedbacks[fn] = defaultdict(list)
                        for key, vals in (by_key or {}).items():
                            feedbacks[fn][key] = list(vals or [])

            # submit up to n_jobs inflight
            while len(inflight) < n_jobs:
                cand_count = len([m for m in global_models if m and getattr(m, "code", None)])
                active_islands = max(1, min(active_islands_target, cand_count)) if cand_count else 1

                job = _build_one_job(active_islands)
                if job is None:
                    break

                gpu_cycle = list(gpu_ids or [])
                if gpu_cycle:
                    job["gpu_id"] = gpu_cycle[submitted % len(gpu_cycle)]
                fut = ex.submit(_dm_evolution_worker_job_payload, job)
                inflight[fut] = time.time()
                submitted += 1

            # status every ~5s (matches your style)
            now_ts = time.time()
            if now_ts - last_status_ts >= 60.0:
                infl = len(inflight)
                cand = len([m for m in global_models if m and getattr(m, "code", None)])
                printer.brief(
                    "status",
                    f"opt_t={now_ts - optimization_start_ts:.1f}s inflight={infl} submitted={submitted} "
                    f"candidates={cand} best={global_best.accuracy:.6f}",
                )
                last_status_ts = now_ts

            if not inflight:
                printer.brief("stop", "No inflight jobs and no job could be built.")
                break

            done, _ = wait(list(inflight.keys()), timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for fut in done:
                inflight.pop(fut, None)

                try:
                    r = fut.result()
                except Exception as e:
                    printer.brief("future_exception", f"{type(e).__name__}: {e}")
                    continue

                # always account cost + log dev entries
                state.add_cost(float(r.get("local_cost", 0.0)))

                for entry in r.get("dev_entries", []) or []:
                    e = dict(entry)
                    e["total_cost"] = float(state.total_cost)
                    logs.log_dev(e)
                    if debug:
                        printer.debug_dump(e.get("stage", "evolve"), dict(e))
                for pio in r.get("prompt_io_entries", []) or []:
                    pp = dict(pio)
                    pp["total_cost"] = float(state.total_cost)
                    logs.log_prompt_io(pp)

                # burn attempted codes globally (even on failure)
                attempted = r.get("attempted_codes", {}) or {}
                for fn, codes in attempted.items():
                    for code in codes:
                        attempts_by_fn.setdefault(fn, set()).add(code)

                if not r.get("ok"):
                    printer.brief(
                        "evolve_fail",
                        f"err={r.get('error')} fn={r.get('fn')} cat={r.get('primary_category')} cluster={r.get('chosen_cluster_id')}",
                    )
                    continue

                # reconstruct child from payload
                payload = r.get("record_payload")
                if not payload:
                    printer.brief("evolve_fail", "worker returned ok=True but missing record_payload")
                    continue

                child = _dm__payload_to_record(payload)

                # merge feedbacks
                for fb_item in r.get("feedback_items", []) or []:
                    fn = fb_item["fn"]
                    key = fb_item["feedback_key"]
                    feedbacks[fn][key].append(fb_item["feedback"])

                # apply probability updates (authoritative in parent)
                for upd in r.get("prob_updates", []) or []:
                    if upd.get("type") == "task_probs":
                        update_probabilities(task_probs, upd["fn"], float(upd["factor"]))
                    elif upd.get("type") == "kt_prob":
                        fn = upd["fn"]
                        ptype = upd["ptype"]
                        _, task = task_by_name[fn]
                        if getattr(task, "kt_prob", None):
                            update_probabilities(task.kt_prob, ptype, float(upd["factor"]))

                # add child
                global_models.append(child)
                step += 1

                logs.log_history(
                    {
                        "stage": "evolve",
                        "step": step,
                        "opt_time_s": float(time.time() - optimization_start_ts),
                        "active_islands_target": int(active_islands_target),
                        "chosen_cluster_id": r.get("chosen_cluster_id"),
                        "lineage": getattr(child, "lineage", ""),
                        "accuracy": child.accuracy,
                        "runtime": child.runtime,
                        "code": child.code,
                        "total_cost": float(state.total_cost),
                        **_dm_snapshot_dynamic_state(task_probs, category_probs, feedbacks, task_by_name),
                    }
                )

                cid = int(r.get("chosen_cluster_id", 0) or 0)
                history.setdefault(cid, {}).setdefault(step, []).append(child)

                if child.accuracy > global_best.accuracy:
                    prev_best = global_best
                    global_best = child
                    printer.brief(
                        "best",
                        f"NEW BEST opt_t={time.time() - optimization_start_ts:.1f}s acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s",
                    )
                    printer.brief(
                        "improve",
                        (
                            f"Optimization {step}: performance {prev_best.accuracy:.6f}->{global_best.accuracy:.6f}, "
                            f"runtime {(prev_best.runtime if prev_best.runtime is not None else float('nan')):.3f}"
                            f"->{(global_best.runtime if global_best.runtime is not None else float('nan')):.3f} seconds, "
                            f"code length {_dm_code_len_lines(prev_best.code)}->{_dm_code_len_lines(global_best.code)} lines."
                        ),
                    )

    printer.hdr("DISCOVER_METHOD RUN END")
    printer.brief(
        "final",
        f"best_acc={global_best.accuracy:.6f} rt={global_best.runtime:.3f}s | total_cost={state.money(state.total_cost)}",
    )
    return global_best, history
    if llm_job_count:
        gpu_cycle = list(gpu_ids or [])
        if gpu_cycle:
            llm_idx = 0
            for j in jobs:
                if j.get("job_type") == "llm_init":
                    j["gpu_id"] = gpu_cycle[llm_idx % len(gpu_cycle)]
                    llm_idx += 1
