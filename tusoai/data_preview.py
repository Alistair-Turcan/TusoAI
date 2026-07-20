import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from tusoai.fs_utils import rmtree_portable
from tusoai.llm import run_prompt

# ---------------------------
# Utilities
# ---------------------------
def _extract_code_block(text: str) -> str:
    """
    Extracts the first fenced code block from LLM output (```python ...``` or ``` ... ```).
    If no block is found, returns the text as-is.
    """
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _first_n(s: str, n: int) -> str:
    """First n characters of s (safe for None)."""
    if s is None:
        return ""
    return s[:n]


def _maybe_parse_filename_from_read_cmd(read_cmd: str) -> Optional[str]:
    """
    Best-effort to extract a filename-like token from a read command.
    Looks for the last quoted string.
    """
    matches = re.findall(r"""(['"])(.*?)\1""", read_cmd)
    if matches:
        # return the last quoted string content
        return matches[-1][1]
    return None



def _read_first_lines(data_path: str, n_lines: int = 10) -> str:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    lines = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _read_cmd_initial_prompt(data_path: str, head_text: str) -> str:
    suffix = Path(data_path).suffix.lower()
    return f"""You are an expert Python data engineer.

Given a dataset path and its first lines, propose a single Python command that loads the file into a variable named df.

Requirements:
- Return exactly ONE line of Python.
- The line must assign into df.
- Prefer pandas unless clearly unsuitable.
- Use the exact data_path above.
- No markdown, no backticks, no explanations.

<data_path>
{data_path}
</data_path>

<file_suffix>
{suffix}
</file_suffix>

<first_10_lines>
{head_text}
</first_10_lines>
"""


def _read_cmd_refine_prompt(data_path: str, head_text: str, prev_cmd: str, err_text: str) -> str:
    return f"""Your previous read command failed.

Return exactly ONE improved Python command that assigns into df.
No markdown, no backticks, no extra text.

<data_path>
{data_path}
</data_path>

<first_10_lines>
{head_text}
</first_10_lines>

<previous_read_cmd>
{prev_cmd}
</previous_read_cmd>

<error>
{_first_n(err_text, 1200)}
</error>
"""


def _extract_read_cmd(text: str) -> str:
    cand = _extract_code_block(text)
    for ln in cand.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def _validate_read_cmd(read_cmd: str, timeout: int = 90) -> Tuple[bool, str]:
    script = (
        "import pandas as pd\n"
        "import numpy as np\n"
        f"{read_cmd}\n"
        "print(type(df).__name__)\n"
    )
    tmp = tempfile.mkdtemp(prefix="infer_read_cmd_")
    try:
        script_path = _write_temp_script(script, tmp, name="validate_read_cmd.py")
        ok, out, err = _run_script_via_subprocess(script_path, timeout=timeout)
        return (ok, out if ok else err)
    finally:
        rmtree_portable(tmp, ignore_errors=True)


def infer_read_cmd_from_data_path(data_path: str, iterations: int = 4) -> str:
    head_text = _read_first_lines(data_path, n_lines=10)
    prompt = _read_cmd_initial_prompt(data_path, head_text)
    prev_cmd = ""

    for _ in range(max(1, iterations)):
        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]
        cmd = _extract_read_cmd(reply)

        if not cmd or "=" not in cmd:
            prompt = _read_cmd_refine_prompt(data_path, head_text, cmd or prev_cmd, "Command was empty or missing assignment.")
            prev_cmd = cmd or prev_cmd
            continue

        ok, msg = _validate_read_cmd(cmd)
        if ok:
            return cmd

        prev_cmd = cmd
        prompt = _read_cmd_refine_prompt(data_path, head_text, cmd, msg)

    raise RuntimeError(f"Could not infer a valid read_cmd for {data_path}")


# ---------------------------
# Code generation prompts
# ---------------------------
def _initial_script_prompt(read_cmd: str) -> str:
    """
    Ask the LLM to generate a COMPLETE runnable Python script that:
      - Imports anything it needs (assume environment has common libs)
      - Reads a dataset using the given read_cmd into a variable named `df`
      - Prints concise, high-signal information useful for the task
      - Writes ONLY code (no prose), inside a single fenced code block
    """
    return f"""You are an expert Python/pandas analyst.

GOAL
- You will write a COMPLETE Python script that runs from the command line.
- It must:
  1) Import whatever libraries you need.
  2) Load the dataset into a variable using the user's read command.
  3) Print concise, high-signal diagnostics that are useful for the user's task.

READ COMMAND (use exactly this):
READ_CMD = {read_cmd!r}

Here's an example of what to print if this data was a dataframe, although the data here can be anything (keep outputs short and skimmable; label sections clearly)
- shape, index info
- column list
- dtypes per column
- missingness summary (per-column % and count of rows with any NA)
- basic numeric ranges (min/max)
- top-3 value counts for categorical-like columns
- memory usage (e.g., df.memory_usage(deep=True).sum())
- small sample: df.head(5)
- any obviously helpful extras for the task (but avoid heavy operations and long dumps)

CONSTRAINTS
- The code must be a COMPLETE script: define `file = ...` using READ_CMD and then print results.
- Keep compute light (e.g., limit value_counts to top 3).
- Print text only to stdout-no files, no plots, no network operations.

RESPONSE FORMAT
- Return ONLY a single Python code block (no commentary outside).
- Do not write any comments.
"""

def _debug_script_prompt(prev_code: str, error_text: str, read_cmd: str) -> str:
    """
    Build a prompt asking the LLM to FIX a previously generated script that failed.
    Includes first 1000 chars of stderr/traceback and guidance about package installation.
    """
    return (
        f"Your previous script errored when executed.\n\n"
        f"READ_CMD = {read_cmd!r}\n"
        "Here is the exact script you produced:\n"
        "```python\n"
        f"{prev_code}\n"
        "```\n\n"
        "Here is the first 1000 characters of stderr / traceback:\n"
        "```\n"
        f"{error_text}\n"
        "```\n\n"
        "Fix the issue and return a COMPLETE replacement Python script (single code block).\n"
        "- You may import any modules available in the environment.\n"
        "- If an error is related to installation, assume the package is not installed and try doing it WITHOUT that specific package.\n"
        "- Keep the script lightweight. Print concise, helpful diagnostics for the task.\n"
        "- The script must load the dataset into `df` using READ_CMD and then print results.\n\n"
        "Return ONLY the fixed script inside one Python code block, with no extra text."
    )


def _refine_script_prompt(prev_stdout: str, prev_code: str, read_cmd: str) -> str:
    """
    Build a prompt asking the LLM to refine the script:
      - add useful information for the task,
      - remove/shorten unhelpful parts,
      - and produce a full runnable script again.
    """
    def _first_n(s: str, n: int) -> str:
        return s[:n] if s else ""

    return (
        "You previously printed the following output (truncated to essentials):\n\n"
        "```\n"
        f"{_first_n(prev_stdout, 20000)}\n"
        "```\n\n"
        f"READ_CMD = {read_cmd!r}\n"
        "REFINE:\n"
        "1) Decide what ADDITIONAL useful information would meaningfully help for the task.\n"
        "2) Decide what information from the previous output is redundant or unhelpful; remove or shorten it, but maintain a description of all information in the file.\n"
        "3) Produce a revised, COMPLETE Python script that:\n"
        "   - Imports what it needs\n"
        "   - Loads the file using READ_CMD\n"
        "   - Prints ONLY the improved, focused set of information\n\n"
        "Return ONLY the full script as a single Python code block. No commentary outside the code block.\n\n"
        "For reference, here is the last script you produced:\n"
        "```python\n"
        f"{prev_code}\n"
        "```"
    )


def _desc_aggregate_prompt(file_label: str, all_outputs: List[str]) -> str:
    """
    Build a prompt asking the LLM to create an aggregated docstring `desc`
    from all successful iteration outputs.
    """
    joined = "\n\n---\n\n".join(all_outputs)
    return (
        "Create a docstring-style description `desc` using only the information below.\n\n"
        f"FILE_LABEL: {file_label!r}\n"
        "AGGREGATED OUTPUTS:\n"
        "```\n"
        f"{joined}\n"
        "```\n\n"
        "Construct `desc` exactly like:\n\n"
        f"\"\"\"\n"
        f"File: {file_label}\n"
        "File overview: <1-2 concise sentences describing the dataset at a high level, based strictly on the filename and what was printed.>\n"
        "Data descriptions:\n"
        "- <Key characteristics, signals, schema patterns, ranges, notable categories, missingness, shape, or other items actually observed.>\n"
        "- <... add bullets as needed but keep it concise.>\n"
        "\"\"\"\n\n"
        "Return ONLY that triple-quoted docstring. No extra text."
    )


def _compress_desc_prompt(desc_block: str) -> str:
    """
    Build a prompt asking the LLM to compress `desc` to information strictly relevant to the task.
    """
    return (
        "Print a concise 1-3 sentence description of this dataset. Think modality, technology, dataset details, etc.:\n\n"
        "Docstring:\n"
        f"{desc_block}\n\n"
        "Return ONLY the triple-quoted docstring, with no extra commentary."
    )


def _write_temp_script(code: str, dirpath: str, name: str = "main.py") -> str:
    """
    Write `code` to a temp script path in `dirpath` and return the full path.
    """
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


def _run_script_via_subprocess(
    script_path: str,
    timeout: int = 1200,
    env_prep: Optional[Callable[[dict], dict]] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
) -> Tuple[bool, str, str]:
    """
    Run a Python script in a subprocess, capturing stdout/stderr.
    Returns: (ok, stdout, stderr).
    """
    cmd = [sys.executable, script_path]
    env = os.environ.copy()
    if env_prep is not None:
        try:
            env = env_prep(env)
        except Exception:
            pass

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            preexec_fn=preexec_fn,
            env=env,
        )
        ok = (result.returncode == 0)
        return ok, (result.stdout or ""), (result.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "", "Error: timed out"
    except MemoryError:
        return False, "", "Error: out of memory"
    except Exception as e:
        return False, "", f"Error: {e}"

def run_llm_script_with_retries(
    code: str,
    retries: int,
    read_cmd: str,
    timeout: int = 1200,
    env_prep: Optional[Callable[[dict], dict]] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
) -> Tuple[bool, str, str, str]:
    """
    Write and run `code` as a temp script. If it fails, ask the LLM to fix it up to `retries` times.
    Returns: (ok, final_stdout, final_stderr, final_code).
    """
    temp_dir = tempfile.mkdtemp(prefix="llm_script_")
    try:
        curr_code = _extract_code_block(code)

        for attempt in range(retries + 1):
            script_path = _write_temp_script(curr_code, temp_dir, name=f"main_attempt{attempt}.py")
            ok, out, err = _run_script_via_subprocess(
                script_path,
                timeout=timeout,
                env_prep=env_prep,
                preexec_fn=preexec_fn,
            )
            if ok:
                return True, out, err, curr_code

            # Prepare debug prompt with first 1000 chars of the error
            debug_prompt = _debug_script_prompt(
                prev_code=curr_code,
                error_text=(err[-1000:] if err else ""),
                read_cmd=read_cmd,
            )
            fixed, cost = run_prompt(debug_prompt)
            curr_code = _extract_code_block(fixed)

        # Failed after retries
        return False, out, err, curr_code
    finally:
        try:
            rmtree_portable(temp_dir)
        except Exception:
            pass


def iterative_desc_from_read_command(
    read_cmd: str,
    iterations: int = 3,
    retries: int = 3,
    timeout: int = 120,
    env_prep: Optional[Callable[[dict], dict]] = None,
    preexec_fn: Optional[Callable[[], None]] = None,
) -> str:
    """
    End-to-end pipeline:
      1) Ask the LLM for a COMPLETE script that reads data with `read_cmd` and prints useful info.
         Run it; on failure, give the LLM up to `retries` attempts to fix (with code + first 1000 chars of error).
      2) Repeat for `iterations` total rounds:
           - Ask what further useful info to print and what to remove.
           - Run the refined script (again with `retries` bug-fix loop).
           - Collect each successful stdout.
      3) Aggregate all outputs and ask the LLM to write a `desc` docstring.
      4) Ask the LLM to compress `desc` to task-relevant info only.
      5) Return the final `desc`.
    """
    # INITIAL script
    prompt0 = _initial_script_prompt(read_cmd)
    code0, cost = run_prompt(prompt0)
    ok, out, err, final_code = run_llm_script_with_retries(
        code=code0,
        retries=retries,
        read_cmd=read_cmd,
        timeout=timeout,
        env_prep=env_prep,
        preexec_fn=preexec_fn,
    )
    if not ok:
        raise RuntimeError(f"LLM script failed after {retries} retries.\n---stderr---\n{err}")

    outputs: List[str] = [out]
    last_stdout = out
    last_code = final_code

    # REFINEMENT rounds
    for it in range(iterations):
        print("Refinement iteration:", it)
        refine_prompt = _refine_script_prompt(
            prev_stdout=last_stdout,
            prev_code=last_code,
            read_cmd=read_cmd,
        )
        new_code, cost = run_prompt(refine_prompt)
        ok, out, err, fixed_code = run_llm_script_with_retries(
            code=new_code,
            retries=retries,
            read_cmd=read_cmd,
            timeout=timeout,
            env_prep=env_prep,
            preexec_fn=preexec_fn,
        )
        if ok:
            outputs.append(out)
            last_stdout = out
            last_code = fixed_code
        else:
            # Skip failed refinement; keep last successful stdout for next iteration.
            continue


    return outputs[-1] #Return last refinement

# ---------------- Few-shot exemplars (generic "data usage" style) ---------------- #
data_usage_exemplars = [
    "by aggregating row-level signals into entity-level features using robust statistics (mean/max/quantiles)",
    "by thresholding confidence scores to create sparse, high-precision binary indicators",
    "by converting the table into a bipartite graph and learning node embeddings from its weighted edges",
    "by normalizing and calibrating raw scores into comparable probabilities across samples or chromosomes",
    "by summarizing multiple links per entity with top-k pooling and distance-aware weighting",
    "by constructing multi-scale features that capture both local (nearby) and distal (long-range) relationships",
    "by encoding uncertainty via soft weights instead of hard assignments (e.g., expectation over links)",
    "by enriching the records with external annotations (IDs, coordinates, ontology) via joins before featurization",
    "by de-duplicating overlapping entities and collapsing them to canonical representatives prior to modeling",
    "by producing contrastive positives/negatives from the table structure to train task-aligned embeddings",
]

# ---------------- Prompt helpers ---------------- #
def _parse_p_tags(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<p>(.*?)</p>", text, flags=re.S) if m.strip()]

def _make_data_usage_prompt(
    task_description: str,
    file_description: str,
    data_stats: str,
    data_usage: str,
    to_generate: int,
    already_have: Optional[List[str]] = None,
) -> str:
    few_shot = "\n".join(f"<p>{ex}</p>" for ex in data_usage_exemplars)


    return f"""
We are designing an LLM-powered AutoML system for the task:

    "{task_description}"

We have a dataset described as:

Short file description:
    "{file_description}"

Comprehensive data_stats (schema + contents summary):
    "{data_stats}"

Goal: produce instructions for how we might be able to **{data_usage}** this data for the task.

Below is a style example of concise, actionable prompts. Each prompt begins with *by ...* and expresses a specific, implementable idea:

{few_shot}

You are a master of machine learning and the domain relevant to this task. Keeping the same concise, actionable style, write **up to {to_generate} distinct and relevant prompts** describing ways to **{data_usage}** this data for the task.

Rules:
- Each prompt MUST start with "by ".
- Keep them focused on how to use/transform this dataset into useful modeling inputs for the task.
- Mix general + conceptual + complex ideas (not overly specific to a single column name unless data_stats clearly suggests it).
- Do NOT include runtime, logging, evaluation, visualization, or generic "clean the data" advice unless it directly enables {data_usage}.
- All ideas must be implementable using the data file provided.

Wrap *each* prompt in its own <p> ... </p> tag.
Return only these <p>...</p> lines, nothing else.
""".strip()

# ---------------- Main driver (single "category": returns a list) ---------------- #
def gather_data_usage_instructions(
    function_name: str,
    task_description: str,
    file_description: str,
    data_stats: str,
    data_usage: str,
    instruction_count: int,
    cache_dir: Optional[str] = None,
    clear: bool = True,
) -> List[str]:
    """
    Generates a list of `instruction_count` concise 'by ...' instructions for how to {data_usage}
    the given dataset for the downstream task.

    Assumes you have a `run_prompt(prompt: str) -> str | tuple` available in scope.
    If cache_dir is provided, caches to: <cache_dir>/kt_data/<function_name>/data_instructions.json
    """
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "kt_data" / function_name / "data_instructions.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if clear and cache_path.exists():
            cache_path.unlink()

        if (not clear) and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            # cached stored as {"instructions":[...]} or just [...]
            if isinstance(cached, dict) and "instructions" in cached:
                return cached["instructions"][:instruction_count]
            if isinstance(cached, list):
                return cached[:instruction_count]

    # Generate, with a couple of retries to reach instruction_count after dedupe
    collected: List[str] = []
    seen_norm = set()

    def _add(items: List[str]) -> None:
        for x in items:
            norm = re.sub(r"\s+", " ", x.strip().lower())
            if not norm or norm in seen_norm:
                continue
            if not x.strip().lower().startswith("by "):
                x = "by " + x.strip()
            collected.append(x.strip())
            seen_norm.add(norm)

    rounds = 0
    while len(collected) < instruction_count and rounds < 3:
        to_generate = max(5, instruction_count - len(collected))
        prompt = _make_data_usage_prompt(
            task_description=task_description,
            file_description=file_description,
            data_stats=data_stats,
            data_usage=data_usage,
            to_generate=to_generate,
            already_have=collected if collected else None,
        )
        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]
        _add(_parse_p_tags(reply))
        rounds += 1

    instructions = collected[:instruction_count]

    additional_instructions = [
        f"by trying to {data_usage} with this data to capture key patterns",
        f"by attempting to {data_usage} with this data into a useful representation",
        f"by exploring ways to {data_usage} with this data using simple aggregations (e.g., top-k, mean/max) over its natural groupings",
        f"by using this data to {data_usage} in a way that is robust to noise and missingness",
        f"by experimenting with multiple formulations to {data_usage} with this data (e.g., binary indicators vs. weighted scores) and keeping what works best",
    ]

    instructions += additional_instructions

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"instructions": instructions}, indent=2),
            encoding="utf-8",
        )
        
    return instructions
