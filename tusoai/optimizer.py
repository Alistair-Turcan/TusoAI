from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict
import time
import trace
import os
import inspect
import sys
import subprocess
import re
import os
import random
import json


import subprocess
import re
import resource
import sys
from pathlib import Path

def replace_functions(file_path, function_names, replacement_function):
    """Replaces executed functions in the file with identical versions."""
    with open(file_path, "r") as f:
        content = f.read()

    all_functions = extract_functions(content)
    for function in all_functions:
        # Match function name in executed functions
        header = re.match(r"def (\w+)\(", function)
        if header and header.group(1) in function_names:
            function_clean = function.strip()
            content = content.replace(function_clean, replacement_function)

    # Write the modified content back to the file
    with open(file_path, "w") as f:
        f.write(content)


def run_prompt(prompt, client, temperature, model):
    if model == 'gpt-5':
        temperature=1.0
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract full content from the assistant's reply
        text = resp.choices[0].message.content.strip()
        #print(resp.usage)
        #print(prompt)
        return text
    except Exception as e:
        print("LLM call failed:", e)
        return ""
        
def set_memory_limit():
    # Set memory limit to 50GB (50 * 1024 * 1024 KB)
    limit_in_bytes = 50 * 1024 * 1024 * 1024
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
def run_and_evaluate(script_name, timeout=1200, val_limit=1.0):
    command = [sys.executable, script_name]
    env = _prepend_lib_paths(os.environ)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            preexec_fn=set_memory_limit,
            env=env,
        )

        stdout = result.stdout
        #print(stdout)
        # --- Core Outputs ---
        # Validation accuracy
        evaluate_match = re.search(r"tuso_evaluate:\s*(-?[\d.]+)", stdout)
        evaluation = float(evaluate_match.group(1)) if evaluate_match else "Function must return accuracy only."

        # Model diagnostics block
        diagnostics_info_match = re.search(
            r"tuso_model_start\n(.*?)\ntuso_model_end", stdout, re.DOTALL)
        model_info = diagnostics_info_match.group(1).strip() if diagnostics_info_match else ""

        if len(model_info)>2000:
            model_info=model_info[0:2000]
        if isinstance(evaluation, str):
            return evaluation

        if evaluation >= val_limit:
            return "Error: validation accuracy is suspiciously high. Some overfitting is likely to have occurred."
        return {
            "evaluation": evaluation,
            "model_info": model_info,
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

import shutil
import os

def copy_file(source, index):
    if not os.path.exists(source):
        print(f"Error: {source} does not exist.")
        return
    
    file_name, file_ext = os.path.splitext(source)
    new_file = f"{file_name}_{index}{file_ext}"
    shutil.copy(source, new_file)
    return new_file


def extract_functions(content):
    """Extracts all function definitions from the file, handling extra newlines."""
    function_pattern = re.compile(r"(?:^|\n)(def .*?:\n(?:\s+.*\n*)*)", re.MULTILINE)
    functions = function_pattern.findall(content)
    return [func.rstrip() for func in functions]  

def extract_function_by_name(file_path, function_name):
    """Extracts a specific function by name from the file."""
    with open(file_path, "r") as f:
        content = f.read()
    
    all_functions = extract_functions(content)
    for function in all_functions:
        if function.startswith(f"def {function_name}("):
            return function.strip()
    
    return None  # Return None if the function is not found



import re

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

    code_block: str | None = None

    # Case 1: Use fenced code block only if BOTH markers exist.
    if start_marker in result and end_marker in result:
        try:
            code_block = result.split(start_marker, 1)[1].split(end_marker, 1)[0]
        except IndexError:
            code_block = None  # Fall through to fallback behavior if splitting is odd.

    # Case 2 (fallback): find region starting at 'def tuso_model' and end at next top-level line
    if code_block is None:
        lines = result.splitlines()
        start_idx = None
        for i, line in enumerate(lines):
            if (
                line.startswith("def tuso_model")
                or line.startswith("import")
                or line.startswith("from")
            ):
                start_idx = i
                break
        if start_idx is None:
            return ""  # Cannot locate fallback start; nothing to extract.

        def_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

        # Find first subsequent *non-empty* line whose indent <= def's indent (new top-level)
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip() == "":
                continue
            indent = len(lines[j]) - len(lines[j].lstrip())
            if indent <= def_indent:
                end_idx = j
                break

        code_block = "\n".join(lines[start_idx:end_idx])

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



# ----------------------------------------------------------------------------------------------------

__all__ = [
    "discover_algorithm",
    "Island",
    "ModelRecord",
]


@dataclass
class ModelRecord:
    """Container for one function variant (the *individual* in GA terms)."""

    code: str
    file: Path
    accuracy: float
    model_info: Dict | None = None
    lineage: str = "root"  # free‑form tag, e.g. "gen0_init_logistic_regression"

    def __hash__(self) -> int:  # allow putting ModelRecord into a set
        return hash(self.code)


@dataclass
class Island:
    """A sub‑population in the island model."""

    id: int
    models: List[ModelRecord] = field(default_factory=list)

    # convenience – best model cached on demand
    def champion(self) -> ModelRecord:
        return max(self.models, key=lambda m: m.accuracy)


# ----------------------------------------------------------------------------------------------------
# Genetic algorithm entry point
# ----------------------------------------------------------------------------------------------------
def generate_feedback(new_code, old_code, new_perf, old_perf, client, temperature, model):
    prompt = f'We attempted to optimize this function:\n {old_code}\n Here is the proposed optimization:\n {new_code}\n'
    prompt += 'Write a concise one line summary of the differences between the original function and the proposed optimization. It should be as short as possible while summarizing the differences.'

    #print(prompt)
    response = run_prompt(prompt, client=client, temperature=temperature, model=model)

    feedback = response
    denom = abs(old_perf) if abs(old_perf) > 0 else 1e-10
    perf_percent = 100.0 * (new_perf - old_perf) / denom
    
    if perf_percent>0:
        feedback += f'\nThis increased model performance by {perf_percent:.4f}%.'
    elif perf_percent<0:
        feedback += f'\nThis reduced model performance by {perf_percent:.4f}%.'
    else:
        feedback += '\nThis did not alter model performance.'
    return feedback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def cluster_functions(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    clustered = {i: [] for i in range(n_clusters)}
    for text, label in zip(texts, labels):
        clustered[label].append(text)

    return clustered
def discover_algorithm(
    llm_model,
    temp: float,
    prompts: Dict[str, List[str]],
    probabilities: Dict[str, float],
    reference_filename: str,
    client,
    *,
    initialisations: List[str],
    n_generations: int = 5,
    children_per_model: int = 2,
    timeout: int = 300,
    bug_retries: int = 2,
    initial_bug_fix_attempts: int = 5,
    n_feedback_buffer: int = 5,
    skip_timeout=False,
    prompt_samples=5,
    alter_info_samples=5,
    alter_info_prompts: List[str],
    drop_island_iter = 5,
    prompt_decay=1.1,
    hints=[],
    filename='',
    use_initial=False,
    TIME_LIMIT = 600,
    task_description="",
    val_limit=1.0,
    debug = False
) -> Tuple[ModelRecord, Dict[int, Dict[int, List[ModelRecord]]]]:
    """Run the island‑based GA and return *(global_best, full_history)*.

    Parameters
    ----------
    llm_model, temp, prompts, probabilities
        Same meaning as in your existing ``optimize_function``.
    reference_filename : str
        Path to the baseline Python file that already contains the target function **signature**.  Its
        implementation is *not* assumed to be good – we only use it to grab the name/signature for the LLM.
    initialisations : list[str]
        Sentences like ``"logistic regression"`` that seed the first generation.
    n_generations : int, default 5
        How many *global* generations (island → cluster → selection) to run.
    children_per_model : int, default 2
        Number of children each model produces inside its island every generation.
    timeout, bug_retries, n_feedback_buffer
        Execution and prompting knobs carried over from the original code.

    Returns
    -------
    best_model : ModelRecord
        The single best function discovered during the whole run.
    history : dict[island_id → dict[generation → list[ModelRecord]]]
        Complete evolutionary trace for later analysis.
    """

    # ----------------------------------------------------------------------------------------------
    # 0. House‑keeping helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    base_fn_code = extract_function_by_name(reference_filename, "tuso_model")  # type: ignore
    base_path = Path(reference_filename).resolve().parent
    attempts: set[str] = {base_fn_code}
    feedbacks: List[str] = []

    def _write_and_evaluate(func_code: str, tag: str) -> Optional[ModelRecord]:
        """Persist *func_code* under a unique filename, run it, return ModelRecord or *None* on failure."""
        safe_tag = re.sub(r"[^a-zA-Z0-9_/]+", "_", tag)  # allow forward slash
        dst = base_path / f"{safe_tag}.py"
        src: Path = Path(reference_filename)
        shutil.copy(src, dst)  # we only overwrite *this* destination – tag keeps it unique
        replace_functions(dst, ["tuso_model"], func_code)  # type: ignore

        metrics = run_and_evaluate(dst, timeout, val_limit)
        if not isinstance(metrics, dict):
            print("Error in running")
            #print(str(metrics))
            #print(metrics)
            return str(metrics)  # invalid candidate (runtime or evaluation failure)

        print(dst, tag, metrics["evaluation"])
        return ModelRecord(
            code=func_code,
            file=dst,
            accuracy=metrics["evaluation"],
            model_info=metrics.get("model_info"),
            lineage=tag,
        )

    from typing import Optional, Tuple
    import random
    
    # prompt categories reused from the user code – we can treat them as constants here
    prompt_categories = {"prompt": 0.8, "model": 0.1, "alter_info": 0.1}
    
    if alter_info_prompts is None:
        # hard-set distribution to 1.0, 0, 0
        prompt_categories = {"prompt": 1.0, "model": 0.0, "alter_info": 0.0}
    else:
        # default distribution
        prompt_categories = {"prompt": 0.8, "model": 0.1, "alter_info": 0.1}

    feedbacks: dict[str, list[str]] = defaultdict(list)   # e.g. {"add_feature": [...], "model": [...]}

    def _make_child(parent: "ModelRecord") -> Optional["ModelRecord"]:
        """Produce a mutated child via one or two LLM calls.
    
        Behavioural change:
        -------------------
        * When the randomly‑selected category is ``alter_info`` we *first* ask the LLM
          to improve / add diagnostic print statements (as before).
        * If that pass succeeds we *immediately* run a **second** pass with the
          ``model`` category so that the fresh diagnostics are actually used to
          propose model / feature improvements.
        * Feedback is only generated for passes that are *not* ``alter_info`` so we
          do not pollute the feedback buffer with purely diagnostic changes.
        * If the optimisation pass fails we fall back to returning the first child
          (print‑update only) so the search can still progress.
        """
        nonlocal attempts, feedbacks
    
        # ---------------------------------------------------------------------
        # Helper: choose an initial category, honouring the feedback buffer size
        # ---------------------------------------------------------------------
        def _choose_primary_category() -> str:
            return random.choices(
                list(prompt_categories),
                weights=prompt_categories.values(),
                k=1
            )[0]
        
        # ---------------------------------------------------------------------
        # Helper: run a *single* mutation   (prompt → suggestion → evaluation)
        # ---------------------------------------------------------------------
        def _single_pass(src_parent: "ModelRecord", category: str) -> Tuple[Optional["ModelRecord"], str]:
            """Return ``(candidate, error_msg)`` where ``candidate`` is ``None`` if the
            pass failed completely and ``error_msg`` contains the last error seen."""
            ptype = None
            feedback_key: str = ""
            
            # ------ build the prompt ------------------------------------------------------------
            if category == "prompt":

                # choose a prompt-type and remember its bucket key
                ptype = random.choices(
                    list(probabilities),
                    weights=probabilities.values(),
                    k=1
                )[0]
                feedback_key = ptype
    
                sampled_prompts = random.sample(
                    prompts[ptype],
                    k=min(prompt_samples, len(prompts[ptype]))
                )
                prompt_options = "\n\n".join(
                    f"Option {i+1}:\n{p}" for i, p in enumerate(sampled_prompts)
                )
    
                prompt_body = (
                    "by choosing one of the following strategies to guide optimisation, "
                    f"based on your assessment of what will most improve this model for {task_description}:\n"
                    f"{prompt_options}"
                )
            elif category in {"model", "alter_info"}:
                # if we have no diagnostic info yet, force an alter_info pass first
                if not src_parent.model_info:
                    category = "alter_info"
    
                if category == "alter_info":
                    feedback_key = ""       # no feedback for diagnostic passes
                    sampled = random.sample(alter_info_prompts, alter_info_samples)
                    alter_str = "\n\n".join(
                        f"Option {i+1}:\n{p}" for i, p in enumerate(sampled)
                    )
    
                    prompt_body = (
                        "by choosing one of the following strategies to print diagnostic information, "
                        f"based on your assessment of what will be most informative for optimisation of this model for {task_description}. "
                        "Ensure the information printed is concise enough to be used in an LLM prompt:\n"
                        f"{alter_str}"
                    )
                else:                        # "model"
                    feedback_key = "model"
                    prompt_body = (
                        f"by assessing this diagnostic info and proposing model/feature improvements for this model for {task_description}:\n"
                        f"{src_parent.model_info}"
                    )

            # -------- append the most-recent feedback for this key --------------------------------
            if feedback_key:
                recent_fb = feedbacks[feedback_key][-n_feedback_buffer:]
                if recent_fb:
                    fb_block = "\n".join(recent_fb)
                    prompt_body += (
                        "\n\nAdditionally, consider the following feedback from earlier attempts "
                        f"that used this same optimisation strategy:\n{fb_block}"
                    )
            base_prompt = (
                f"Optimize this model for {task_description}'s performance "
                f"{prompt_body}\n"
                f"Hints:\n- " + "\n- ".join(hints) + "\n"
                f"{src_parent.code}\n"
                "Output only python code, and do not include comments."
            )


            ##print(prompt_body)
            suggestion: str = ""
            error_msg: str = ""
            for i in range(bug_retries):
                final_prompt = base_prompt if not error_msg else (
                    f"Fix this function:\n{suggestion}.\nHere's the error: {error_msg}\nIgnore warnings. If an error is related to installation, assume the package is not installed and try doing it without that specific package.\n"
                    "Output only python code, and do not include comments."
                )
                #print(final_prompt)
                if error_msg == 'Error: timed out' and skip_timeout:
                    print("Skipping timeout")
                    break
                #print(final_prompt)
                reply = run_prompt(final_prompt, client=client, temperature=temp, model=llm_model)
                #print(reply)
                if not reply:
                    print('LLM failed')
                    continue
                #print(reply)
                suggestion = extract_function(reply)
                if not suggestion or suggestion in attempts:
                    print("No or duplicate function")
                    return None, "duplicate or extraction failure"
                attempts.add(suggestion)
    
                tag = f"{src_parent.file.stem}X"  # we'll replace gen later
                cand = _write_and_evaluate(suggestion, tag)
                
                def update_probabilities(probabilities, ptype, factor):
                    """Multiply given ptype probability by factor and renormalize."""
                    probabilities[ptype] *= factor
                    total = sum(probabilities.values())
                    for key in probabilities:
                        probabilities[key] /= total
                    #print(probabilities)
                
                
                if not isinstance(cand, ModelRecord):
                    error_msg = cand  # continue bug-fix loop
                    if debug:
                        print(error_msg)
                
                    # --- NEW: if on final retry, update ptype weightings too ---
                    if i == bug_retries - 1 and prompt_decay != 1.0 and category == "prompt" and ptype is not None:
                        update_probabilities(probabilities, ptype, 1 / prompt_decay)
                
                    continue
                
                # --- NEW: update ptype weight based on success/failure ---
                if prompt_decay != 1.0 and category == "prompt" and ptype is not None:
                    improved = cand.accuracy > src_parent.accuracy
                    factor = (prompt_decay) if improved else (1 / prompt_decay)
                    update_probabilities(probabilities, ptype, factor)
    
                    
                # store feedback unless this was a pure diagnostic pass
                if feedback_key:
                    fb = generate_feedback(
                        new_code=suggestion,
                        old_code=src_parent.code,
                        new_perf=cand.accuracy,
                        old_perf=src_parent.accuracy,
                        client=client,
                        temperature=temp,
                        model=llm_model,
                    )
                    feedbacks[feedback_key].append(fb)

                return cand, ""
    
            return None, error_msg or "retries exhausted"
    
        # ------------------------- primary pass -----------------------------------------------
        primary_category = _choose_primary_category()
        child, _ = _single_pass(parent, primary_category)
        if child is None:
            return None  # nothing useful came out of the first attempt
    
        # ------------------------- optional second pass ---------------------------------------
        if primary_category == "alter_info":
            # now that the code prints diagnostics, immediately attempt to exploit them
            optimised_child, _ = _single_pass(child, "model")
            if optimised_child is not None:
                return optimised_child  # best‑case: diagnostics + optimisation
            # otherwise fall through – at least we have better diagnostics now
    
        return child  # either we already optimised or we did an alter_info‑only improvement
    

    # ----------------------------------------------------------------------------------------------
    # 1. Seed initial islands –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    islands: List[Island] = []
    history: Dict[int, Dict[int, List[ModelRecord]]] = {}
    global_models: List[ModelRecord] = []


    if use_initial:
        print("0. TRYING BASE FUNCTION AS INITIAL ISLAND")
        suggestion = base_fn_code
        if False: #suggestion in attempts:
            print("Base function already tried—skipping")
        else:
            attempts.add(suggestion)
            tag = f"{filename}_initial"
            record = _write_and_evaluate(suggestion, tag)
            if isinstance(record, ModelRecord):
                # give it a special island id of -1 (or choose any convention you like)
                island = Island(id=-1, models=[record])
                islands.append(island)
                history.setdefault(island.id, {})[0] = [record]
                global_models.append(record)
            else:
                print(f"Base function eval failed: {record!r}")
    
    print("1. SEEDING INITIAL ISLANDS")
    for idx, init in enumerate(initialisations):
        # ---------- prime the conversation ----------
        base_prompt = (
            f"Write a basic version of this model for {task_description} using "
            f"{init}.\n"
            f"Hints:\n- " + "\n- ".join(hints) + "\n"
            f"\n{base_fn_code}\n"
            "Output only python code, and do not include comments."
        )
    
        suggestion: str = ""          # current code candidate
        error_msg: str = ""           # error string from last evaluation
        
        for try_idx in range(initial_bug_fix_attempts):
            prompt = base_prompt if not error_msg else (
                f"Fix this function:\n{suggestion}.\n"
                f"Here's the error: {error_msg}\n"
                "Ignore warnings. If an error is related to installation, "
                "assume the package is not installed and try doing it without that specific package.\n"
                "Output only python code, and do not include comments."
            )
            #print(prompt)
            if debug:
                print(error_msg)
            #print(prompt)
            if error_msg == "Error: timed out" and skip_timeout:
                print("Skipping timeout")
                break
    
            reply = run_prompt(prompt, client=client, temperature=temp, model=llm_model)
            #print(reply)
            if not reply:
                print("LLM failed to reply")
                break    # give up on this initialisation
    
            suggestion = extract_function(reply)
            if not suggestion or suggestion in attempts:
                print("No function extracted or duplicate function")
                break    # unrecoverable for this initialisation
            attempts.add(suggestion)
    
            tag = f"{filename}{idx}_{re.sub(r'[^a-zA-Z0-9]+', '_', init)[:20]}"
            record = _write_and_evaluate(suggestion, tag)
    
            if isinstance(record, ModelRecord):
                # ---------- success ----------
                island = Island(id=idx, models=[record])
                islands.append(island)
                history.setdefault(island.id, {})[0] = [record]
                global_models.append(record)
                break      # done with this initialisation
            else:
                # ---------- keep trying ----------
                error_msg = record
                continue   # iterate bug-fix loop


    if not islands:
        raise RuntimeError("No valid initial islands – all seed evaluations failed.")

    # ----------------------------------------------------------------------------------------------
    # 2‑3. Evolutionary loop ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    global_best = max((mdl for isl in islands for mdl in isl.models), key=lambda m: m.accuracy)
    
     
    start_time = time.time()
    
    print("2. EVOLVING")
    for gen in range(1, n_generations + 1):
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT:
            print("Time limit exceeded. Stopping evolution.")
            break
        print("Generation: ", gen)
        # --- 2. In-island recursive child creation ------------------------------------------------
        for isl in islands:
            print("Island")
            all_new_children: List[ModelRecord] = []
            current_parent = isl.models[0]
            for i in range(children_per_model):
                new_children: List[ModelRecord] = []
                child = _make_child(current_parent)
                if child:
                    child.lineage = f"{gen}{isl.id}{i+1}"
                    new_children.append(child)
                    current_parent = child
                else:
                    print("failed to make child")
                isl.models.extend(new_children)
                all_new_children.extend(new_children)
                global_models.append(child)

            history.setdefault(isl.id, {})[gen] = all_new_children

        # --- 3. Archipelago selection via clustering -------------------------------------------
        all_models: List[ModelRecord] = [m for isl in islands for m in isl.models]
        all_codes   = [m.code for m in global_models if m and m.code]
        
        # drop islands every drop_island_iter gens
        current_num_islands = len(islands)
        target_clusters = (
            current_num_islands - 1
            if gen % drop_island_iter == 0 and current_num_islands > 1
            else current_num_islands
        )
        
        print("Clustering among", len(all_codes), "potential models")
        clustered = cluster_functions(all_codes, n_clusters=target_clusters)
        
        # quick lookup: code-string → ModelRecord
        code_to_model: dict[str, ModelRecord] = {m.code: m for m in all_models}
        
        new_islands: List[Island] = []
        for new_id, cluster in enumerate(clustered.values()):
            if isinstance(cluster, tuple):
                _, codes_in_cluster = cluster
            else:
                codes_in_cluster = cluster
        
            cluster_models = [code_to_model[c] for c in codes_in_cluster if c in code_to_model]
            if not cluster_models:
                continue
        
            # Find best accuracy
            best_acc = max(m.accuracy for m in cluster_models)
        
            # Filter for models within 0.01% of the best accuracy
            
            close_models = []
            for m in cluster_models:
                if best_acc == 0:
                    # both best and candidate are 0 → keep it
                    if m.accuracy == 0:
                        close_models.append(m)
                else:
                    if (
                        abs(best_acc - m.accuracy) / abs(best_acc) < 0.001
                        or abs(best_acc - m.accuracy) < 0.001
                    ):
                        close_models.append(m)


        
            # Among those, pick the one with the shortest code
            champ = min(close_models, key=lambda m: len(m.code))
            new_islands.append(Island(id=new_id, models=[champ]))
        
        islands = new_islands
        best_in_gen = max((isl.champion() for isl in islands), key=lambda m: m.accuracy)
        if best_in_gen.accuracy > global_best.accuracy:
            global_best = best_in_gen
            
                    
    return global_best, history
