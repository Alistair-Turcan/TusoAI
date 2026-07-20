from __future__ import annotations


def _target_def_name(name: str) -> str:
    return str(name).split(".")[-1].strip()


def _target_contract(name: str, code: str) -> str:
    target_name = _target_def_name(name)
    text = code or ""
    if any(line.lstrip().startswith(f"class {target_name}") for line in text.splitlines()):
        return f"Ensure the target class is defined as: class {target_name}(...): or class {target_name}:"
    return f"Ensure the target method is defined as: def {target_name}(...):"


def build_seed_base_prompt(
    *,
    task_description: str,
    init_idea: str,
    hint_block: str,
    base_fn_code: str,
    fn: str,
) -> str:
    target_contract = _target_contract(fn, base_fn_code)
    return f"""
<role>
You are an expert scientific method developer and senior Python engineer.
You design robust, high-performing methods under external evaluation.
</role>

<task>
Create a working baseline implementation for the method for:
{task_description}
</task>

<context>
You are editing code inside an existing codebase. The evaluation harness is external and must not be modified.
You MAY add imports, helper functions/classes, and additional utilities if necessary.
Assume many packages may already be available.
</context>

<hints>
{hint_block}
</hints>

<current_code>
{base_fn_code}
</current_code>

<success_criteria>
- Runs successfully in the existing environment
- Preserves compatibility with likely call sites (do not break the function's callable contract)
- Implements the idea concretely (no placeholders/TODOs)
- Avoids "feature flags": do NOT add boolean parameters or conditions that default to disabling the new behavior
- {target_contract}
</success_criteria>

<initial_idea>
Incorporate this approach/idea as the starting point:
{init_idea}
</initial_idea>

<output_format>
Return exactly two sections:

<plan>
1) Briefly state what you will change and why (5-12 lines)
2) Mention any added imports/helpers
</plan>

BEGIN_CODE
END_CODE
</output_format>

<constraints>
- Do NOT output Markdown or code fences
- Do NOT modify external evaluation logic
- Do NOT add "optional=False" style toggles or new booleans that gate the improvement
</constraints>

<verification>
Before final output:
- Does the plan match the code?
- Will the new behavior run by default (no disabled-by-default toggles)?
- Is the code complete and runnable?
</verification>
""".strip()


def build_seed_repair_prompt(
    *,
    task_description: str,
    error_msg: str,
    suggestion: str,
    fn: str,
) -> str:
    target_contract = _target_contract(fn, suggestion)
    return f"""
<role>
You are an expert scientific method developer and senior Python engineer.
You debug and repair code while preserving intended method behavior.
</role>

<task>
Fix the code so it runs and matches the intended baseline behavior for:
{task_description}
</task>

<context>
Assume many packages may already be available. If the error indicates a missing import/dependency,
replace it with an alternative approach that is likely available.
Preserve compatibility with call sites.
Do not modify external evaluation logic.
</context>

<success_criteria>
- Fixes the reported error
- Keeps the intended baseline behavior (do not "cheat" by removing core logic)
- No disabled-by-default feature flags (avoid new booleans gating behavior)
- {target_contract}
</success_criteria>

<error>
{error_msg}
</error>

<broken_code>
{suggestion}
</broken_code>

<output_format>
<plan>
1) Root cause of the error (1-2 lines)
2) Exact fix you'll apply (2-6 lines)
</plan>

BEGIN_CODE
END_CODE
</output_format>

<constraints>
- No Markdown, no backticks
- Avoid introducing new optional boolean flags or default-false toggles
</constraints>

<verification>
Confirm the specific error is addressed and the method still runs by default.
</verification>
""".strip()


def build_mutation_base_prompt(
    *,
    task_description: str,
    fn_name: str,
    kind: str,
    ctx_block: str,
    prompt_body: str,
    hint_block: str,
    src_code: str,
) -> str:
    target_def_name = _target_def_name(fn_name)
    target_contract = _target_contract(fn_name, src_code)
    return f"""
<role>
You are an expert scientific method developer and senior Python engineer.
You improve algorithms under an external evaluation harness.
</role>

<context>
Task: {task_description}
Editing target: {target_def_name}
Task kind: {kind}
{ctx_block}
You MAY add imports, helper functions/classes, and utilities if needed.
Assume many packages may already be available; if a missing dependency causes an error, adapt accordingly.
External evaluation is done elsewhere; do not modify evaluator code.
</context>

<goal>
Improve evaluation score for this method.
Make changes that are scientifically motivated and likely to move the metric.
Avoid broad refactors unless necessary for the improvement.
</goal>

<hints>
{hint_block}
</hints>

<constraints>
- Output must follow <output_format> exactly
- Feedback in <f> must describe concrete code changes/reasoning only; do not mention option IDs/numbers
- You may add imports/helpers above the target function
- Do not modify external evaluation logic
- If you introduce new parameters/configurations, defaults MUST differ from the current implementation
- Do NOT add feature flags (no new booleans / default-false toggles that gate behavior)
- No file I/O, no networking, no subprocess calls
- print() is allowed ONLY if the selected strategy explicitly requests it (diagnostic/ablation). Otherwise, no prints.
- {target_contract}
</constraints>

{prompt_body}

<current_code>
{src_code}
</current_code>

<output_format>
<plan>
1) Hypothesis: why this change should improve the metric (2-4 lines)
2) Step-by-step plan (3-8 bullets)
</plan>

<f>
Briefly explain what the previous code did, what you changed, and why that change should help (2-5 lines). Do NOT reference option numbers or quote the option list.
</f>

BEGIN_CODE
END_CODE
</output_format>

<verification>
Before output:
- Does the code implement the plan?
- Will the improvement run by default (no disabled-by-default toggles)?
- Are prints present only when requested by this pass?
- Is the code self-contained for this codebase?
</verification>
""".strip()


def build_mutation_repair_prompt(
    *,
    task_description: str,
    category: str,
    error_msg: str,
    suggestion: str,
    src_code: str,
    hint_block: str,
    fn_name: str,
) -> str:
    target_contract = _target_contract(fn_name, src_code or suggestion)
    return f"""
<role>
You are an expert scientific method developer and senior Python engineer.
You repair optimized code while preserving the intended improvement.
</role>

<task>
Fix the code so it runs under the existing project setup and preserves the optimization intent:
Improve evaluation for: {task_description}
</task>

<hints>
{hint_block}
</hints>

<context>
Assume many packages may already be available. If the error indicates missing dependency/import, adapt using an alternative that fits the current environment.
Do not modify external evaluation logic.
If this is not a diagnostic/ablation pass, remove prints.
</context>

<constraints>
- Output must follow <output_format> exactly
- Feedback in <f> must describe concrete code changes/reasoning only; do not mention option IDs/numbers
- No Markdown, no backticks
- Avoid feature flags / default-false toggles that would disable the improvement
- No file I/O, no networking, no subprocess
- {target_contract}
</constraints>

<original_reference_code>
{src_code}
</original_reference_code>

<error>
{error_msg}
</error>

<broken_code>
{suggestion}
</broken_code>

<output_format>
<plan>
1) Root cause (1-2 lines)
2) Minimal fix that preserves the improvement (2-6 lines)
</plan>

<f>
Briefly explain what the previous code did, what this repaired version changes, and why this should help (2-5 lines). Do NOT reference option numbers or quote the option list.
</f>

BEGIN_CODE
END_CODE
</output_format>

<verification>
Ensure the specific error is fixed and the improved behavior still runs by default.
</verification>
""".strip()
