import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from tusoai.llm import run_prompt

# ---------------- Few-shot exemplar ---------------------------------- #
regularisation_prompts = [
    "by introducing L1 sparsity constraints to prune features",
    "by subsampling training rows each iteration to inject stochasticity",
    "by shrinking updates with a smaller learning rate for smoother convergence",
    "by refining regularisation strategies",
    "by combining complementary regularisation methods",
    "by adapting regularisation strength across epochs",
    "by scaling regularisation to the dataset size",
    "by combining elastic-net with adaptive polynomial penalties to capture curved relationships",
    "by adding Jacobian norm regularisation to control sharp non-linear gradients",
    "by introducing spectral norm constraints for stable non-linear layers"
]

# ---------------- Prompt helpers ------------------------------------ #
# ---------------- Prompt helpers ------------------------------------ #
def make_category_prompt(
    task_description: str,
    data_available: str,
    category: str,
    to_generate: int,
    features_available: Dict = "",
) -> str:
    few_shot = "\n".join(f"<p>{ex}</p>" for ex in regularisation_prompts)

    return f"""<role>
You are a master of machine learning and the domain relevant to this task. You generate high-signal AutoML "instruction prompts" that describe concrete optimization interventions. You are careful about feasibility given the available data and about staying inside the specified optimization axis.
</role>

<task>
Generate a set of concise, actionable optimization prompts that belong to the given category (optimization axis) and are appropriate for this task and data.
</task>

<style_reference>
Below are style examples of prompts for a regularisation category for a classification task.
Each prompt:
- begins with "by ..."
- is concise and actionable
- expresses a specific optimization intervention

{few_shot}
</style_reference>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<category>
{category}
</category>

<to_generate>
{to_generate}
</to_generate>

<features_available>
{features_available}
</features_available>
</inputs>

<success_criteria>
- Produce up to <to_generate> DISTINCT prompts.
- Every prompt is clearly within the <category> axis (no drifting into other axes).
- Every prompt is feasible given <data_available> (no new labels, sensors, annotations, or external datasets unless clearly already available).
- Prompts target predictive performance (generalization/robustness/calibration/accuracy), not runtime, scalability, logging, visualization, interpretability, or evaluation setup.
- Prompts are not trivial rephrasings; they cover diverse sub-ideas within the same axis.
- Mix of:
  - general conceptual moves,
  - practical techniques,
  - and a few more complex/advanced ideas
  while remaining implementable.
</success_criteria>

<constraints>
- Output ONLY <p>...</p> lines; no other text.
- Each prompt must be wrapped in its own <p> ... </p> tag.
- Each prompt MUST start with "by ".
- Keep each prompt short (generally one clause; avoid multi-sentence prompts).
- Avoid overly specific hyperparameter values unless they are essential and broadly applicable.
- Do not mention tool usage, benchmarking, ablation studies, logging, or "evaluate".
</constraints>

<analysis_step>
Privately:
1) Interpret what <category> means as a search axis for this specific task.
2) Enumerate feasible interventions within that axis that could plausibly improve performance given <data_available>.
3) Write diverse prompts that cover the highest-signal interventions first.
If you are unsure whether an intervention is feasible with <data_available>, do NOT include it.
</analysis_step>

<verification>
Before finalizing:
- Check each prompt starts with "by ".
- Check each prompt is within <category>.
- Check feasibility with <data_available>.
- Remove duplicates / near-duplicates.
- Ensure output contains only <p>...</p> lines.
</verification>
""".strip()


def parse_p_tags(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<p>(.*?)</p>", text, flags=re.S) if m.strip()]

# ---------------- Main driver to build prompts per category --------- #
def generate_prompts_for_categories(
    function_name: str,
    task_description: str,
    data_available: str,
    categories: List[str],
    to_generate: int,
    *,
    cache_dir: str,
    clear: bool = True,
) -> Dict[str, List[str]]:
    """
    If folder/initial_prompts.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/initial_prompts.json (if present) and rebuilds prompts.
    """
    prompts_path = Path(cache_dir) / f"kt_data/{function_name}/initial_prompts.json"

    # Clear cached prompts if requested
    if clear and prompts_path.exists():
        try:
            prompts_path.unlink()
            print(f"[CLEAR] Removed cache: {prompts_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {prompts_path}: {e}")

    # Load cache if available and not clearing
    if (not clear) and prompts_path.exists():
        print(f"[CACHE] Loading prompts from {prompts_path}")
        return json.loads(prompts_path.read_text(encoding="utf-8"))

    def _process_one(cat: str) -> tuple[str, List[str], str]:
        prompt = make_category_prompt(task_description, data_available, cat, to_generate)
        print(f"[LLM] Generating prompts for category: {cat}")
        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]
        prompts = parse_p_tags(reply)
        return (cat, prompts, reply)

    max_workers = max(1, len(categories))

    # Run each category in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, categories))

    # Collect results deterministically in original order
    all_prompts: Dict[str, List[str]] = {}
    for cat, prompts, reply in results:
        all_prompts[cat] = prompts

    # Save to folder/initial_prompts.json
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps(all_prompts, indent=2), encoding="utf-8")
    print(f"[SAVED] Prompts written to {prompts_path}")

    return all_prompts

# ------------------------------------------------------------
# Few shot helper: sample k prompts we already have per category
# ------------------------------------------------------------
def sample_existing(category: str,
                    existing_prompts: Dict[str, List[str]],
                    k: int = 2) -> List[str]:
    pool = list(existing_prompts.get(category, []) or [])
    random.shuffle(pool)
    return pool[:k]

# ------------------------------------------------------------
# Build the per paper prompt ---------------------------------
# ------------------------------------------------------------
def make_summary_prompt(task_description: str,
                        data_available: str,
                        summary: str,
                        categories: List[str],
                        existing_prompts: Dict[str, List[str]],
                        info_per_paper: int) -> str:
    # build few shot block
    few_shot_lines = []
    sampled_categories = random.sample(categories, k=min(len(categories), 5))

    for cat in sampled_categories:
        ex_list = sample_existing(cat, existing_prompts, k=1)
        if ex_list:
            few_shot_lines.append(f"<c>{cat}</c><p>{ex_list[0]}</p>")

    few_shot_block = "\n".join(few_shot_lines) or "[no few shot prompts available]"
    categories_line = ", ".join(categories)

    return f"""<role>
You are a master of machine learning and the domain relevant to this task. You generate ONLY high-signal, feasible AutoML instruction prompts grounded in the task and available data. You are careful about applicability and implementability; prefer keeping potentially useful ideas when they are plausible with the available data.
</role>

<task>
Using the paper summary, propose up to <info_per_paper> NEW optimization prompts (instructions) that are:
1) relevant to the task,
2) feasible given the available data,
3) and assignable to exactly one existing category.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<paper_summary>
{summary}
</paper_summary>

<allowed_categories>
{categories_line}
</allowed_categories>

<style_examples>
Below are valid examples of previously-generated prompt lines (category + prompt style):
{few_shot_block}
</style_examples>

<info_per_paper>
{info_per_paper}
</info_per_paper>
</inputs>

<decision_rule>
For each candidate prompt, output it ONLY IF ALL are true:
- It is a distinct optimization intervention inspired by the paper summary (not a rephrase of an existing example).
- It is directly applicable to <task_description>.
- It is implementable using ONLY <data_available> (no new labels, sensors, annotations, or external datasets unless explicitly included in <data_available>).
- It targets predictive performance (generalization/robustness/calibration/accuracy), not runtime, scalability, logging, visualization, interpretability, or evaluation setup.
- You can confidently assign it to ONE category in <allowed_categories>.
If any of these fail, DO NOT output that prompt.
</decision_rule>

<assignment_rules>
- Choose the single closest category from <allowed_categories>.
- Do NOT invent new categories or rename categories.
- If a prompt could fit multiple categories, pick the best match; if it's ambiguous, do not output it.
</assignment_rules>

<constraints>
- Output ONLY lines in this exact format:
  <c>CategoryName</c><p>by ...</p>
- Each prompt must start with "by " (exactly).
- Each prompt must be short (generally one clause) and actionable.
- Output at most <info_per_paper> lines total (you may output fewer or none).
- No other text, no bullet points, no numbering.
</constraints>

<analysis_step>
Privately:
1) Extract the paper's concrete techniques and assumptions.
2) Filter to those feasible given <data_available>.
3) Translate each into a concise "by ..." intervention that an AutoML system could explore.
4) Assign to the closest allowed category.
5) Discard anything uncertain, data-infeasible, off-task, or low-signal.
</analysis_step>

<verification>
Before finalizing each line:
- Feasibility: does it require data not present in <data_available>? If yes, drop it.
- Relevance: does it clearly improve performance for <task_description>? If doubtful, drop it.
- Category: is the category exactly one from <allowed_categories>? If not, drop it.
- Format: exactly <c>...</c><p>by ...</p> on one line; no extra text.
</verification>
""".strip()


# ------------------------------------------------------------
# Parse <c>...</c><p>...</p> lines ---------------------------
# ------------------------------------------------------------
def parse_cat_prompts(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cat, prompt in re.findall(r"<c>(.*?)</c>\s*<p>(.*?)</p>", text, flags=re.S):
        cat, prompt = cat.strip(), prompt.strip()
        out.setdefault(cat, []).append(prompt)
    return out

def generate_prompts_from_summaries(
    function_name: str,
    task_description: str,
    summaries: List[Dict[str, Any]],
    categories: List[str],
    data_available: str,
    existing_prompts: Dict[str, List[str]],
    info_per_paper: int,
    *,
    cache_dir: str,
    clear: bool = True,
) -> Dict[str, List[str]]:
    """
    If folder/refined_prompts.json exists and clear=False, loads and returns it.
    If clear=True, deletes folder/refined_prompts.json (if present) and rebuilds.

    Returns:
      refined_prompts: Dict[category -> List[prompts]]
      where refined_prompts = existing_prompts + new prompts inferred from paper summaries.

    IMPORTANT:
      - Does NOT add new categories. Any categories not in `categories` are ignored.
    """
    refined_path = Path(cache_dir) / f"kt_data/{function_name}/refined_prompts.json"
    allowed_cats = set(categories)

    if clear and refined_path.exists():
        try:
            refined_path.unlink()
            print(f"[CLEAR] Removed cache: {refined_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {refined_path}: {e}")

    if (not clear) and refined_path.exists():
        print(f"[CACHE] Loading refined prompts from {refined_path}")
        return json.loads(refined_path.read_text(encoding="utf-8"))

    # Start from existing prompts, but keep only allowed categories
    refined_prompts: Dict[str, List[str]] = {
        k: list(v) for k, v in (existing_prompts or {}).items() if k in allowed_cats
    }
    # Ensure all allowed categories exist
    for c in categories:
        refined_prompts.setdefault(c, [])

    def _process_one(item: Dict[str, Any]) -> tuple[str, Dict[str, List[str]]]:
        title = item.get("title", "Untitled")
        summary = item.get("summary", "")

        prompt = make_summary_prompt(
            task_description=task_description,
            data_available=data_available,
            summary=summary,
            categories=categories,
            existing_prompts=refined_prompts,  # snapshot used in prompt text
            info_per_paper=info_per_paper,
        )

        print(f"[LLM] Generating prompts from summary: {title}")
        reply = run_prompt(prompt)
        if isinstance(reply, tuple):
            reply = reply[0]
        print(reply)

        cat_dict = parse_cat_prompts(reply) or {}
        return (title, cat_dict)

    max_workers = max(1, len(summaries))

    # Run each summary in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, summaries))

    # Merge results sequentially (dedupe, enforce allowed categories)
    for title, cat_dict in results:
        if not cat_dict:
            continue

        for cat, new_list in cat_dict.items():
            if cat not in allowed_cats:
                continue
            if not new_list:
                continue

            seen = set(refined_prompts[cat])
            for p in new_list:
                p = (p or "").strip()
                if p and p not in seen:
                    refined_prompts[cat].append(p)
                    seen.add(p)

    refined_path.parent.mkdir(parents=True, exist_ok=True)
    refined_path.write_text(json.dumps(refined_prompts, indent=2), encoding="utf-8")
    print(f"[SAVED] Refined prompts written to {refined_path}")

    return refined_prompts

def _parse_p_tags(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<p>(.*?)</p>", text, flags=re.S) if m.strip()]

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out

def clean_refined_prompts_to_instructions(
    function_name: str,
    refined_prompts: Dict[str, List[str]],
    task_description: str,
    data_available: str,
    instruction_count: int,  # kept for compatibility; not enforced
    *,
    cache_dir: str,
    clear: bool = True,
    max_attempts: int = 3,
) -> Dict[str, List[str]]:
    """
    For each category in refined_prompts, ask the LLM to filter/merge instructions so that:
      - Everything is implementable for the task given the available data
      - Redundancy is minimized (may merge similar instructions)
      - No brand-new ideas are invented (only select/merge from candidates)

    NOTE: This version does NOT enforce returning exactly `instruction_count` instructions.
    `instruction_count` is treated as a soft upper bound / target.

    Cache:
      - folder/final_instructions.json (load if clear=False, rebuild if clear=True)

    Returns:
      final_instructions: Dict[category -> List[str]]
    """
    out_path = Path(cache_dir) / f"kt_data/{function_name}/final_instructions.json"

    if clear and out_path.exists():
        try:
            out_path.unlink()
            print(f"[CLEAR] Removed cache: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {out_path}: {e}")

    if (not clear) and out_path.exists():
        print(f"[CACHE] Loading final instructions from {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    final_instructions: Dict[str, List[str]] = {}
    categories = list(refined_prompts.keys())
    total_cost = 0.0

    def _process_one(cat: str) -> tuple[str, List[str], float]:
        prompts = refined_prompts.get(cat) or []
        prompts = _dedupe_keep_order([p for p in prompts if isinstance(p, str)])

        if not prompts:
            return (cat, [], 0.0)

        candidates_block = "\n".join(f"<p>{p}</p>" for p in prompts)

        base_prompt = f"""<role>
You are a master of machine learning and the domain relevant to this task. You are careful about feasibility: keep instructions that are implementable or plausibly useful with the available data. Reduce redundancy by merging near-duplicates when appropriate (without inventing new ideas).
</role>

<task>
Clean and de-redundant the candidate optimization instructions for ONE category in an LLM-powered AutoML system.
</task>

<inputs>
<task_description>
{task_description}
</task_description>

<data_available>
{data_available}
</data_available>

<category>
{cat}
</category>

<candidate_instructions>
{candidates_block}
</candidate_instructions>
</inputs>

<decision_rule>
Keep or merge candidate instructions ONLY if ALL are true:
1) It is directly relevant to <task_description>.
2) It is implementable using ONLY <data_available> (no new labels, sensors, annotations, or external datasets unless clearly already included).
3) It is truly within this <category> axis (do not drift into other categories).
4) It is likely to improve predictive performance (generalization/robustness/calibration/accuracy), not runtime, scalability, logging, visualization, interpretability, or evaluation setup.
If any condition is uncertain, DROP the instruction.
</decision_rule>

<merging_rules>
- You MAY merge two or more candidate instructions if they are essentially the same intervention or one is a strict subset of another.
- Merges must preserve meaning and remain implementable given <data_available>.
- Do NOT create "new" interventions; merged text must be a faithful consolidation of existing candidates.
- Prefer fewer, higher-signal instructions.
</merging_rules>

<constraints>
- Output ONLY instruction lines; no explanations.
- Each line must be wrapped exactly like: <p>by ...</p>
- Each instruction must start with "by " (exactly).
- Do NOT introduce any instruction that is not supported by the candidate set.
</constraints>

<analysis_step>
Privately:
1) Remove infeasible instructions (anything requiring unavailable data).
2) Remove off-task or low-signal instructions for this category.
3) Cluster near-duplicates / overlaps.
4) Merge clusters into the shortest faithful "by ..." instruction.
5) Output the minimal set that covers the best performance levers for this category.
</analysis_step>

<verification>
Before finalizing:
- Confirm every output instruction is derived from (or a faithful merge of) the provided candidates.
- Confirm feasibility with <data_available>.
- Confirm each line starts with "by ".
- Confirm output contains ONLY <p>...</p> lines with no extra text.
</verification>
""".strip()

        chosen: Optional[List[str]] = None
        cost_accum = 0.0

        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt += "\n\n<constraints>\nREMINDER: Output only <p>by ...</p> lines and nothing else.\n</constraints>"

            reply = run_prompt(prompt)
            if isinstance(reply, tuple):
                reply_text, cost = reply
                cost_accum += float(cost or 0.0)
            else:
                reply_text = reply

            parsed = _dedupe_keep_order(_parse_p_tags(reply_text))

            if parsed:
                chosen = parsed
                break
            else:
                chosen = parsed  # keep last attempt for warning path

        if not chosen:
            print(
                f"[WARN] Could not parse cleaned instructions for category '{cat}' after {max_attempts} attempts; keeping deduped originals."
            )
            return (cat, prompts, cost_accum)

        return (cat, chosen, cost_accum)

    max_workers = max(1, len(categories))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_process_one, categories))

    for idx, cat in enumerate(categories, start=1):
        cat_res = next(r for r in results if r[0] == cat)
        _, chosen, cost_accum = cat_res
        total_cost += cost_accum
        final_instructions[cat] = chosen
        print(f"Cleaned {cat} ({idx}/{len(categories)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_instructions, indent=2), encoding="utf-8")
    print(f"[SAVED] Final instructions written to {out_path}")
    print(f"Total cost: ${total_cost:.6f}")

    return final_instructions
