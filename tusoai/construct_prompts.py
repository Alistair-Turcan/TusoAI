import re
from collections import defaultdict
from typing import List, Dict

# ---------------- Few‑shot exemplar ---------------------------------- #
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
        return text
    except Exception as e:
        print("LLM call failed:", e)
        return ""

# ---------------- Prompt helpers ------------------------------------ #
def make_category_prompt(task_description: str,
                         data_available: str,
                         category: str,
                        to_generate: int,
                        features_available: Dict = "") -> str:
    few_shot = "\n".join(f"<p>{ex}</p>" for ex in regularisation_prompts)

    features_sentence = ""
    if features_available:
        features_sentence = "The available features for this task are: " + "; ".join(
            f"{name} ({desc})" for name, desc in features_available.items()
        ) + "."
        
    return f"""
We are designing an LLM‑powered AutoML system for the task:

    "{task_description}"

    "{features_sentence}"

Current optimisation axis: **{category}**

Below is a style example of prompts for a *regularisation* category for a classification task. Each prompt begins with *by ...* and expresses a specific, actionable optimisation idea:

{few_shot}

You are a master of machine learning and the domain relevant to this task. Keeping the same concise, actionable style, write **exactly {to_generate} distinct prompts** that belong to the **{category}** category **and are appropriate for this task**. 

These should be a mix of general, conceptual, and complex prompts, and not overly specific, similar to the example.

Wrap *each* prompt in its own <p> ... </p> tag.
Return only these <p>...</p> lines, nothing else.
By optimization we mean strictly performance, not runtime, scalability, logging, visualizing, evaluating, etc. Assume the evaluation metrics already exist. We will only have access to {data_available}.
"""

def parse_p_tags(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"<p>(.*?)</p>", text, flags=re.S) if m.strip()]

# ---------------- Main driver to build prompts per category --------- #
def generate_prompts_for_categories(task_description: str,
                                    data_available: str,
                                    categories: List[str],
                                    to_generate: int,
                                    client,
                                    temperature: float = 0.5,
                                    model: str = '') -> Dict[str, List[str]]:
    all_prompts = {}
    for cat in categories:
        prompt = make_category_prompt(task_description, data_available, cat, to_generate)
        print(f"[LLM] Generating prompts for category: {cat}")
        reply = run_prompt(prompt, client=client, temperature=temperature, model=model)
        prompts = parse_p_tags(reply)

        print("Raw reply:\n", reply)
        all_prompts[cat] = prompts
    return all_prompts







import random, re
from typing import List, Dict

# ------------------------------------------------------------
# Few‑shot helper: sample k prompts we already have per category
# ------------------------------------------------------------
def sample_existing(category: str,
                    existing_prompts: Dict[str, List[str]],
                    k: int = 2) -> List[str]:
    pool = existing_prompts.get(category, [])
    random.shuffle(pool)
    return pool[:k]

# ------------------------------------------------------------
# Build the per‑paper prompt ---------------------------------
# ------------------------------------------------------------
def make_summary_prompt(task_description: str,
                        data_available: str,
                        summary: str,
                        categories: List[str],
                        existing_prompts: Dict[str, List[str]],
                        n_new_min: int,
                        n_new_max: int) -> str:
    # build few‑shot block
    few_shot_lines = []
    # Step 1: Sample 5 categories
    sampled_categories = random.sample(categories, k=min(len(categories), 5))
    
    # Step 2: For each, get 1 example
    for cat in sampled_categories:
        ex_list = sample_existing(cat, existing_prompts, k=1)
        if ex_list:
            few_shot_lines.append(f"<c>{cat}</c><p>{ex_list[0]}</p>")
        
    few_shot_block = "\n".join(few_shot_lines) or "[no few‑shot prompts available]"

    categories_line = ", ".join(categories)

    return f"""
We are designing an LLM powered AutoML system for the task:

    "{task_description}"

We will only have access to {data_available}.

Here is a concise summary of the baseline method:
\"\"\"{summary}\"\"\"

Below are style examples of valid prompt lines taken from earlier work:
{few_shot_block}

Your job: generate between {n_new_min} and {n_new_max} new prompts. These will ultimately be assigned into one of the following categories:

{categories_line}

First, generate these prompts, independently of the categories. Second, assign each to it's most relevant category.
For each prompt output a line in this exact format:

<c>CategoryName</c><p>by …</p>

* Every prompt must begin with “by …”.  
* Cover a mix of general, conceptual, and complex ideas.  
* Focus strictly on *performance* optimisation (ignore runtime, scalability, logging, etc.).  
* Return only the `<c>…</c><p>…</p>` lines, nothing else.
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

# ------------------------------------------------------------
# Main driver: per‑summary prompt generation -----------------
# ------------------------------------------------------------
def generate_prompts_from_summaries(task_description: str,
                                    summaries: Dict[str, str],
                                    categories: List[str],
                                    data_available: str,
                                    existing_prompts: Dict[str, List[str]],
                                    n_new_min: int,
                                    n_new_max: int,
                                    client,
                                    temperature: float = 0.5,
                                    model: str = '') -> Dict[str, Dict[str, List[str]]]:
    """
    Returns {paper_title: {category: [new prompts]}}
    """
    results = {}
    for title, summary in summaries.items():
        prompt = make_summary_prompt(task_description=task_description,
                                     data_available=data_available,
                                     summary=summary,
                                     categories=categories,
                                     existing_prompts=existing_prompts,
                                     n_new_min=n_new_min,
                                     n_new_max=n_new_max)
        print(f"[LLM] Generating prompts for: {title}")
        print(prompt)
        reply = run_prompt(prompt, client=client, temperature=temperature, model=model)
        cat_dict = parse_cat_prompts(reply)
        print(cat_dict)

        results[title] = cat_dict
    return results



