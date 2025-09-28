import re
from typing import Dict, List

classification_categories = ['regularisation', 'feature_engineering',
                             'hyperparameter_tuning', 'sampling',
                             'ensemble_methods', 'calibration', 'feature_selection',
                             'general', 'drastic']


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


def get_task_categories(task_description: str,
                        data_available: str,
                        num_cat: int,
                        client,
                        model: str,
                        temperature: float = 0.5,
                        features_available: Dict = "") -> list:

    features_sentence = ""
    if features_available:
        features_sentence = "The available features for this task are: " + "; ".join(
            f"{name} ({desc})" for name, desc in features_available.items()
        ) + "."
        
    prompt = f"""
We are building an LLM-powered AutoML system for the task:

    "{task_description}"

    "{features_sentence}"

As a reference, some generic categories for optimizing classification models include:
{classification_categories}

You are a master of machine learning and the domain relevant to this task. Please first briefly reason about what kinds of modeling interventions or optimization strategies could be helpful for this specific task. Then propose a list of concise, task-relevant optimization categories.

Your list should include conceptual ideas that are tailored to this task and each should reflect a specific axis of improvement (e.g., architectural choices, preprocessing tricks, domain constraints, evaluation metrics, robustness techniques, etc.).

Output exactly {num_cat} proposed categories, one per line, each enclosed in: <c>Category Name</c>

Do not include any other text, explanation, or formatting. By optimization we mean strictly performance, not runtime, scalability, logging, visualization, post-evaluation, etc. We will only have access to {data_available}.
"""
    reply = run_prompt(prompt, client, temperature=temperature, model=model)

    # Extract <c>Category</c> tags
    matches = re.findall(r"<c>(.*?)</c>", reply)
    categories = [m.strip() for m in matches if m.strip()]

    if not categories:
        print("Warning: No categories extracted from LLM response.")
        print("Raw response:\n", reply)

    return categories


def refine_categories_with_summaries(
    categories: List[str],
    summaries: Dict[str, str],
    task_description: str,
    data_available: str,
    client,
    model: str,
    temperature: float = 0.5,
    features_available: Dict = ""
) -> List[str]:
    """
    Iteratively updates `categories` by reading each paper’s summary and
    querying an LLM about additions or condensations.

    Parameters
    ----------
    categories   : initial list of category names
    summaries    : dict mapping {title: bullet‑point string of methods}
    temperature  : sampling temperature for `run_prompt`
    model        : model name for `run_prompt`

    Returns
    -------
    List[str]    : final, de‑duplicated category list
    """
    current = categories.copy()

    features_sentence = ""
    if features_available:
        features_sentence = "The available features for this task are: " + "; ".join(
            f"{name} ({desc})" for name, desc in features_available.items()
        ) + "."
        
    for title, bullet_points in summaries.items():
        prompt = f"""
We are building an LLM-powered AutoML system for the task:

    "{task_description}"

    "{features_sentence}"

We will curate and refine our categories based on the current categories and a paper.

Current categories:
{current}

Paper: "{title}"
Key method points:
{bullet_points}

TASK ▸
1. If the paper suggests a *new* axis of optimization missing from the list,
   propose a concise category for it.
2. If two or more current categories can be merged, instead give a single name that
   subsumes them.
3. Otherwise, if the category is irrelevant given only {data_available}, leave the list unchanged.

Return **one updated list only**—one category per line,
each wrapped exactly like <c>Category</c>. No other text."""
        reply = run_prompt(prompt, client, temperature=temperature, model=model)

        # pull the LLM’s updated list
        matches = re.findall(r"<c>(.*?)</c>", reply)
        print(matches)
        if matches:
            # keep order while removing duplicates
            seen = set()
            current = [c.strip() for c in matches if not (c.strip() in seen or seen.add(c.strip()))]

    # final safety de‑duplication
    final = []
    seen = set()
    for cat in current:
        if cat not in seen:
            final.append(cat)
            seen.add(cat)

    return final

