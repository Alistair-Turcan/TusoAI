import re
from typing import List

# ---------------- Few‑shot exemplar ---------------------------------- #
# Generic starters for a *classification* context ─ adapt / extend as you like
classification_initializations = [
    "logistic regression",
    "XGBoost",
    "random forest",
    "MLP classifier"
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


# ---------------- Prompt builder ------------------------------------ #
def make_initialization_prompt(task_description: str,
                               data_available: str,
                               num_init: int) -> str:
    few_shot = "\n".join(f"<m>{ex}</m>" for ex in classification_initializations)
    return f"""
We are designing an LLM‑powered AutoML system for the task:

    "{task_description}"

Below is an example list of generic model initializations for a **classification** task:
{few_shot}

You are a master of machine learning and of the domain relevant to this task.
Propose **exactly {num_init} concise model initializations** that could serve as
starting baselines **for this specific task** given that we only have
{data_available}. They should be general task-specific methods, model families, or high‑level architectural
descriptions, not fully‑specified pipelines.

Output one per line, each wrapped in <m> ... </m> tags.
Return *only* these <m>...</m> lines — no explanations, no extra text.
"""

def parse_m_tags(text: str) -> List[str]:
    """Extract <m> ... </m> contents, stripping whitespace."""
    return [m.strip() for m in re.findall(r"<m>(.*?)</m>", text, flags=re.S) if m.strip()]

# ---------------- Driver -------------------------------------------- #
def get_initializations(task_description: str,
                        data_available: str,
                        num_init: int,
                        client,
                        temperature: float = 0.5,
                        model: str = '') -> List[str]:
    """
    Use an LLM to draft an initialization list for the given task / data.
    """
    prompt = make_initialization_prompt(task_description, data_available, num_init)
    reply = run_prompt(prompt, client=client, temperature=temperature, model=model)

    # Debug / inspection
    print("Raw LLM reply:\n", reply)

    inits = parse_m_tags(reply)

    if not inits:
        print("Warning: No initializations extracted from LLM response.")

    # De‑duplicate while preserving order
    seen = set()
    unique_inits = [m for m in inits if not (m in seen or seen.add(m))]
    return unique_inits




import re
from typing import Dict, List

def refine_initializations_with_summaries(
    initializations: List[str],
    summaries: Dict[str, str],
    task_description: str,
    data_available: str,
    client,
    temperature: float = 0.5,
    model: str = '',
) -> List[str]:
    """
    Iterate through each paper summary, asking the LLM whether to add a new
    initialization or consolidate existing ones. Returns a de‑duplicated list.

    Parameters
    ----------
    initializations : List[str]
        Starting list of model initializations (e.g. "graph autoencoder").
    summaries : Dict[str, str]
        Mapping of {paper_title: bullet‑point string of methods}.
    task_description : str
        The task we are optimising for (used in the prompt).
    data_available : str
        Short description of what data the AutoML system sees (for context).
    temperature : float
        Sampling temperature for `run_prompt`.
    model : str
        Model name passed to `run_prompt`.

    Returns
    -------
    List[str]
        Final list of unique initializations, order preserved.
    """
    current = initializations.copy()

    for title, bullet_points in summaries.items():
        prompt = f"""
We are building an LLM‑powered AutoML system for the task:

    "{task_description}"

We will curate and refine our *model initializations* list using insights
from the following paper.

Current initializations:
{current}

Paper: "{title}"
Key method points:
{bullet_points}

TASK ▸
1. If the paper presents a **model family or architecture** not covered above,
   propose it as a concise initialization (≤ 6 words).
2. If two or more current initializations are effectively the same family,
   merge them by giving a single, clear name that subsumes them.
3. If the above are not met, or we can not implement the model using {data_available}, leave the list unchanged.

Return **one updated list only**—one initialization per line,
each wrapped exactly like <m>Initialization</m>.  No other text."""
        reply = run_prompt(prompt, client=client, temperature=temperature, model=model)

        # pull the LLM’s updated list
        matches = re.findall(r"<m>(.*?)</m>", reply, flags=re.S)
        print(matches)
        if matches:
            seen = set()
            current = [m.strip() for m in matches
                       if not (m.strip() in seen or seen.add(m.strip()))]

    # final safety de‑duplication
    final = []
    seen = set()
    for init in current:
        if init not in seen:
            final.append(init)
            seen.add(init)

    return final


