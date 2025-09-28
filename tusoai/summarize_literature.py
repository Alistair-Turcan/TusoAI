import re
from typing import List, Dict

WORD_LIMIT = 1200              # max words per paper
MIN_SECTION_WORDS = 100         # merge until section ≥ this

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

def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _merge_sections(sections: List[str]) -> List[str]:
    """
    Merge consecutive sections until each merged chunk has ≥ MIN_SECTION_WORDS.
    """
    merged, buf = [], []
    for sec in sections:
        buf.append(sec)
        if _word_count(" ".join(buf)) >= MIN_SECTION_WORDS:
            merged.append(" ".join(buf).strip())
            buf = []
    if buf:  # leftover
        merged.append(" ".join(buf).strip())
    return merged

def _init_description(abstract: str, client, TEMP, LLM_MODEL) -> str:
    prompt = f"""
You are a scientific summariser. Draft a concise yet technically accurate
description of the paper's method based **only** on the abstract below, to the extent possible. Capture the main points using bullets points. Do not waste words on complete sentences or details irrelevant to the methods.

Abstract:
\"\"\"{abstract}\"\"\"
"""
    return run_prompt(prompt, client, temperature=TEMP, model=LLM_MODEL)

def _update_description(current_desc: str, new_text: str, bp_limit, client, TEMP, LLM_MODEL) -> str:
    prompt = f"""
The current method description ):

\"\"\"{current_desc}\"\"\"

New excerpt from the paper:
\"\"\"{new_text}\"\"\"

Update the description by **incorporating any new technical details or correcting
existing ones** found in the excerpt. Preserve conciseness and clarity. Return **only**
the revised description. Capture the main points using bullets points.
Do not waste words on complete sentences or details irrelevant to the methods.
Do not exceed {bp_limit} bullet points.
"""
    return run_prompt(prompt, client, temperature=TEMP, model=LLM_MODEL)

def summarise_papers(papers: List[Dict], client, LLM_MODEL: str, TEMP=0.5, bp_limit=15) -> Dict[str, str]:
    summaries = {}
    for paper in papers:
        if not paper.get("abstract"):
            continue

        print(f"[PROCESS] {paper['title']}")
        desc = _init_description(paper["abstract"], client, TEMP, LLM_MODEL)
        print(desc)
        words = _word_count(desc)

        # merge short sections first
        merged_sections = _merge_sections(paper["sections"])

        for chunk in merged_sections:
            if words >= WORD_LIMIT:
                break
            desc = _update_description(desc, chunk, bp_limit, client, TEMP, LLM_MODEL)
            print(desc)
            words+= _word_count(desc)
            print(words)
        summaries[paper["title"]] = desc
        print(f"  → final length: {words} words")

    return summaries

