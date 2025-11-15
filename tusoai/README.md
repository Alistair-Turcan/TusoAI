## TusoAI Source Code Overview

- **construct_categories.py** — Generates and refines instruction categories.
- **construct_initializations.py** — Generates and refines initial solution descriptions.
- **construct_prompts.py** — Generates and refines instructions within categories.
- **diagnostic_prompts.json** — Contains the default diagnostic prompts.
- **extract_literature.py** — Extracts and parses literature from Semantic Scholar.
- **generic_prompts.json** — Contains general prompts used in ablation studies.
- **initialize_llm.py** — Provides generic LLM client initializations.
- **optimizer.py** — Implements core TusoAI logic including generating, refining, debugging, and optimizing solutions.
- **summarize_literature.py** — Summarizes extracted Semantic Scholar literature into concise technical summaries.
- **tusoai.py** — Main class coordinating calls to other modules.
